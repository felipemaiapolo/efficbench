import numpy as np
import random
from sklearn_extra.cluster import KMedoids
from scipy.integrate import nquad, quad
from scipy.stats import norm
from tqdm import tqdm
from irt import *
from utils import *

def get_random(scenarios_choosen, scenarios, number_item, subscenarios_position, responses_test, random_seed):
    
    """
    Stratified sample items (seen_items). 'unseen_intems' gives the complement.
    
    Parameters:
    - scenarios_choosen: A list of considered scenarios.
    - scenarios: A dictionary where keys are scenario identifiers and values are lists of subscenarios.
    - number_item: The total number of items to be considered across all chosen scenarios.
    - subscenarios_position: A nested dictionary where the first key is the scenario and the second key is the subscenario, 
      and the value is a list of item positions for that subscenario.
    - responses_test: A numpy array of the test subject's responses to all items. (this is only used to get the number of items)
    
    Returns:
    - seen_items: A list of item indices that the subject has been exposed to.
    - unseen_items: A list of item indices that the subject has not been exposed to.
    """
    
    random.seed(random_seed)
    
    def shuffle_list(lista):
        """
        Shuffles a list in place and returns the shuffled list.
        
        Parameters:
        - lista: The list to be shuffled.
        
        Returns:
        - A shuffled version of the input list.
        """
        return random.sample(lista, len(lista))

    
    seen_items = []  # Initialize an empty list to hold the indices of seen items.
    item_weights = {}
    
    # Iterate through each chosen scenario to determine the seen items.
    for scenario in scenarios_choosen:
        
        item_weights[scenario] = np.ones(number_item)/number_item
        
        # Allocate the number of items to be seen in each subscenario.
        number_items_sub = np.zeros(len(scenarios[scenario])).astype(int)
        number_items_sub += number_item // len(scenarios[scenario])
        number_items_sub[:(number_item - number_items_sub.sum())] += 1
        
        i = 0  # Initialize a counter for the subscenarios.
        # Shuffle the subscenarios and iterate through them to select seen items.
        for sub in shuffle_list(scenarios[scenario]):
            # Randomly select items from the subscenario and add them to the seen items.
            seen_items += random.sample(subscenarios_position[scenario][sub], k=number_items_sub[i])
            i += 1

    # Determine the unseen items by finding all item indices that are not in the seen items list.
    unseen_items = [i for i in range(responses_test.shape[1]) if i not in seen_items]

    return item_weights, seen_items, unseen_items


def select_initial_adaptive_items(A, B, Theta, number_item, try_size=2000, seed=42):
    random.seed(seed)
    mats = np.stack([np.outer(A[0, :, i], A[0, :, i]) for i in range(A.shape[2])])
    samples = [random.sample(range(A.shape[-1]), number_item) for _ in range(try_size)]
    samples_infos = np.stack([np.linalg.det(np.array([(p * (1 - p))[:, None, None] * mats[s] for p in item_curve(Theta, A[:, :, s], B[:, :, s])]).sum(axis=1)).sum() for s in samples])
    seen_items = samples[np.argmax(samples_infos)]
    unseen_items = [i for i in range(A.shape[-1]) if i not in seen_items]
    return seen_items, unseen_items, mats


def run_adaptive_selection(responses_test, seen_items, unseen_items, scenarios_choosen, scenarios_position, A, B, mats, target_count, balance=False, ki=False):
    
    count = 0
    scenario_counts = {scenario: 0 for scenario in scenarios_choosen}
    while True:
        for scenario in scenarios_choosen:
            scenario_counts[scenario] += 1
            if count >= target_count:
                item_weights = {scenario: np.ones(final_count)/final_count for scenario, final_count in scenario_counts.items()}
                return item_weights, seen_items, unseen_items 
            
            if not ki:
                seen_items, unseen_items = select_next_adaptive_item(responses_test, seen_items, unseen_items, scenario, scenarios_position, A, B, mats, balance)
            else:
                seen_items, unseen_items = select_next_adaptive_item_KI(responses_test, seen_items, unseen_items, scenario, scenarios_position, A, B, mats, balance)

            count += 1

def select_next_adaptive_item(responses_test, seen_items, unseen_items, scenario, scenarios_position, A, B, mats, balance):
    
    D = A.shape[1]

    if balance:
        unseen_items_scenario = [u for u in unseen_items if u in scenarios_position[scenario]]
    else:
        unseen_items_scenario = unseen_items

    optimal_theta = estimate_ability_parameters(responses_test, seen_items, A, B)
    P = item_curve(optimal_theta, A, B).squeeze()

    # Compute information matrices for seen and unseen items
    I_seen = ((P * (1 - P))[:, None, None] * mats)[seen_items].sum(axis=0)
    I_unseen = ((P * (1 - P))[:, None, None] * mats)[unseen_items_scenario]

    # Select the next item based on the maximum determinant of information
    next_item = unseen_items_scenario[np.argmax(np.linalg.det(I_seen[None, :, :] + I_unseen))]                    
    seen_items.append(next_item)
    unseen_items.remove(next_item)

    return seen_items, unseen_items

def select_next_adaptive_item_KI(responses_test, 
                                 seen_items, 
                                 unseen_items, 
                                 scenario, 
                                 scenarios_position, 
                                 A, B, mats, 
                                 balance):

    # Define the KL divergence for a single item. (Equation 7)
    def kl_divergence(theta_0, theta_hat, xj, a, b):
        f_theta_0 = item_response_function(xj, theta_0, a, b)
        f_theta_hat = item_response_function(xj, theta_hat, a, b)
        if f_theta_0 > 0 and f_theta_hat > 0:
            return f_theta_0 * np.log(f_theta_0 / f_theta_hat)
        else:
            return 0

    # Define the bounds for the integration (as in Equation 8/9)
    def integration_bounds(theta_p0, k, r):
        lower_bound = theta_p0 - r / np.sqrt(k)
        upper_bound = theta_p0 + r / np.sqrt(k)
        return lower_bound, upper_bound

    # Integrate the KL divergence over the p-dimensional space.
    def multivariate_ki(theta_hat, k, a, b, xj, r=3):
        """ k -> number of seen samples  
            r -> some constant usually set to 3
            xj -> binary item response
            """
        
        # Define the limits for each dimension.
        #limits = [integration_bounds(th, k, r) for th in theta.squeeze()]
        limits = [integration_bounds(theta_hat, k, r)]

        def integrand(theta_0, theta_hat, xj, a, b):
            return kl_divergence(theta_0, theta_hat, xj, a, b)

        ki, _ = quad(integrand, limits[0][0], limits[0][1], args=(theta_hat, xj, a, b))
        #ki, _ = nquad(integrand, limits)

        return ki

    D = A.shape[1]

    if balance:
        unseen_items_scenario = [u for u in unseen_items if u in scenarios_position[scenario]]
    else:
        unseen_items_scenario = unseen_items

    optimal_theta = estimate_ability_parameters(responses_test, seen_items, A, B)
    
    ki_values = []
    for unseen_item in unseen_items_scenario:
        item_response = responses_test[unseen_item]

        a = A[:, :, [unseen_item]]
        b = B[:, :, [unseen_item]]

        ki_value = multivariate_ki(optimal_theta, k=len(seen_items),
                                   a=a, b=b, xj=item_response,
                                   r=3)

        ki_values.append(ki_value)
    
    '''
    # batched: looks like you cannot do batched integration. 
    # However, parallel processing (i.e. multithreading) might be possible

    item_response = responses_test[unseen_items_scenario]

    a = A[:, :, unseen_items_scenario]
    b = B[:, :, unseen_items_scenario]

    ki_values = multivariate_ki(optimal_theta, k=len(seen_items),
                                   a=a, b=b, xj=item_response,
                                   r=8)
    '''

    next_item = unseen_items_scenario[np.argmax(ki_values)]

    seen_items.append(next_item)
    unseen_items.remove(next_item)

    return seen_items, unseen_items


def get_gpirt_weighing(seen_items: list,
                       unseen_items: list, 
                       scenarios_position: list,
                       chosen_scenarios: list,
                       A: np.array,
                       B: np.array,
                       ) -> list:
    """
    Gets weights for unseen items based on proximity in IRT parameter space

    """
    by_scenario = True
    weights = {}

    IRT_params = np.concatenate((A, B), axis=1)

    if by_scenario:
        for i, scenario in enumerate(chosen_scenarios):
            # select indices of current scenario
            scenario_idxs = scenarios_position[scenario]

            scenario_seen_items = [s for s in seen_items if s in scenarios_position[scenario]]

            weights[scenario] = get_weights(IRT_params, 
                                            scenario_seen_items,
                                            scenario_idxs)
    else:
        weights['all'] = get_weights(IRT_params, 
                                     seen_items,
                                     unseen_items)

    return weights

def get_weights(IRT_params: np.array,
                seen_items: list,
                all_items: list):
    """
    Assign each item a seen item based on distance in IRT-parameter space.
    Calculate weights for each seen item based on other items assigned to it.
    """
    # Prepare parameter space
    params_seen = IRT_params[:, :, seen_items]
    params_all = IRT_params[:, :, all_items]

    # Prepare an array to hold the indices of the closest points
    closest_indices = np.zeros(params_all.shape[2], dtype=int)

    # Iterate over each unseen item
    for i in range(params_all.shape[2]):
        # Calculate the Euclidean distances to all seen items
        distances = np.linalg.norm(params_seen - params_all[:, :, i:i+1], axis=1)

        # Find the index of the closest seen item
        closest_indices[i] = np.argmin(distances)
    
    # Count the frequency of each index in closest_indices
    frequency = np.bincount(closest_indices, minlength=params_seen.shape[2])

    # Normalize the frequencies to get weights
    return frequency / np.sum(frequency)

def get_anchor(scores_train, chosen_scenarios, scenarios_position, number_item, random_seed):
    """
    Calculates anchor points, anchor weights, seen and unseen items.

    Parameters:
    scores_train (array): The training scores.
    chosen_scenarios: List of considered scenarios.
    scenarios_position (dict): Positions of the scenarios.
    number_items (list): List containing numbers of clusters/items.

    Returns:
    tuple: A tuple containing anchor points, anchor weights, seen items, and unseen items.
    """
    anchor_points = {}
    anchor_weights = {}
    for scenario in chosen_scenarios:
        anchor_points[scenario] = {}
        anchor_weights[scenario] = {}
        anchor_points[scenario], anchor_weights[scenario] = get_anchor_points_weights(scores_train, scenarios_position, scenario, number_item, random_seed)
    
    seen_items = [list(np.array(scenarios_position[scenario])[anchor_points[scenario]]) for scenario in chosen_scenarios]
    seen_items = list(np.array(seen_items).reshape(-1))
    unseen_items = [i for i in range(scores_train.shape[1]) if i not in seen_items]
    
    return anchor_points, anchor_weights, seen_items, unseen_items


def get_anchor_points_weights(scores_train, scenarios_position, scenario, number_item, random_seed, trials = 10):

    """
    Calculates anchor points and weights using KMedoids clustering.

    Parameters:
    scores_train (array): The training scores.
    scenarios_position (list): Positions of the scenarios.
    scenario (int): The specific scenario index.
    number_item (int): Number of clusters/items.

    Returns:
    tuple: A tuple containing the anchor points and anchor weights.
    """
    # Fitting the KMedoids model
    kmedoids_models = [KMedoids(n_clusters=number_item, metric='euclidean', init='k-medoids++', random_state=1000*t+random_seed).fit(scores_train[:,scenarios_position[scenario]].T) for t in range(trials)] #method='pam', 

    kmedoids = kmedoids_models[np.argmin([m.inertia_ for m in kmedoids_models])]
    
    # Calculating anchor points
    anchor_points = kmedoids.medoid_indices_
    
    # Calculating anchor weights
    anchor_weights = np.array([np.mean(kmedoids.labels_ == anchor) for anchor in range(number_item)])
    anchor_weights /= anchor_weights.sum()
    
    return anchor_points, anchor_weights

def sample_items(number_item, iterations, sampling_name, chosen_scenarios, scenarios, subscenarios_position, responses_test, scores_train, scenarios_position, A, B, inital_items=None):
    item_weights_dic, seen_items_dic, unseen_items_dic = {}, {}, {}

    for it in range(iterations):
        if sampling_name == 'random':
            item_weights, seen_items, unseen_items = get_random(chosen_scenarios, scenarios, number_item, subscenarios_position, responses_test, random_seed=it)

        elif sampling_name == 'anchor':
            _, item_weights, seen_items, unseen_items = get_anchor(scores_train, chosen_scenarios, scenarios_position, number_item, random_seed=it)

        elif sampling_name == 'anchor-irt':
            _, item_weights, seen_items, unseen_items = get_anchor(np.vstack((A.squeeze(), B.reshape((1,-1)))), chosen_scenarios, scenarios_position, number_item, random_seed=it)
        elif sampling_name == 'adaptive':
            continue

        item_weights_dic[it], seen_items_dic[it], unseen_items_dic[it] = item_weights, seen_items, unseen_items

    if sampling_name == 'adaptive':
        seen_items, unseen_items, mats = inital_items
        for n_model, responses in enumerate(responses_test):
            item_weights_dic[n_model], seen_items_dic[n_model], unseen_items_dic[n_model] = run_adaptive_selection(responses, seen_items, unseen_items, 
                                                                                                                    chosen_scenarios, 
                                                                                                                    scenarios_position, A, B, 
                                                                                                                    mats, number_item, 
                                                                                                                    balance=False, ki=False)

    return item_weights_dic, seen_items_dic, unseen_items_dic

