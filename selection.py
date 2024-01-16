import numpy as np
import random
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from irt import *
from utils import *

def get_seen_unseen_items(scenarios_choosen, scenarios, number_item, subscenarios_position, responses_test):
    
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
    # Iterate through each chosen scenario to determine the seen items.
    for scenario in scenarios_choosen:
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

    return seen_items, unseen_items

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

def get_disc_items(responses_train, number_items, chosen_scenarios, rows_to_hide_str, scenarios_position, device, bench):
    # First part: Dataset creation and model training
    for scenario in tqdm(chosen_scenarios):
        dataset_name = f'data/{bench}/rows-{rows_to_hide_str}_scenario-{scenario}_IRT-1D.jsonlines'
        create_irt_dataset(responses_train[:,scenarios_position[scenario]], dataset_name)
        
        model_name = f'models/{bench}/model-2pl_rows-{rows_to_hide_str}_D-1_scenario-{scenario}_IRT-1D/'
        with SuppressPrints(): os.system(f"py-irt train '2pl' {dataset_name} {model_name} --device {device} --priors 'vague' --seed 42 --deterministic --log-every 2000")
    
    # Second part: Extracting and processing parameters
    seen_items = {}
    for number_item in number_items:
        seen_items[number_item] = []
        for scenario in chosen_scenarios:
            model_name = f'models/{bench}/model-2pl_rows-{rows_to_hide_str}_D-1_scenario-{scenario}_IRT-1D/'
            with open(model_name + 'best_parameters.json') as f:
                A = np.array(json.load(f)['disc'])
            
            seen_items[number_item] += list(np.array(scenarios_position[scenario])[np.argsort(-A)[:number_item]])
    
    return seen_items