#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import autograd.numpy as np
from autograd import grad
from tqdm import tqdm

def get_lambda(number_item, b, v):
    return (number_item*b**2)/(v+(number_item*b**2))

def debias_irt(A, B, Theta, responses_train, max_iter=500, lr=100000, alpha=0.5, beta=0.9):
    E_initial = np.hstack((A,B))
    
    def neg_log_like(E, Theta=Theta, shape=E_initial.shape, responses_train=responses_train, eps=1e-20):
        E = E.reshape(shape)
        A, B = E[:, :-1, :], E[:, -1, :]
        P = item_curve(Theta, A, B).squeeze()
        log_likelihood = np.mean(responses_train * np.log(P + eps) + (1 - responses_train) * np.log(1 - P + eps))
        return -log_likelihood

    gradient = grad(neg_log_like)
    E = E_initial.copy()

    for it in tqdm(range(max_iter)):
        current_E = E.copy()
        grad_E = gradient(E.reshape(-1)).reshape(E.shape)
        initial_descent = -alpha * np.sum(grad_E**2)

        # Backtracking line search
        while neg_log_like(E - lr * grad_E) > neg_log_like(current_E) + lr * initial_descent:
            lr *= beta

        # Gradient update
        E -= lr * grad_E

        #if it % 50 == 0:
        #    print(neg_log_like(E))

    A, B = E[:, :-1, :], E[:, -1, :].reshape(1,1,-1)
    return A, B

class SuppressPrints:
    
    """
    A context manager to suppress prints to the console, useful for making output cleaner.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
def sigmoid(z):
    
    """
    Compute the sigmoid function for the input z.
    
    Parameters:
    - z: A numeric value or numpy array.
    
    Returns:
    - The sigmoid of z.
    """
    
    return 1/(1+np.exp(-z))

def item_curve(theta, a, b):
    
    """
    Compute the item response curve for given parameters.
    
    Parameters:
    - theta: The ability parameter of the subject.
    - a: The discrimination parameter of the item.
    - b: The difficulty parameter of the item.
    
    Returns:
    - The probability of a correct response given the item parameters and subject ability.
    """
    
    z = np.clip(a*theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)

def create_sublists_corrected(numbers, s):
    
    """
    Create sublists of a given size from the input list.
    
    Parameters:
    - numbers: The list to be divided into sublists.
    - s: The desired size of each sublist.
    
    Returns:
    - A list of sublists, each of size s.
    """
    
    # Create sublists of size s
    sublists = [numbers[i:i + s] for i in range(0, len(numbers), s)]
    return sublists

def prepare_data(chosen_scenarios, scenarios, data):
    
    """
    Prepare the data by determining the positions of items within each scenario and subscenario.
    
    Parameters:
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.
    
    Returns:
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - subscenarios_position: A nested dictionary mapping each scenario and subscenario to the positions of its items.
    """
    
    i = 0
    subscenarios_position = {}
    
    # Iterate through each chosen scenario and its subscenarios to record item positions
    for scenario in chosen_scenarios:
        subscenarios_position[scenario] = {}
        for sub in scenarios[scenario]:
            subscenarios_position[scenario][sub] = []
            for j in range(data['data'][sub]['correctness'].shape[0]):
                subscenarios_position[scenario][sub].append(i)
                i += 1
    
    # Prepare a simplified mapping of scenarios to their item positions
    scenarios_position = {}
    for scenario in chosen_scenarios:
        scenarios_position[scenario] = []
        for key in subscenarios_position[scenario].keys():
            scenarios_position[scenario] += subscenarios_position[scenario][key]
    return scenarios_position, subscenarios_position

def create_responses(chosen_scenarios, scenarios, data):
    
    """
    Create a matrix of responses for the chosen scenarios.
    
    Parameters:
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - data: A dictionary containing correctness data for each subscenario.
    
    Returns:
    - A numpy array of responses for the chosen scenarios.
    """
    
    responses = [np.vstack([data['data'][sub]['correctness'] for sub in scenarios[scenario]]).T for scenario in chosen_scenarios]
    responses = np.hstack(responses)
    return responses

def create_space_accs_results(accs, results, row_to_hide, number_items, chosen_scenarios):
    
    """
    Initialize dictionaries to hold accuracy and results data.
    
    Parameters:
    - accs: A dictionary to hold accuracy data.
    - results: A dictionary to hold result data.
    - row_to_hide: The index of the row to be hidden in the analysis.
    - number_items: A list of the number of items to consider.
    - chosen_scenarios: A list of scenarios to be considered.
    """
    
    accs[row_to_hide] = {}
    results[row_to_hide] = {}

    # Initialize dictionaries for each number of items and scenario
    for number_item in number_items:
        accs[row_to_hide][number_item] = {
            'random_naive': {},
            'random_cirt': {},
            'random_pirt': {},
            'random_gpirt': {},
            'anchor_naive': {},
            'anchor_cirt': {},
            'anchor_pirt': {},
            'anchor_gpirt': {},
            'anchor-irt_naive': {},
            'anchor-irt_cirt': {},
            'anchor-irt_pirt': {},
            'anchor-irt_gpirt': {},
            'disc_naive': {},
            'disc_cirt': {},
            'disc_pirt': {},
            'disc_gpirt': {}
        }
        
        
        results[row_to_hide][number_item] = {
            'random_naive': {},
            'random_cirt': {},
            'random_pirt': {},
            'random_gpirt': {},
            'anchor_naive': {},
            'anchor_cirt': {},
            'anchor_pirt': {},
            'anchor_gpirt': {},
            'anchor-irt_naive': {},
            'anchor-irt_cirt': {},
            'anchor-irt_pirt': {},
            'anchor-irt_gpirt': {},
            'disc_naive': {},
            'disc_cirt': {},
            'disc_pirt': {},
            'disc_gpirt': {}
        }

        for scenario in chosen_scenarios:
            accs[row_to_hide][number_item]['random_naive'][scenario] = []
            accs[row_to_hide][number_item]['random_cirt'][scenario] = []
            accs[row_to_hide][number_item]['random_pirt'][scenario] = []
            accs[row_to_hide][number_item]['random_gpirt'][scenario] = []
            accs[row_to_hide][number_item]['anchor_naive'][scenario] = []
            accs[row_to_hide][number_item]['anchor_cirt'][scenario] = []
            accs[row_to_hide][number_item]['anchor_pirt'][scenario] = []
            accs[row_to_hide][number_item]['anchor_gpirt'][scenario] = []
            accs[row_to_hide][number_item]['anchor-irt_naive'][scenario] = []
            accs[row_to_hide][number_item]['anchor-irt_cirt'][scenario] = []
            accs[row_to_hide][number_item]['anchor-irt_pirt'][scenario] = []
            accs[row_to_hide][number_item]['anchor-irt_gpirt'][scenario] = []
            accs[row_to_hide][number_item]['disc_naive'][scenario] = []
            accs[row_to_hide][number_item]['disc_cirt'][scenario] = []
            accs[row_to_hide][number_item]['disc_pirt'][scenario] = []
            accs[row_to_hide][number_item]['disc_gpirt'][scenario] = []

                
def compute_acc_irt(scenario, scores_test, scenarios_position, seen_items, unseen_items, A, B, theta, balance_weights, lambd=None, item_weights=None, thresh=None):
    
    """
    Compute the PIRT or G-PIRT
    
    Parameters:
    - scenario: The scenario being considered.
    - scores_test: The test scores for the scenario.
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - seen_items: A list of item indices that the subject has been exposed to.
    - unseen_items: A list of item indices that the subject has not been exposed to.
    - A: The discrimination parameter of the item.
    - B: The difficulty parameter of the item.
    - theta: The ability parameter of the subject.
    - lambd: A weighting parameter between seen and unseen items (optional).
    
    Returns:
    - The computed accuracy for the scenario.
    """
    
    # Extract the responses for the seen items in the scenario
    seen_responses = scores_test[[s for s in seen_items if s in scenarios_position[scenario]]]
    
    # Weighting
    if type(item_weights)==np.ndarray:
        assert item_weights.shape == seen_responses.shape
        assert np.sum(item_weights>=0)
        assert np.round(np.sum(item_weights),4) == 1
    else:
        item_weights = np.ones(seen_responses.shape)
        item_weights /= item_weights.sum()
    
    
    # Determine the weighting parameter if not provided (PIRT case)
    if lambd == None: lambd = np.round(seen_responses.shape[0]/len(scenarios_position[scenario]),2)
    
    D = A.shape[1] # The number of dimensions in the IRT model
    
    # Compute the first part of the accuracy equation based on seen items
    if lambd == 0: first_part = 0
    else: first_part = lambd * (item_weights*seen_responses).sum()

    # Compute the second part of the accuracy equation based on unseen items (and IRT model)
    if thresh==None:
        second_part = (1 - lambd) * (balance_weights*item_curve(theta.reshape(1, D, 1), A, B))[0, [u for u in unseen_items if u in scenarios_position[scenario]]].mean()
    else:
        second_part = (1 - lambd) * (balance_weights*(item_curve(theta.reshape(1, D, 1), A, B)>=thresh).astype(float))[0, [u for u in unseen_items if u in scenarios_position[scenario]]].mean()
    return first_part + second_part


def update_results(key, scores_test, row_to_hide, chosen_scenarios, scenarios_position, accs, results, number_item):
    
    """
    Update the results dictionary with the mean absolute difference between the computed accuracies and the actual scores.
    
    Parameters:
    - key: The key indicating the type of accuracy being updated (e.g., 'random_naive', 'random_pirt', or 'random_gpirt').
    - scores_test: A numpy array of the test subject's responses to all items.
    - row_to_hide: The index of the row to be hidden in the analysis.
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - accs: A nested dictionary containing computed accuracies for various scenarios and items.
    - results: A nested dictionary where the updated results will be stored.
    - number_item: The number of items to consider in the current analysis.
    """
    
    # Iterate through each chosen scenario to update results
    for scenario in chosen_scenarios:
        
        # Calculate the mean absolute difference between the computed accuracies and the mean of the responses
        # for the current scenario and the specified number of items.
        abs_diff = np.abs(np.array(accs[row_to_hide][number_item][key][scenario]) - scores_test[scenarios_position[scenario]].mean())
        
        # Update the results dictionary with the mean of the absolute differences for the current scenario, 
        # the specified number of items, and the specified key.
        results[row_to_hide][number_item][key][scenario] = abs_diff 


def plot_results(results, scenarios, methods):
    ###
    models = list(results.keys())
    example_key1 = list(results.keys())[0]
    number_items = list(results[example_key1].keys())
    example_key2 = number_items[0]
    #methods = list(results[example_key1][example_key2].keys())

    ###
    colors = ['red', 'yellow', 'green', 'blue']
    
    results2 = {}
    for scenario in scenarios:
        results2[scenario] = {}
        for group in methods:
            results2[scenario][group] = np.array([[results[model][number_item][group][scenario] for model in models] for number_item in number_items]).T

    for scenario in scenarios:
        plt.figure(figsize=(5, 3))
        for i in range(len(number_items)):
            positions = [i - 0.3, i - 0.1, i + 0.1, i + 0.3]
            for pos, d, color in zip(positions, [np.array(results2[scenario][group])[:, :, i] for group in methods], colors):
                plt.boxplot(d.reshape(-1), positions=[pos], patch_artist=True, showfliers=False, showmeans=True, boxprops=dict(facecolor=color))
        plt.title(scenario)
        plt.xlabel("Number of seen items")
        plt.ylabel("Absolute error (acc estimation)")
        plt.ylim(0, 0.25)
        plt.grid(alpha=.5)
        xticks = [i for i in range(len(number_items))]
        plt.xticks(xticks, [str(number_item) for number_item in number_items])
        legend_elements = [Patch(facecolor=color, label=group) for group, color in zip(methods, colors)]
        plt.legend(handles=legend_elements)
        #plt.savefig(f'plots/boxplot_metric-{metric}_scenario-{scenario}-{typ}.png', bbox_inches='tight', dpi=300, transparent=True)
        plt.show()
        
def plot_agg_results(results, scenarios, methods):
    ###
    example_key1 = list(results.keys())[0]
    number_items = list(results[example_key1].keys())
    example_key2 = number_items[0]
    #methods = list(results[example_key1][example_key2].keys())
    
    ###
    agg_results = {}
    for scenario in scenarios: 
        for method in methods:
            agg_results[method] = {}
            for number_item in number_items:
                agg_results[method][number_item] = []

    for scenario in scenarios: 
        for method in methods:
            for number_item in number_items:
                agg_results[method][number_item]+=[np.mean([results[model][number_item][method][scenario] for model in results.keys()])]

    ###
    colors = ['red', 'yellow', 'green', 'blue']
    results2 = {}
    for method in methods:
        results2[method] = np.array([agg_results[method][number_item] for number_item in number_items]).T

    for i in range(len(number_items)):
        positions = [i - 0.3, i - 0.1, i + 0.1, i + 0.3]
        for pos, d, color in zip(positions, [np.array(results2[method])[:, i] for method in methods], colors):
            plt.boxplot(d, positions=[pos], patch_artist=True, showfliers=False, showmeans=True, boxprops=dict(facecolor=color))
    plt.title("Aggregated results")
    plt.xlabel("Number of seen items (per scenario)")
    plt.ylabel("Absolute error (acc estimation)")
    plt.ylim(0, 0.15)
    plt.grid(alpha=.5)
    xticks = [i for i in range(len(number_items))]
    plt.xticks(xticks, [str(number_item) for number_item in number_items])
    legend_elements = [Patch(facecolor=color, label=group) for group, color in zip(methods, colors)]
    plt.legend(handles=legend_elements)
    #plt.savefig(f'plots/boxplot_metric-{metric}_scenario-{scenario}-{typ}.png', bbox_inches='tight', dpi=300, transparent=True)
    plt.show()