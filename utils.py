import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os

def create_sublists_corrected(numbers, s):
    # Create sublists of size s
    sublists = [numbers[i:i + s] for i in range(0, len(numbers), s)]
    return sublists

def clip(x, x_min=10):
    return (x>x_min)*x + (x<=x_min)*x_min

def item_curve(theta, a, b):
    z = np.clip(a*theta - b, -30, 30).sum(axis=1)
    return 1/(1+np.exp(-z))

# Function to find group ranges based on repeating labels
def find_group_ranges(labels):
    group_ranges = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            group_ranges.append((start, i-1))
            start = i
    group_ranges.append((start, len(labels)-1))  # Add the last group
    return group_ranges

class SuppressPrints:
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


def prepare_data(scenarios_choosen, scenarios, data):
    i = 0
    subscenarios_position = {}
    for scenario in scenarios_choosen:
        subscenarios_position[scenario] = {}
        for sub in scenarios[scenario]:
            subscenarios_position[scenario][sub] = []
            for j in range(data['data'][sub]['correctness'].shape[0]):
                subscenarios_position[scenario][sub].append(i)
                i += 1
    scenarios_position = {}
    for scenario in scenarios_choosen:
        scenarios_position[scenario] = []
        for key in subscenarios_position[scenario].keys():
            scenarios_position[scenario] += subscenarios_position[scenario][key]
    return scenarios_position, subscenarios_position

def create_responses(scenarios_choosen, scenarios, data):
    responses = [np.vstack([data['data'][sub]['correctness'] for sub in scenarios[scenario]]).T for scenario in scenarios_choosen]
    responses = np.hstack(responses)
    return responses


def sigmoid(z):
    return 1/(1+np.exp(-z))


def anchor_accs(new_theta, A, B, seen_items, scenario, scenarios_position, regs, kmedoids, anchor_weights):
    
    eps=1e-50
    P_new = item_curve(new_theta.reshape(1, -1, 1), A[:, :, seen_items], B[:, :, seen_items])
    X_new = np.log((P_new + eps) / (1 - P_new + eps))
           
    # Method 1
    y_hat = []
    for j in range(A.shape[-1]):
        if j in scenarios_position[scenario]:
            y_hat.append(sigmoid(regs[j].predict(X_new[:, kmedoids.labels_[j]].reshape(1, -1))[0]))
    out1 = np.mean([sigmoid(y) for y in y_hat])
    
    # Method 2
    #ind = [j for j in range(A.shape[-1]) if j in scenarios_position[scenario]]
    #print(X_new.shape, P_new.shape)
    #P_new = P_new[ind]
    #X_new = X_new[ind]   
    #anchor_weights = anchor_weights[ind]
    #anchor_weights /= np.sum(anchor_weights)
           
    out2 = sigmoid(anchor_weights @ np.log((P_new + eps) / (1 - P_new + eps)).squeeze())
    
    return out1, out2    

def create_space_accs_results(accs, results, row_to_hide, number_items, scenarios_choosen):
    accs[row_to_hide] = {}
    results[row_to_hide] = {}

    for number_item in number_items:
        accs[row_to_hide][number_item] = {
            'random_naive': {},
            'random_irt': {},
            'adaptive_irt': {},
            #'anchor_naive': {},
            #'anchor_irt': {},
            #'anchor_method1': {},
            #'anchor_method2': {},
        }
        results[row_to_hide][number_item] = {
            'random_naive': {},
            'random_irt': {},
            'adaptive_irt': {},
            #'anchor_naive': {},
            #'anchor_irt': {},
            #'anchor_method1': {},
            #'anchor_method2': {},
        }

        for scenario in scenarios_choosen:
            accs[row_to_hide][number_item]['random_naive'][scenario] = []
            accs[row_to_hide][number_item]['random_irt'][scenario] = []
            accs[row_to_hide][number_item]['adaptive_irt'][scenario] = []
            #accs[row_to_hide][number_item]['anchor_naive'][scenario] = []
            #accs[row_to_hide][number_item]['anchor_irt'][scenario] = []
            #accs[row_to_hide][number_item]['anchor_method1'][scenario] = []
            #accs[row_to_hide][number_item]['anchor_method2'][scenario] = []

                
def update_accs_naive(key, responses_test, row_to_hide, scenarios_choosen, scenarios_position, seen_items, accs, number_item):
    for scenario in scenarios_choosen:
        # Calculate the mean of the responses for the seen items in the current scenario
        mean_responses = responses_test[[s for s in seen_items if s in scenarios_position[scenario]]].mean()
        # Update the accs dictionary for random naive scenario
        accs[row_to_hide][number_item][key][scenario].append(mean_responses)
        
def update_accs_irt(key, scores_test, responses_test, row_to_hide, scenarios_choosen, scenarios_position, seen_items, unseen_items, A, B, new_theta, accs, number_item):
    D = A.shape[1]
    for scenario in scenarios_choosen:
        seen_responses = scores_test[[s for s in seen_items if s in scenarios_position[scenario]]]
        alpha = seen_responses.shape[0]/len(scenarios_position[scenario])
        if alpha == 0:first_part = 0
        else:first_part = alpha * seen_responses.mean()
        second_part = (1 - alpha) * item_curve(new_theta.reshape(1, D, 1), A, B)[0, [u for u in unseen_items if u in scenarios_position[scenario]]].mean()
        accs[row_to_hide][number_item][key][scenario].append(first_part + second_part)
 
def update_results(key, scores_test, row_to_hide, scenarios_choosen, scenarios_position, accs, results, number_item):
    for scenario in scenarios_choosen:
        # Calculate the mean absolute difference between the computed accuracies and the mean of the responses
        abs_diff = np.abs(np.array(accs[row_to_hide][number_item][key][scenario]) - scores_test[scenarios_position[scenario]].mean())
        # Update the results dictionary with the mean of the absolute differences
        results[row_to_hide][number_item][key][scenario] = abs_diff #.mean()

def plot_results(results, scenarios_choosen, number_items, metric, scenario_name, typ, methods = ['random_naive', 'random_irt', 'adaptive_irt']):
    models = list(results.keys())
    groups = methods #groups = list(results[0][number_items[0]].keys())
    irt_model = 'pl2_mirt'
    colors = ['green', 'blue', 'red', 'yellow']

    results2 = {}
    for scenario in scenarios_choosen:
        results2[scenario] = {}
        for group in groups:
            results2[scenario][group] = np.array([[results[model][number_item][group][scenario] for model in models] for number_item in number_items]).T

    for scenario in scenarios_choosen:
        plt.figure(figsize=(5, 3))
        for i in range(len(number_items)):
            positions = [i - 0.2, i, i + 0.2]
            for pos, d, color in zip(positions, [np.array(results2[scenario][group])[:, :, i] for group in groups], colors):
                plt.boxplot(d.reshape(-1), positions=[pos], patch_artist=True, boxprops=dict(facecolor=color))
        plt.title(scenario+', model='+irt_model+', metric='+str(metric))
        plt.xlabel("Number of seen items")
        plt.ylabel("Absolute error (acc estimation)")
        plt.ylim(0, 0.5)
        plt.grid(alpha=.5)
        xticks = [i for i in range(len(number_items))]
        plt.xticks(xticks, [str(number_item) for number_item in number_items])
        legend_elements = [Patch(facecolor=color, label=group) for group, color in zip(groups, colors)]
        plt.legend(handles=legend_elements)
        plt.savefig(f'plots/boxplot_metric-{metric}_scenario-{scenario}-{typ}.png', bbox_inches='tight', dpi=300, transparent=True)
        plt.show()
        
        
def plot_results2(results, number_items, methods):

    irt_model = 'pl2_mirt'
    colors = ['green', 'blue', 'red', 'yellow']

    results2 = {}
    for method in methods:
        results2[method] = np.array([results[method][number_item] for number_item in number_items]).T


    for i in range(len(number_items)):
        positions = [i - 0.2, i, i + 0.2]
        for pos, d, color in zip(positions, [np.array(results2[method])[:, i] for method in methods], colors):
            plt.boxplot(d, positions=[pos], patch_artist=True, showfliers=False, showmeans=True, boxprops=dict(facecolor=color))
    plt.xlabel("Number of seen items (per scenario)")
    plt.ylabel("Absolute error (acc estimation)")
    plt.ylim(0, 0.3)
    plt.grid(alpha=.5)
    xticks = [i for i in range(len(number_items))]
    plt.xticks(xticks, [str(number_item) for number_item in number_items])
    legend_elements = [Patch(facecolor=color, label=group) for group, color in zip(methods, colors)]
    plt.legend(handles=legend_elements)
    #plt.savefig(f'plots/boxplot_metric-{metric}_scenario-{scenario}-{typ}.png', bbox_inches='tight', dpi=300, transparent=True)
    plt.show()