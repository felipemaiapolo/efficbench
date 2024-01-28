from irt import *
from utils import *

def compute_acc_pirt(scenario, scores_test, scenarios_position, seen_items, unseen_items, A, B, theta, balance_weights, thresh=None):
    
    """
    Compute the PIRT or CIRT
    
    Parameters:
    - scenario: The scenario being considered.
    - scores_test: The test scores for the scenario.
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - seen_items: A list of item indices that the subject has been exposed to.
    - unseen_items: A list of item indices that the subject has not been exposed to.
    - A: The discrimination parameter of the item.
    - B: The difficulty parameter of the item.
    - theta: The ability parameter of the subject.
    - balance_weights: balancing weights (mmlu/civil comments).
    - thresh: classification threshold for CIRT (if None, PIRT will be computed).
    
    Returns:
    - The computed accuracy for the scenario.
    """
    
    # Extract the responses for the seen items in the scenario
    seen_responses = scores_test[[s for s in seen_items if s in scenarios_position[scenario]]]
    
    # Determine the weighting parameter 
    lambd = seen_responses.shape[0]/len(scenarios_position[scenario])
    
    # Compute the first part of the accuracy equation based on seen items
    if lambd == 0: data_part = 0
    else: data_part = seen_responses.mean()

    # Compute the second part of the accuracy equation based on unseen items (and IRT model) 
    D = A.shape[1] # The number of dimensions in the IRT model
    if thresh==None:
        irt_part = (balance_weights*item_curve(theta.reshape(1, D, 1), A, B))[0, [u for u in unseen_items if u in scenarios_position[scenario]]].mean()
    else:
        irt_part = (balance_weights*(item_curve(theta.reshape(1, D, 1), A, B)>=thresh).astype(float))[0, [u for u in unseen_items if u in scenarios_position[scenario]]].mean()
    
    return lambd * data_part + (1 - lambd) * irt_part

def calculate_accuracies(j, sampling_names, item_weights_dic, seen_items_dic, unseen_items_dic, A, B, scores_test, responses_test, scenarios_position, chosen_scenarios, balance_weights, opt_lambds, rows_to_hide):

    number_items = list(item_weights_dic[sampling_names[0]].keys())
    
    
    # Creating output format
    accs = {rows_to_hide[j]: {}}
    for number_item in number_items:
        accs[rows_to_hide[j]][number_item] = {} 
        for est in ['naive', 'pirt', 'cirt', 'gpirt']:
            for sampling_name in sampling_names:
                accs[rows_to_hide[j]][number_item][sampling_name+'_'+est] = {}
                for scenario in chosen_scenarios:
                    accs[rows_to_hide[j]][number_item][sampling_name+'_'+est][scenario] = []

    # Populating output 
    for sampling_name in sampling_names:
        for number_item in number_items:
            if 'adaptive' in sampling_name:
                # Getting sample for specific model j
                item_weights, seen_items, unseen_items = item_weights_dic[sampling_name][number_item][j], seen_items_dic[sampling_name][number_item][j], unseen_items_dic[sampling_name][number_item][j]

                # Estimate ability parameters for the test set (IRT)
                new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                # Update accuracies 
                for scenario in chosen_scenarios:
                    naive = (item_weights[scenario]*scores_test[j][[s for s in seen_items if s in scenarios_position[scenario]]]).sum()
                    pirt = compute_acc_pirt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, thresh=None)
                    cirt = compute_acc_pirt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, thresh=0.5)
                    lambd = opt_lambds[sampling_name+'_gpirt'][scenario][number_item]

                    accs[rows_to_hide[j]][number_item][sampling_name+'_naive'][scenario].append(naive)
                    accs[rows_to_hide[j]][number_item][sampling_name+'_pirt'][scenario].append(pirt)
                    accs[rows_to_hide[j]][number_item][sampling_name+'_cirt'][scenario].append(cirt)
                    accs[rows_to_hide[j]][number_item][sampling_name+'_gpirt'][scenario].append(lambd*naive + (1-lambd)*pirt)
            else:
                iterations = len(list(item_weights_dic[sampling_name][number_items[0]].keys()))
                for it in range(iterations):
                    #print(sampling_name, number_item, it)
                    # Getting sample
                    try:
                        item_weights, seen_items, unseen_items = item_weights_dic[sampling_name][number_item][it], seen_items_dic[sampling_name][number_item][it], unseen_items_dic[sampling_name][number_item][it]
                    except:
                        breakpoint()
                    # Estimate ability parameters for the test set (IRT)
                    new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                    # Update accuracies 
                    for scenario in chosen_scenarios:
                        if sampling_name == 'random' and scenario=='mmlu': naive = (item_weights[scenario]*responses_test[j][[s for s in seen_items if s in scenarios_position[scenario]]]).sum()
                        #elif sampling_name == 'random' and scenario=='civil_comments': 
                        else: naive = (item_weights[scenario]*scores_test[j][[s for s in seen_items if s in scenarios_position[scenario]]]).sum()
                        pirt = compute_acc_pirt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, thresh=None)
                        cirt = compute_acc_pirt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, thresh=0.5)
                        lambd = opt_lambds[sampling_name+'_gpirt'][scenario][number_item]

                        accs[rows_to_hide[j]][number_item][sampling_name+'_naive'][scenario].append(naive)
                        accs[rows_to_hide[j]][number_item][sampling_name+'_pirt'][scenario].append(pirt)
                        accs[rows_to_hide[j]][number_item][sampling_name+'_cirt'][scenario].append(cirt)
                        accs[rows_to_hide[j]][number_item][sampling_name+'_gpirt'][scenario].append(lambd*naive + (1-lambd)*pirt)
    
    # Return output
    return accs