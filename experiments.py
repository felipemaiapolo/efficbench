from tqdm import tqdm
import multiprocessing as mp
import time
from irt import *
from selection import *
from utils import *
from acc import *

def evaluate_scenarios(data, scenario_name, chosen_scenarios, 
                       scenarios, set_of_rows, Ds, iterations, device, bench, 
                       sampling_names = ['random', 'anchor', 'anchor-irt']):


    """
    Evaluates scenarios by training and validating IRT models, then computing accuracies and updating results.
    
    Parameters:
    - data: A dictionary containing the dataset.
    - scenario_name: The name of the current scenario.
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - set_of_rows: A set of row indices to hide during training (to simulate missing data).
    - Ds: A list of dimension values to consider for the IRT model.
    - iterations: The number of iterations to perform for random evaluations.
    - device: The computing device ('cpu' or 'gpu') to use for training.
    
    Returns:
    - A dictionary containing the updated results.
    """
    
    assert bench in ['irt_helm', 'irt_lb', 'irt_lb_perf', 'irt_mmlu']
    assert np.mean([s in ['random', 'anchor', 'anchor-irt'] for s in sampling_names]) == 1
    
    number_items = [10, 25, 50, 75, 100, 150]  # Number of items to consider in evaluations

    cpu = mp.cpu_count()  # Number of available CPU cores
    epochs = 2000  # Number of epochs for IRT model training (package default is 2000)
    lr = .1  # Learning rate for IRT model training (package default is .1)

    # Iterate through each set of rows to hide
    accs_true = {}  # Initialize a dictionary to hold real accuracies
    out = [] # To store intermediate results
    for rows_to_hide in set_of_rows:
        rows_to_hide_str = ':'.join([str(r) for r in rows_to_hide])[:30] + ':'.join([str(r) for r in rows_to_hide])[-30:]
   
        print(f"\nEvaluating models {rows_to_hide}")
        
        # Prepare data and scenarios
        scenarios_position, subscenarios_position = prepare_data(chosen_scenarios, scenarios, data)
        scores = create_responses(chosen_scenarios, scenarios, data)
        
        balance_weights = np.ones(scores.shape[1]) #for scenario=='civil_comments', some items need to be downweighted, for other scenarios not
        if 'civil_comments' in chosen_scenarios:
            balance_weights[scenarios_position['civil_comments']] = scores[:,scenarios_position['civil_comments']].max(axis=0)
            #(balance_weights==0).sum(axis=0) verifying that no item had weight 0 (the output should be zero)
            scores[:,scenarios_position['civil_comments']] = (scores[:,scenarios_position['civil_comments']]>0).astype(float)
        if 'mmlu' in chosen_scenarios:
            N = len(scenarios_position['mmlu'])
            n_sub = len(scenarios['mmlu'])
            for sub in scenarios['mmlu']:
                n_i = len(subscenarios_position['mmlu'][sub])
                balance_weights[subscenarios_position['mmlu'][sub]] = N/(n_sub*n_i)             
        
        # Create training and test sets by hiding specific rows
        scores_train = scores[[i for i in range(scores.shape[0]) if i not in rows_to_hide]]
        scores_test = scores[[i for i in range(scores.shape[0]) if i in rows_to_hide]]
        responses_train = np.zeros(scores_train.shape)
        responses_test = np.zeros(scores_test.shape)

        # Threshold responses 
        cs = np.linspace(0.01,.99,1000)  # Threshold values to consider
        for scenario in chosen_scenarios:
            ind = scenarios_position[scenario]
            # Find the best threshold value that minimizes the difference between mean responses and mean scores
            c = cs[np.argmin([np.mean((np.abs((scores_train[:,ind]>c).mean(axis=1)-scores_train[:,ind].mean(axis=1)))) for c in cs])]
            # Apply the threshold to train and test responses
            responses_train[:,ind] = (scores_train[:,ind]>c).astype(int)
            responses_test[:,ind] = (scores_test[:,ind]>c).astype(int)

        # Transforming scores back
        scores = balance_weights*scores
        scores_train = scores[[i for i in range(scores.shape[0]) if i not in rows_to_hide]]
        scores_test = scores[[i for i in range(scores.shape[0]) if i in rows_to_hide]]
        #print("shape of scores=",scores.shape, "shape of scores train=",scores_train.shape,"shape of scores test=",scores_test.shape)
        
        # Storing true accs to use later
        for j in range(len(rows_to_hide)):
            accs_true[rows_to_hide[j]] = {}
            for scenario in chosen_scenarios:
                accs_true[rows_to_hide[j]][scenario] = scores_test[j, scenarios_position[scenario]].mean()
        
        # Choosing D through validation
        val_ind = list(range(0,responses_train.shape[0],5)) #list(range(int(responses_train.shape[0]/3)))
        train_ind = [i for i in range(responses_train.shape[0]) if i not in val_ind]
        
        # Create IRT dataset for validation and train IRT models
        dataset_name = f'data/{bench}/rows-{rows_to_hide_str}_scenario-{scenario_name}_val.jsonlines'
        create_irt_dataset(responses_train[train_ind], dataset_name)

        errors = []  # Initialize a list to hold validation errors
        errors2 = []
        print("\ni) choosing optimal D")
        for D in tqdm(Ds):
            # Train IRT model for the current dimension (D)
            model_name = f'models/{bench}/rows-{rows_to_hide_str}_D-{D}_scenario-{scenario_name}_val/'
            train_irt_model(dataset_name, model_name, D, lr, epochs, device)
            # Load trained IRT model parameters
            A, B, Theta = load_irt_parameters(model_name)
            # Determine seen and unseen items for validation
            seen_items = list(range(0, responses_train.shape[1], 2))
            unseen_items = list(range(1, responses_train.shape[1], 2))
            # Estimate ability parameters for the validation set
            #thetas = [estimate_ability_parameters(responses_train[val_ind][j], seen_items, A, B) for j in tqdm(range(len(val_ind)))]
            print(" - fit. theta in the val set")
            pool = mp.Pool(cpu)
            thetas = pool.starmap(estimate_ability_parameters, [(responses_train[val_ind][j], seen_items, A, B) for j in range(len(val_ind))])
            pool.close()
            pool.join()

            # Compute validation errors for each scenario and update the errors list (in the end, we give the same weight for all scenarios)
            errors2.append([])
            for scenario in chosen_scenarios:
                ind = [u for u in unseen_items if u in scenarios_position[scenario]]
                errors2[-1].append(np.mean([abs((balance_weights*item_curve(thetas[j], A, B))[0,ind].mean()-scores_train[val_ind][j,ind].mean())for j in range(len(val_ind))]))
            errors.append(np.mean(errors2[-1]))
            print(errors[-1])

        # Choose the simplest model (D) that is not far from the best model based on validation errors
        ind_D = np.argmax(np.array(errors)-np.min(errors)<.0025)
        D = Ds[ind_D] 
        print("- opt D=", D, "errors=", errors, "\n")

        # Choosing lambdas (For random G-PIRT)
        print("\nii) choosing optimal lambdas")
        
        opt_lambds = {'random_gpirt': {}, 'anchor_gpirt': {}, 'anchor-irt_gpirt': {}}  # Initialize a dictionary to hold optimal lambda values
      
        vs = {}
        bs = {}
        for i,scenario in enumerate(chosen_scenarios):
            vs[scenario] = np.var(scores_train[:,scenarios_position[scenario]], axis=1).mean()
            bs[scenario] = np.mean(errors2[ind_D][i]) 

        for scenario in tqdm(chosen_scenarios):
            for key in opt_lambds.keys():
                opt_lambds[key][scenario] = {}
                for number_item in number_items: 
                    if key == 'random_gpirt':
                        opt_lambds[key][scenario][number_item] = get_lambda(bs[scenario], vs[scenario]/number_item)
                    else:
                        opt_lambds[key][scenario][number_item] = get_lambda(bs[scenario], vs[scenario]/(4*number_item))

        # Save the final dataset and train the final IRT model
        print("\niii) fitting final IRT model")
        dataset_name = f'data/{bench}/row-{rows_to_hide_str}_scenario-{scenario_name}.jsonlines'

        create_irt_dataset(responses_train, dataset_name)
        model_name = f'models/{bench}/row-{rows_to_hide_str}_D-validate_scenario-{scenario_name}'
        train_irt_model(dataset_name, model_name, D, lr, epochs, device)

        # Load the final IRT model
        A, B, Theta = load_irt_parameters(model_name)

        print("\niv) sampling")
        item_weights_dic, seen_items_dic, unseen_items_dic = {}, {}, {}
        for sampling_name in tqdm(sampling_names):
            item_weights_dic[sampling_name], seen_items_dic[sampling_name], unseen_items_dic[sampling_name] = {}, {}, {}
            pool = mp.Pool(cpu)
            samples = pool.starmap(sample_items, [(number_item, iterations, sampling_name, chosen_scenarios, scenarios, subscenarios_position, responses_test, scores_train, scenarios_position, A, B) for number_item in number_items])
            pool.close()
            pool.join()

            for i,number_item in enumerate(number_items):
                item_weights_dic[sampling_name][number_item], seen_items_dic[sampling_name][number_item], unseen_items_dic[sampling_name][number_item] = samples[i]
                
            
        print("\nv) computing accuracies")
        start_time = time.time()
        pool = mp.Pool(cpu)
        out += pool.starmap(calculate_accuracies, [(j, sampling_names, item_weights_dic, seen_items_dic, unseen_items_dic, A, B, scores_test, responses_test, scenarios_position, chosen_scenarios, balance_weights, opt_lambds, rows_to_hide) for j in range(len(rows_to_hide))])
        pool.close()
        pool.join()
        elapsed_time = np.round(time.time()-start_time)
        print(f" - finished in {elapsed_time} seconds")
        
    ### Final results
    
    accs_hat = {}
    results = {}
    for item in out:
        key = list(item.keys())[0]
        accs_hat[key] = item[key]

    # Update results with the mean absolute difference for each approach
    for rows_to_hide in set_of_rows:
        for j in range(len(rows_to_hide)):
            results[rows_to_hide[j]] = {}
            for number_item in number_items:
                results[rows_to_hide[j]][number_item] = {}
                for sampling_name in sampling_names:
                    for estimators in ['naive', 'pirt', 'cirt', 'gpirt']:
                        results[rows_to_hide[j]][number_item][sampling_name+'_'+estimators] = {}
                        for scenario in chosen_scenarios:
                            results[rows_to_hide[j]][number_item][sampling_name+'_'+estimators][scenario] = np.abs(np.array(accs_hat[rows_to_hide[j]][number_item][sampling_name+'_'+estimators][scenario]) - accs_true[rows_to_hide[j]][scenario])

    return results, accs_hat # Return the updated results dictionary
