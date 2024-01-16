from tqdm import tqdm
import multiprocessing as mp
from irt import *
from selection import *
from utils import *
import os

JOB_ID = os.getenv('SLURM_JOB_ID')
JOB_ID = JOB_ID if JOB_ID is not None else 'local'

def validate_lambda_random(it, scenario, number_item, scenarios, subscenarios_position, responses_test, responses_train, scores_train, val_ind, scenarios_position, A, B, balance_weights, lambds):
    
    """
    Validates lambda (a weighting parameter) for the random IRT model by computing the mean absolute difference between computed accuracies and actual scores.
    
    Parameters:
    - it: Iteration number or seed for random number generation.
    - scenario: The scenario being considered.
    - number_item: The number of items to consider.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - subscenarios_position: A nested dictionary mapping each scenario and subscenario to the positions of its items.
    - responses_test: A numpy array of the test subject's responses to all items.
    - responses_train: A numpy array of the training subject's responses to all items.
    - scores_train: Actual scores for training data.
    - val_ind: Indices of the validation set.
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - A: The discrimination parameter of the item.
    - B: The difficulty parameter of the item.
    - lambds: A list of lambda values to validate.
    
    Returns:
    - A numpy array of mean absolute differences for each lambda value.
    """
    
    random.seed(it)  # Set the random seed
    
    # Determine seen and unseen items based on the given parameters
    seen_items, unseen_items = get_seen_unseen_items([scenario], scenarios, number_item, subscenarios_position, responses_test)
    
    # Estimate ability parameters for the validation set
    thetas = [estimate_ability_parameters(responses_train[val_ind][j], seen_items, A, B) for j in range(len(val_ind))]
    
    # Compute and return the mean absolute differences for each lambda value
    return np.array([[abs(scores_train[val_ind][j][scenarios_position[scenario]].mean()-compute_acc_irt(scenario, scores_train[val_ind][j], scenarios_position, seen_items, unseen_items, A, B, thetas[j], balance_weights=balance_weights, lambd=lambd)) for lambd in lambds] for j in range(len(val_ind))]).mean(axis=0)

def validate_lambda_anchor(it, scenario, number_item, responses_train, scores_train, val_ind, scenarios_position, A, B, balance_weights, lambds, anchor_irt = False):
    
    """
    Validates lambda (a weighting parameter) for the random IRT model by computing the mean absolute difference between computed accuracies and actual scores.
    
    Parameters:
    - scenario: The scenario being considered.
    - number_item: The number of items to consider.
    - responses_train: A numpy array of the training subject's responses to all items.
    - scores_train: Actual scores for training data.
    - val_ind: Indices of the validation set.
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - A: The discrimination parameter of the item.
    - B: The difficulty parameter of the item.
    - lambds: A list of lambda values to validate.
    
    Returns:
    - A numpy array of mean absolute differences for each lambda value.
    """
    
    train_ind = [i for i in range(responses_train.shape[0]) if i not in val_ind]
    
    if anchor_irt == True:
        E = np.vstack((A.squeeze(), B.reshape((1,-1)))) #embeddings
        _, _, seen_items, unseen_items = get_anchor(E, [scenario], scenarios_position, number_item, it)
        item_weights = None
    else:
        _, anchor_weights, seen_items, unseen_items = get_anchor(scores_train[train_ind], [scenario], scenarios_position, number_item, it)
        item_weights = anchor_weights[scenario]
    
    # Estimate ability parameters for the validation set
    thetas = [estimate_ability_parameters(responses_train[val_ind][j], seen_items, A, B) for j in range(len(val_ind))]
    
    # Compute and return the mean absolute differences for each lambda value
    return np.array([[abs(scores_train[val_ind][j][scenarios_position[scenario]].mean()-compute_acc_irt(scenario, scores_train[val_ind][j], scenarios_position, seen_items, unseen_items, A, B, thetas[j], balance_weights=balance_weights, lambd=lambd, item_weights=item_weights)) for lambd in lambds] for j in range(len(val_ind))]).mean(axis=0)

def validate_lambda_disc(seen_items, scenario, responses_train, scores_train, val_ind, scenarios_position, balance_weights, A, B, lambds):
    
    """
    Validates lambda (a weighting parameter) for the random IRT model by computing the mean absolute difference between computed accuracies and actual scores.
    
    Parameters:
    - scenario: The scenario being considered.
    - number_item: The number of items to consider.
    - responses_train: A numpy array of the training subject's responses to all items.
    - scores_train: Actual scores for training data.
    - val_ind: Indices of the validation set.
    - scenarios_position: A dictionary mapping each scenario to the positions of its items.
    - A: The discrimination parameter of the item.
    - B: The difficulty parameter of the item.
    - lambds: A list of lambda values to validate.
    
    Returns:
    - A numpy array of mean absolute differences for each lambda value.
    """
    
    train_ind = [i for i in range(responses_train.shape[0]) if i not in val_ind]
    unseen_items = [i for i in range(responses_train.shape[1]) if i not in seen_items]
    item_weights = None
    
    # Estimate ability parameters for the validation set
    thetas = [estimate_ability_parameters(responses_train[val_ind][j], seen_items, A, B) for j in range(len(val_ind))]
    
    # Compute and return the mean absolute differences for each lambda value
    return np.array([[abs(scores_train[val_ind][j][scenarios_position[scenario]].mean()-compute_acc_irt(scenario, scores_train[val_ind][j], scenarios_position, seen_items, unseen_items, A, B, thetas[j], balance_weights=balance_weights, lambd=lambd, item_weights=item_weights)) for lambd in lambds] for j in range(len(val_ind))]).mean(axis=0)

def evaluate_scenarios_adaptive(data, scenario_name, chosen_scenarios, 
                       scenarios, set_of_rows, Ds, iterations, device, bench='irt_helm', epochs=2000,
                       #sampling = {'random_sampling':True,'anchor_sampling':False,
                       #            'anchor-irt_sampling':False,'disc_sampling':False}
                                   ):
    """
    """

    #epochs = 5000 #5 #1000 #default 
    lr = .1 #default .1
    number_items = [10, 25, 50, 75, 100] 
    balance = True
    random_state = 42

    accs_final = []
    results_final = []

    for scenario in list(scenarios.keys()):

        #scenario_name = scenario
        #chosen_scenarios = [scenario]
        #set_of_rows = create_sublists_corrected(list(range(len(data['models']))), num_elements)
        
        accs = {}
        results = {}

        for rows_to_hide in tqdm(set_of_rows):
            rows_to_hide_str = ':'.join([str(r) for r in rows_to_hide])

            ### Prep data
            scenarios_position, subscenarios_position = prepare_data(chosen_scenarios, scenarios, data)
            scores = create_responses(chosen_scenarios, scenarios, data)
            scores_train = scores[[i for i in range(scores.shape[0]) if i not in rows_to_hide]]
            scores_test = scores[[i for i in range(scores.shape[0]) if i in rows_to_hide]]

            responses_train = np.zeros(scores_train.shape)
            responses_test = np.zeros(scores_test.shape)

            '''
            cs = np.linspace(0.01,.99,1000) #np.linspace(0,1,1000)
            c = cs[np.argmin([np.mean((np.abs((scores_train>c).mean(axis=1)-scores_train.mean(axis=1)))) for c in cs])]
            responses_train = (scores_train>c).astype(int)
            responses_test = (scores_test>c).astype(int)
            '''

            # Threshold responses 
            cs = np.linspace(0.01,.99,1000)  # Threshold values to consider
            for scenario in chosen_scenarios:
                ind = scenarios_position[scenario]
                # Find the best threshold value that minimizes the difference between mean responses and mean scores
                c = cs[np.argmin([np.mean((np.abs((scores_train[:,ind]>c).mean(axis=1)-scores_train[:,ind].mean(axis=1)))) for c in cs])]
                # Apply the threshold to train and test responses
                responses_train[:,ind] = (scores_train[:,ind]>c).astype(int)
                responses_test[:,ind] = (scores_test[:,ind]>c).astype(int)
            
            ### Choosing D
            train_ind = list(range(0,responses_train.shape[0],2))
            val_ind = [i for i in range(responses_train.shape[0]) if i not in train_ind]
            #responses_train[train_ind].shape

            dataset_name = f'data/irt_helm/rows-{rows_to_hide_str}_scenario-{scenario_name}_val_all_models_{JOB_ID}.jsonlines'
            create_irt_dataset(responses_train[train_ind], dataset_name)

            if len(Ds) > 1:
                errors = []
                for D in Ds:
                    model_name = f'models/irt_helm/rows-{rows_to_hide_str}_D-{D}_scenario-{scenario_name}_val_all_models_{JOB_ID}/'
                    train_irt_model(dataset_name, model_name, D, lr, epochs, device) ## (dataset_name, model_name, D, hidden, dropout, lr, epochs, device
                    A, B, Theta = load_irt_parameters(model_name)
                    seen_items, unseen_items, _ = select_initial_adaptive_items(A, B, Theta, 2*D)
                    errors.append(np.median(np.abs(
                        responses_train[val_ind][:,unseen_items].mean(axis=1)-
                        np.array([item_curve(estimate_ability_parameters(subject, seen_items, A, B), A, B)[:,unseen_items].mean() 
                                for subject in responses_train[val_ind]]))))
                D = Ds[np.argmin(errors)]
                print(D,errors)
            else:
                D = Ds[0]   

            ### Saving dataset
            dataset_name = f'data/irt_helm/row-{rows_to_hide_str}_scenario-{scenario_name}_all_models_{JOB_ID}.jsonlines'
            create_irt_dataset(responses_train, dataset_name)

            ### Train final IRT model
            model_name = f'models/irt_helm/row-{rows_to_hide_str}_D-validate_scenario-{scenario_name}_all_models_{JOB_ID}/'
            train_irt_model(dataset_name, model_name, D, lr, epochs, device) ## (dataset_name, model_name, D, hidden, dropout, lr, epochs, device

            ### Load IRT model
            A, B, Theta = load_irt_parameters(model_name)

            ### Creating storage space in acc and results to store new results
            [create_space_accs_results(accs, results, subject, number_items, chosen_scenarios) for subject in rows_to_hide]

            ### Running adaptive evaluation
            for j in range(len(rows_to_hide)):

                seen_items, unseen_items, mats = select_initial_adaptive_items(A, B, Theta, 2*D) #number_items[0]

                for number_item in number_items:

                    # Number of samples
                    target_count = len(chosen_scenarios)*number_item

                    # Sampling new items
                    seen_items, unseen_items = run_adaptive_selection(responses_test[j], seen_items, unseen_items, chosen_scenarios, scenarios_position, A, B, mats, target_count, balance=balance)

                    # Running IRT in the remaining sample
                    new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                    # Updating 'accs' and 'results'
                    update_accs_irt('adaptive_irt', scores_test[j], responses_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, seen_items, unseen_items, A, B, new_theta, accs, number_item)
                    update_results('adaptive_irt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)

            ### Running random evaluation
            for j in range(len(rows_to_hide)):

                for number_item in number_items:

                    ### Running with different seeds (ie, different seen_items)
                    for it in range(iterations):
                        random.seed(random_state*it)
                        seen_items, unseen_items = get_seen_unseen_items(chosen_scenarios, scenarios, number_item, subscenarios_position, responses_test)

                        ### naive
                        # Updating 'accs'
                        update_accs_naive('random_naive', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, seen_items, accs, number_item)

                        ### IRT
                        new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                        # Updating 'accs'
                        update_accs_irt('random_irt', scores_test[j], responses_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, seen_items, unseen_items, A, B, new_theta, accs, number_item)

                    ### Updating 'results'
                    update_results('random_naive', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('random_irt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)

        ### plots
        #plot_results(results, chosen_scenarios, number_items, scenarios_metrics[scenario], scenario_name, 'partial')

        accs_final.append(accs)
        results_final.append(results)

        return accs_final, results_final



def evaluate_scenarios(data, scenario_name, chosen_scenarios, 
                       scenarios, set_of_rows, Ds, iterations, device, bench, 
                       sampling = {'random_sampling':True,'anchor_sampling':False,
                                   'anchor-irt_sampling':False,'disc_sampling':False}):

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
    
    assert bench in ['irt_helm', 'irt_lb']
    
    lambds = [None] + np.round(np.linspace(0,1,10),2).tolist()  # Lambda values to consider
    number_items = [10, 25, 50, 75, 100]  # Number of items to consider in evaluations

    cpu = mp.cpu_count()  # Number of available CPU cores
    epochs = 2000  # Number of epochs for IRT model training (package default is 2000)
    lr = .1  # Learning rate for IRT model training (package default is .1)

    accs = {}  # Initialize a dictionary to hold accuracies
    results = {}  # Initialize a dictionary to hold results

    # Iterate through each set of rows to hide
    for rows_to_hide in set_of_rows:
        rows_to_hide_str = ':'.join([str(r) for r in rows_to_hide])

        print(f"\nEvaluating models {rows_to_hide}")
        
        # Prepare data and scenarios
        scenarios_position, subscenarios_position = prepare_data(chosen_scenarios, scenarios, data)
        scores = create_responses(chosen_scenarios, scenarios, data)
        
        balance_weights = np.ones(scores.shape[1]) #for scenario=='civil_comments', some items need to be downweighted, for other scenarios not
        if 'civil_comments' in chosen_scenarios:
            balance_weights[scenarios_position['civil_comments']] = scores[:,scenarios_position['civil_comments']].max(axis=0)
            #(balance_weights==0).sum(axis=0) verifying that no item had weight 0 (the output should be zero)
            scores[:,scenarios_position['civil_comments']] = (scores[:,scenarios_position['civil_comments']]>0).astype(float)

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
        
        # Choosing D through validation
        val_ind = list(range(int(responses_train.shape[0]/3)))
        train_ind = [i for i in range(responses_train.shape[0]) if i not in val_ind]
        
        # Create IRT dataset for validation and train IRT models
        dataset_name = f'data/{bench}/rows-{rows_to_hide_str}_scenario-{scenario_name}_val_{JOB_ID}.jsonlines'
        create_irt_dataset(responses_train[train_ind], dataset_name)

        errors = []  # Initialize a list to hold validation errors
        errors2 = []
        print("\ni) choosing optimal D")
        for D in tqdm(Ds):
            # Train IRT model for the current dimension (D)
            model_name = f'models/{bench}/rows-{rows_to_hide_str}_D-{D}_scenario-{scenario_name}_val_{JOB_ID}/'
            train_irt_model(dataset_name, model_name, D, lr, epochs, device)
            # Load trained IRT model parameters
            A, B, Theta = load_irt_parameters(model_name)
            # Determine seen and unseen items for validation
            seen_items = list(range(0, responses_train.shape[1], 2))
            unseen_items = list(range(1, responses_train.shape[1], 2))
            # Estimate ability parameters for the validation set
            thetas = [estimate_ability_parameters(responses_train[val_ind][j], seen_items, A, B) for j in range(len(val_ind))]

            # Compute validation errors for each scenario and update the errors list (in the end, we give the same weight for all scenarios)
            errors2.append([])
            for scenario in chosen_scenarios:
                ind = [u for u in unseen_items if u in scenarios_position[scenario]]
                errors2[-1].append(np.mean([abs((balance_weights*item_curve(thetas[j], A, B))[0,ind].mean()-scores_train[val_ind][j,ind].mean())for j in range(len(val_ind))]))
            errors.append(np.mean(errors2[-1]))

        # Choose the simplest model (D) that is not far from the best model based on validation errors
        ind_D = np.argmax(np.array(errors)-np.min(errors)<.0025)
        D = Ds[ind_D] 
        #D = Ds[np.argmin(errors)] 
        print("- opt D=", D, "errors=", errors, "\n")

        # Choosing lambdas (For random G-PIRT)
        print("\nii) choosing optimal lambdas")
        
        opt_lambds = {'random_gpirt': {}, 'anchor_gpirt': {}, 'anchor-irt_gpirt': {}, 'disc_gpirt': {}}  # Initialize a dictionary to hold optimal lambda values
        
        vs = {}
        bs = {}
        for i,scenario in enumerate(chosen_scenarios):
            vs[scenario] = np.var(scores_train[:,scenarios_position[scenario]])
            bs[scenario] = np.mean(errors2[ind_D][i]) 

        for scenario in tqdm(chosen_scenarios):
            for key in opt_lambds.keys():
                opt_lambds[key][scenario] = {}
                for number_item in number_items: 
                    if key == 'random_gpirt':
                        opt_lambds[key][scenario][number_item] = get_lambda(number_item, bs[scenario], vs[scenario])
                    else:
                        opt_lambds[key][scenario][number_item] = get_lambda(number_item, bs[scenario], vs[scenario]/2)

        if False:
            iterations_val = 2*iterations                            
            if sampling['disc_sampling']==True:
                seen_items_dic = get_disc_items(responses_train[train_ind], number_items, chosen_scenarios, rows_to_hide_str, scenarios_position, device, bench)
            # Reload trained IRT model parameters for the chosen dimension (D)
            model_name = f'models/{bench}/rows-{rows_to_hide_str}_D-{D}_scenario-{scenario_name}_val/'
            A, B, Theta = load_irt_parameters(model_name)
            #print(" - debiasing IRT")
            #A, B = debias_irt(A, B, Theta, responses_train[train_ind])

            for scenario in tqdm(chosen_scenarios):
                for key in opt_lambds.keys():
                    opt_lambds[key][scenario] = {}
                for number_item in number_items: 
                    if sampling['anchor_sampling']==True: 
                        # Evaluate lambda values for the anchor 
                        anchor_irt = False
                        pool = mp.Pool(cpu)
                        errors = pool.starmap(validate_lambda_anchor, [(it+1000, scenario, number_item, responses_train, scores_train, val_ind, scenarios_position, A, B, balance_weights, lambds, anchor_irt) for it in range(iterations_val)])
                        pool.close()
                        pool.join()
                        opt_lambds['anchor_gpirt'][scenario][number_item] = lambds[np.argmin(np.array(errors).mean(axis=0))] # Choose the lambda value that minimizes the mean error

                    if sampling['random_sampling']==True:
                        # Evaluate lambda values for the random IRT model using multiprocessing
                        pool = mp.Pool(cpu)
                        errors = pool.starmap(validate_lambda_random, [(it+1000, scenario, number_item, scenarios, subscenarios_position, responses_test, responses_train, scores_train, val_ind, scenarios_position, A, B, balance_weights, lambds) for it in range(iterations_val)])
                        pool.close()
                        pool.join()
                        opt_lambds['random_gpirt'][scenario][number_item] = lambds[np.argmin(np.array(errors).mean(axis=0))]

                    if sampling['disc_sampling']==True:
                        # Evaluate lambda values for the disc-irt method
                        errors = validate_lambda_disc(seen_items_dic[number_item], scenario, responses_train, scores_train, val_ind, scenarios_position, balance_weights, A, B, lambds)
                        opt_lambds['disc_gpirt'][scenario][number_item] = lambds[np.argmin(errors)]

                    if sampling['anchor-irt_sampling']==True:
                        # Evaluate lambda values for the anchor-irt method
                        anchor_irt = True
                        pool = mp.Pool(cpu)
                        errors = pool.starmap(validate_lambda_anchor, [(it+1000, scenario, number_item, responses_train, scores_train, val_ind, scenarios_position, A, B, balance_weights, lambds, anchor_irt) for it in range(iterations_val)])
                        pool.close()
                        pool.join()
                        opt_lambds['anchor-irt_gpirt'][scenario][number_item] = lambds[np.argmin(np.array(errors).mean(axis=0))] # Choose the lambda value that minimizes the mean error

        print(opt_lambds)
        
        # Save the final dataset and train the final IRT model
        dataset_name = f'data/{bench}/row-{rows_to_hide_str}_scenario-{scenario_name}_{JOB_ID}.jsonlines'
        create_irt_dataset(responses_train, dataset_name)
        model_name = f'models/{bench}/row-{rows_to_hide_str}_D-validate_scenario-{scenario_name}_{JOB_ID}/'
        train_irt_model(dataset_name, model_name, D, lr, epochs, device)

        # Load the final IRT model
        A, B, Theta = load_irt_parameters(model_name)
        #print(" - debiasing IRT")
        #A, B = debias_irt(A, B, Theta, responses_train)
        
        # Initialize storage space in accuracies and results dictionaries
        [create_space_accs_results(accs, results, r, number_items, chosen_scenarios) for r in rows_to_hide]

        ### Running anchor evaluation for each hidden row ###
        if sampling['anchor_sampling']==True:
            print("\niii) running anchor points")
            for number_item in tqdm(number_items):
                for j in range(len(rows_to_hide)):
                    for it in range(iterations):

                        _, anchor_weights, seen_items, unseen_items = get_anchor(scores_train, chosen_scenarios, scenarios_position, number_item, random_seed = it) #this part os fhte code is ineficient cause we dont need to run this line for every j again (however, we would have to change the code that updates the results)

                        ### Naive approach
                        # Update accuracies for the naive approach
                        for scenario in chosen_scenarios:
                            accs[rows_to_hide[j]][number_item]['anchor_naive'][scenario].append((anchor_weights[scenario]*scores_test[j][[s for s in seen_items if s in scenarios_position[scenario]]]).sum())

                        ### IRT approach
                        # Estimate ability parameters for the test set
                        new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                        # Update accuracies for the IRT and G-PIRT approaches
                        for scenario in chosen_scenarios:
                            accs[rows_to_hide[j]][number_item]['anchor_cirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None, item_weights=anchor_weights[scenario], thresh=.5))
                            accs[rows_to_hide[j]][number_item]['anchor_pirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None, item_weights=anchor_weights[scenario]))
                            accs[rows_to_hide[j]][number_item]['anchor_gpirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=opt_lambds['anchor_gpirt'][scenario][number_item], item_weights=anchor_weights[scenario]))

                    ### Updating results
                    # Update results with the mean absolute difference for each approach
                    update_results('anchor_naive', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('anchor_cirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('anchor_pirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('anchor_gpirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)


        if sampling['random_sampling']==True:
            ### Running random evaluation for each hidden row ###
            print("\niv) running random eval")
            for j in tqdm(range(len(rows_to_hide))):
                for number_item in number_items:
                    # Running evaluations with different seeds (i.e., different seen_items)
                    for it in range(iterations):
                        random.seed(it)
                        # Determine seen and unseen items for the current evaluation
                        seen_items, unseen_items = get_seen_unseen_items(chosen_scenarios, scenarios, number_item, subscenarios_position, responses_test)

                        ### Naive approach
                        # Update accuracies for the naive approach
                        for scenario in chosen_scenarios:
                            accs[rows_to_hide[j]][number_item]['random_naive'][scenario].append(scores_test[j][[s for s in seen_items if s in scenarios_position[scenario]]].mean())

                        ### IRT approach
                        # Estimate ability parameters for the test set
                        new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                        # Update accuracies for the IRT and G-PIRT approaches
                        for scenario in chosen_scenarios:
                            accs[rows_to_hide[j]][number_item]['random_cirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None, thresh=.5))
                            accs[rows_to_hide[j]][number_item]['random_pirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None))
                            accs[rows_to_hide[j]][number_item]['random_gpirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=opt_lambds['random_gpirt'][scenario][number_item]))

                    ### Updating results
                    # Update results with the mean absolute difference for each approach
                    update_results('random_naive', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('random_cirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('random_pirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('random_gpirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)

        ### Running disc evaluation for each hidden row ###
        if sampling['disc_sampling']==True:
            print("\nvi) running disc IRT method")

            seen_items_dic = get_disc_items(responses_train[train_ind], number_items, chosen_scenarios, rows_to_hide_str, scenarios_position, device, bench)

            for number_item in tqdm(number_items):
                for j in range(len(rows_to_hide)):

                    seen_items = seen_items_dic[number_item]
                    unseen_items = [i for i in range(responses_train.shape[1]) if i not in seen_items]

                    ### Naive approach
                    # Update accuracies for the naive approach
                    for scenario in chosen_scenarios:
                        accs[rows_to_hide[j]][number_item]['disc_naive'][scenario].append((scores_test[j][[s for s in seen_items if s in scenarios_position[scenario]]]).mean())

                    ### IRT approach
                    # Estimate ability parameters for the test set
                    new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                    # Update accuracies for the IRT and G-PIRT approaches
                    for scenario in chosen_scenarios:
                        accs[rows_to_hide[j]][number_item]['disc_cirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None, item_weights=None, thresh=.5))
                        accs[rows_to_hide[j]][number_item]['disc_pirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None, item_weights=None))
                        accs[rows_to_hide[j]][number_item]['disc_gpirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=opt_lambds['disc_gpirt'][scenario][number_item], item_weights=None))

                    ### Updating results
                    # Update results with the mean absolute difference for each approach
                    update_results('disc_naive', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('disc_cirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('disc_pirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('disc_gpirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
        
        ### Running anchor evaluation for each hidden row ###
        if sampling['anchor-irt_sampling']==True:
            print("\nv) running anchor points with IRT embeddings")

            E = np.vstack((A.squeeze(), B.reshape((1,-1)))) #embeddings

            for number_item in tqdm(number_items):
                for j in range(len(rows_to_hide)):
                    for it in range(iterations):

                        _, anchor_weights, seen_items, unseen_items = get_anchor(E, chosen_scenarios, scenarios_position, number_item, random_seed = it) #this part os fhte code is ineficient cause we dont need to run this line for every j again (however, we would have to change the code that updates the results)

                        ### Naive approach
                        # Update accuracies for the naive approach
                        for scenario in chosen_scenarios:
                            accs[rows_to_hide[j]][number_item]['anchor-irt_naive'][scenario].append((anchor_weights[scenario]*scores_test[j][[s for s in seen_items if s in scenarios_position[scenario]]]).sum())

                        ### IRT approach
                        # Estimate ability parameters for the test set
                        new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                        # Update accuracies for the IRT and G-PIRT approaches
                        for scenario in chosen_scenarios:
                            accs[rows_to_hide[j]][number_item]['anchor-irt_cirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None, item_weights=None, thresh=.5))
                            accs[rows_to_hide[j]][number_item]['anchor-irt_pirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=None, item_weights=None))
                            accs[rows_to_hide[j]][number_item]['anchor-irt_gpirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, balance_weights, lambd=opt_lambds['anchor-irt_gpirt'][scenario][number_item], item_weights=None))

                    ### Updating results
                    # Update results with the mean absolute difference for each approach
                    update_results('anchor-irt_naive', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('anchor-irt_cirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('anchor-irt_pirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                    update_results('anchor-irt_gpirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                
    return results, accs # Return the updated results dictionary
