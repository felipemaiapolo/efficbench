from tqdm import tqdm
import multiprocessing as mp
from irt import *
from selection import *
from utils import *


def validate_lambda_random(it, scenario, number_item, chosen_scenarios, scenarios, subscenarios_position, responses_test, responses_train, scores_train, val_ind, scenarios_position, A, B, lambds):
    
    """
    Validates lambda (a weighting parameter) for the random IRT model by computing the mean absolute difference between computed accuracies and actual scores.
    
    Parameters:
    - it: Iteration number or seed for random number generation.
    - scenario: The scenario being considered.
    - number_item: The number of items to consider.
    - chosen_scenarios: A list of scenarios to be considered.
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
    seen_items, unseen_items = get_seen_unseen_items(chosen_scenarios, scenarios, number_item, subscenarios_position, responses_test)
    
    # Estimate ability parameters for the validation set
    thetas = [estimate_ability_parameters(responses_train[val_ind][j], seen_items, A, B) for j in range(len(val_ind))]
    
    # Compute and return the mean absolute differences for each lambda value
    return np.array([[abs(scores_train[val_ind][j][scenarios_position[scenario]].mean()-compute_acc_irt(scenario, scores_train[val_ind][j], scenarios_position, seen_items, unseen_items, A, B, thetas[j], lambd=lambd)) for lambd in lambds] for j in range(len(val_ind))]).mean(axis=0)
    
def evaluate_scenarios(data, scenario_name, chosen_scenarios, scenarios, set_of_rows, Ds, iterations, device):

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
    
    lambds = [None] + np.linspace(0,1,10).tolist()  # Lambda values to consider
    number_items = [10, 25, 50, 75, 100]  # Number of items to consider in evaluations

    cpu = mp.cpu_count()  # Number of available CPU cores
    epochs = 2000  # Number of epochs for IRT model training (package default is 2000)
    lr = .1  # Learning rate for IRT model training (package default is .1)

    accs = {}  # Initialize a dictionary to hold accuracies
    results = {}  # Initialize a dictionary to hold results

    # Iterate through each set of rows to hide
    for rows_to_hide in tqdm(set_of_rows):
        rows_to_hide_str = ':'.join([str(r) for r in rows_to_hide])

        
        # Prepare data and scenarios
        scenarios_position, subscenarios_position = prepare_data(chosen_scenarios, scenarios, data)
        scores = create_responses(chosen_scenarios, scenarios, data)
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

        # Choosing D through validation
        val_ind = list(range(0,responses_train.shape[0],3))
        train_ind = [i for i in range(responses_train.shape[0]) if i not in val_ind]
        
        # Create IRT dataset for validation and train IRT models
        dataset_name = f'data/irt_helm/rows-{rows_to_hide_str}_scenario-{scenario_name}_val.jsonlines'
        create_irt_dataset(responses_train[train_ind], dataset_name)

        errors = []  # Initialize a list to hold validation errors
        errors2 = []
        for D in Ds:
            # Train IRT model for the current dimension (D)
            model_name = f'models/irt_helm/rows-{rows_to_hide_str}_D-{D}_scenario-{scenario_name}_val/'
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
                errors2[-1].append(np.mean([abs(item_curve(thetas[j], A, B)[0,ind].mean()-scores_train[val_ind][j,ind].mean())for j in range(len(val_ind))]))
            errors.append(np.mean(errors2[-1]))

        # Choose the simplest model (D) that is not far from the best model based on validation errors
        D = Ds[np.argmax(np.array(errors)-np.min(errors)<.005)] 
        print("\nD", D, errors, "\n")

        # Choosing lambdas (For random G-PIRT)
        opt_lambds = {'random_gpirt': {}}  # Initialize a dictionary to hold optimal lambda values
        # Reload trained IRT model parameters for the chosen dimension (D)
        model_name = f'models/irt_helm/rows-{rows_to_hide_str}_D-{D}_scenario-{scenario_name}_val/'
        A, B, Theta = load_irt_parameters(model_name)
        for scenario in chosen_scenarios:
            opt_lambds['random_gpirt'][scenario] = {}
            for number_item in number_items:

                # Evaluate lambda values for the random IRT model using multiprocessing
                pool = mp.Pool(cpu)
                errors = pool.starmap(validate_lambda_random, [(it, scenario, number_item, chosen_scenarios, scenarios, subscenarios_position, responses_test, responses_train, scores_train, val_ind, scenarios_position, A, B, lambds) for it in range(iterations)])
                pool.close()
                pool.join()
                # Choose the lambda value that minimizes the mean error
                opt_lambds['random_gpirt'][scenario][number_item] = lambds[np.argmin(np.array(errors).mean(axis=0))]

        # Save the final dataset and train the final IRT model
        dataset_name = f'data/irt_helm/row-{rows_to_hide_str}_scenario-{scenario_name}.jsonlines'
        create_irt_dataset(responses_train, dataset_name)
        model_name = f'models/irt_helm/row-{rows_to_hide_str}_D-validate_scenario-{scenario_name}/'
        train_irt_model(dataset_name, model_name, D, lr, epochs, device)

        # Load the final IRT model
        A, B, Theta = load_irt_parameters(model_name)

        # Initialize storage space in accuracies and results dictionaries
        [create_space_accs_results(accs, results, r, number_items, chosen_scenarios) for r in rows_to_hide]

        # Running random evaluation for each hidden row
        for j in range(len(rows_to_hide)):
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
                        accs[rows_to_hide[j]][number_item]['random_pirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, lambd=None))
                        accs[rows_to_hide[j]][number_item]['random_gpirt'][scenario].append(compute_acc_irt(scenario, scores_test[j], scenarios_position, seen_items, unseen_items, A, B, new_theta, lambd=opt_lambds['random_gpirt'][scenario][number_item]))

                ### Updating results
                # Update results with the mean absolute difference for each approach
                update_results('random_naive', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                update_results('random_pirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                update_results('random_gpirt', scores_test[j], rows_to_hide[j], chosen_scenarios, scenarios_position, accs, results, number_item)
                
    return results # Return the updated results dictionary