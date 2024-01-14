from tqdm import tqdm
import pickle
from irt import *
from selection import *
from utils import *

random_state = 42



scenarios = {'boolq:':['boolq:'],
             #'civil_comments':['civil_comments:demographic=LGBTQ,',
             #                  'civil_comments:demographic=all,',
             #                  'civil_comments:demographic=black,',
             #                  'civil_comments:demographic=christian,',
             #                  'civil_comments:demographic=female,',
             #                  'civil_comments:demographic=male,',
             #                  'civil_comments:demographic=muslim,',
             #                  'civil_comments:demographic=other_religions,',
             #                  'civil_comments:demographic=white,'],
             'commonsense:dataset=hellaswag,method=multiple_choice_separate_original,':['commonsense:dataset=hellaswag,method=multiple_choice_separate_original,'],
             'commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,':['commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,'],
             'imdb:':['imdb:'],
             'mmlu':['mmlu:subject=abstract_algebra,method=multiple_choice_joint,',
                     'mmlu:subject=college_chemistry,method=multiple_choice_joint,',
                     'mmlu:subject=computer_security,method=multiple_choice_joint,',
                     'mmlu:subject=econometrics,method=multiple_choice_joint,',
                     'mmlu:subject=us_foreign_policy,method=multiple_choice_joint,'],
             'msmarco:track=regular,valid_topk=30,':['msmarco:track=regular,valid_topk=30,'],
             #'msmarco:track=trec,valid_topk=30,':['msmarco:track=trec,valid_topk=30,'],
             'narrative_qa:':['narrative_qa:'],
             'natural_qa:mode=closedbook,':['natural_qa:mode=closedbook,'],
             'natural_qa:mode=openbook_longans,':['natural_qa:mode=openbook_longans,'],
             'quac:':['quac:'],
             'raft':['raft:subset=ade_corpus_v2,',
                     'raft:subset=banking_77,',
                     'raft:subset=neurips_impact_statement_risks,',
                     'raft:subset=one_stop_english,',
                     'raft:subset=overruling,',
                     'raft:subset=semiconductor_org_types,',
                     'raft:subset=systematic_review_inclusion,',
                     'raft:subset=tai_safety_research,',
                     'raft:subset=terms_of_service,',
                     'raft:subset=tweet_eval_hate,',
                     'raft:subset=twitter_complaints,'],
             'truthful_qa:task=mc_single,method=multiple_choice_joint,':['truthful_qa:task=mc_single,method=multiple_choice_joint,']}

scenarios_metrics = {'boolq:':'em',
                     'commonsense:dataset=hellaswag,method=multiple_choice_separate_original,':'em',
                     'commonsense:dataset=openbookqa,method=multiple_choice_separate_calibrated,':'em',
                     'imdb:':'em',
                     'mmlu':'em',
                     'msmarco:track=regular,valid_topk=30,':'RR@10',
                     'msmarco:track=trec,valid_topk=30,':'NDCG@10',
                     'narrative_qa:':'f1',
                     'natural_qa:mode=closedbook,':'f1',
                     'natural_qa:mode=openbook_longans,':'f1',
                     'quac:':'f1',
                     'raft':'em',
                     'truthful_qa:task=mc_single,method=multiple_choice_joint,':'em'}



with open('data/helm.pickle', 'rb') as handle:
    data = pickle.load(handle)

PATH_TO_RESULTS = './results/MCAT'

dropout = .5 #default .5
hidden = 100 #default 100
epochs = 1000 #default 2000
lr = .1 #default .1
device = 'cuda'
balance = False

num_elements = 4
iterations = 30
number_items = [10, 25, 50, 100, 150]
Ds = [3] #[1, 3, 5]



accs_final = []
results_final = []

for scenario in list(scenarios.keys()):

    scenario_name = scenario
    scenarios_choosen = [scenario]
    set_of_rows = create_sublists_corrected(list(range(len(data['models']))), num_elements)

    accs = {}
    results = {}

    for rows_to_hide in tqdm(set_of_rows):
        rows_to_hide_str = ':'.join([str(r) for r in rows_to_hide])

        ### Prep data
        scenarios_position, subscenarios_position = prepare_data(scenarios_choosen, scenarios, data)
        scores = create_responses(scenarios_choosen, scenarios, data)
        scores_train = scores[[i for i in range(scores.shape[0]) if i not in rows_to_hide]]
        scores_test = scores[[i for i in range(scores.shape[0]) if i in rows_to_hide]]

        cs = np.linspace(0,1,1000)
        c = cs[np.argmin([np.mean((np.abs((scores_train>c).mean(axis=1)-scores_train.mean(axis=1)))) for c in cs])]
        responses_train = (scores_train>c).astype(int)
        responses_test = (scores_test>c).astype(int)

        ### Choosing D
        train_ind = list(range(0,responses_train.shape[0],2))
        val_ind = [i for i in range(responses_train.shape[0]) if i not in train_ind]
        responses_train[train_ind].shape

        dataset_name = f'data/irt_helm/rows-{rows_to_hide_str}_scenario-{scenario_name}_val_all_models.jsonlines'
        create_irt_dataset(responses_train[train_ind], dataset_name)

        errors = []
        for D in Ds:
            model_name = f'models/irt_helm/rows-{rows_to_hide_str}_D-{D}_scenario-{scenario_name}_val_all_models/'
            train_irt_model(dataset_name, model_name, D, hidden, dropout, lr, epochs, device)
            A, B, Theta = load_irt_parameters(model_name)
            seen_items, unseen_items, _ = select_initial_adaptive_items(A, B, Theta, 2*D)
            errors.append(np.median(np.abs(responses_train[val_ind][:,unseen_items].mean(axis=1)-np.array([item_curve(estimate_ability_parameters(r, seen_items, A, B), A, B)[:,unseen_items].mean() for r in responses_train[val_ind]]))))
        D = Ds[np.argmin(errors)]
        print(D,errors)

        ### Saving dataset
        dataset_name = f'data/irt_helm/row-{rows_to_hide_str}_scenario-{scenario_name}_all_models.jsonlines'
        create_irt_dataset(responses_train, dataset_name)

        ### Train final IRT model
        model_name = f'models/irt_helm/row-{rows_to_hide_str}_D-validate_scenario-{scenario_name}_all_models/'
        train_irt_model(dataset_name, model_name, D, hidden, dropout, lr, epochs, device)

        ### Load IRT model
        A, B, Theta = load_irt_parameters(model_name)

        ### Creating storage space in acc and results to store new results
        [create_space_accs_results(accs, results, r, number_items, scenarios_choosen) for r in rows_to_hide]

        ### Running adaptive evaluation
        for j in range(len(rows_to_hide)):

            seen_items, unseen_items, mats = select_initial_adaptive_items(A, B, Theta, 2*D) #number_items[0]

            for number_item in number_items:

                # Number of samples
                target_count = len(scenarios_choosen)*number_item

                # Sampling new items
                seen_items, unseen_items = run_adaptive_selection(responses_test[j], seen_items, unseen_items, scenarios_choosen, scenarios_position, A, B, mats, target_count, balance=balance)

                # Running IRT in the remaining sample
                new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                # Updating 'accs' and 'results'
                update_accs_irt('adaptive_irt', scores_test[j], responses_test[j], rows_to_hide[j], scenarios_choosen, scenarios_position, seen_items, unseen_items, A, B, new_theta, accs, number_item)
                update_results('adaptive_irt', scores_test[j], rows_to_hide[j], scenarios_choosen, scenarios_position, accs, results, number_item)

        ### Running random evaluation
        for j in range(len(rows_to_hide)):

            for number_item in number_items:

                ### Running with different seeds (ie, different seen_items)
                for it in range(iterations):
                    random.seed(random_state*it)
                    seen_items, unseen_items = get_seen_unseen_items(scenarios_choosen, scenarios, number_item, subscenarios_position, responses_test)

                    ### naive
                    # Updating 'accs'
                    update_accs_naive('random_naive', scores_test[j], rows_to_hide[j], scenarios_choosen, scenarios_position, seen_items, accs, number_item)

                    ### IRT
                    new_theta = estimate_ability_parameters(responses_test[j], seen_items, A, B)

                    # Updating 'accs'
                    update_accs_irt('random_irt', scores_test[j], responses_test[j], rows_to_hide[j], scenarios_choosen, scenarios_position, seen_items, unseen_items, A, B, new_theta, accs, number_item)

                ### Updating 'results'
                update_results('random_naive', scores_test[j], rows_to_hide[j], scenarios_choosen, scenarios_position, accs, results, number_item)
                update_results('random_irt', scores_test[j], rows_to_hide[j], scenarios_choosen, scenarios_position, accs, results, number_item)

    ### plots
    #plot_results(results, scenarios_choosen, number_items, scenarios_metrics[scenario], scenario_name, 'partial')

    accs_final.append(accs)
    results_final.append(results)


with open(os.path.join(PATH_TO_RESULTS, 'r1_less_models.pickle', 'wb')) as handle:
    pickle.dump({'accs':accs_final, 'res':results_final}, handle, protocol=pickle.HIGHEST_PROTOCOL)



