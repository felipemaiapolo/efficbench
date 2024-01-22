from tqdm import tqdm
import pickle
from irt import *
from selection import *
from utils import *
from experiments import evaluate_scenarios
import os

JOB_ID = os.getenv('SLURM_JOB_ID')
JOB_ID = JOB_ID if JOB_ID is not None else 'local'

IGNORE_CONTINUOUS = True
continuous_scenarios = ['msmarco:track=regular,valid_topk=30,', 
                        'narrative_qa:', 'natural_qa:mode=closedbook,', 
                        'natural_qa:mode=openbook_longans,', 'quac:', ]


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

#scenarios = {'boolq:':['boolq:']}

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


if IGNORE_CONTINUOUS:
    keep = set(scenarios.keys()).difference(continuous_scenarios)
    scenarios = {k: scenarios[k] for k in keep}

with open('data/helm.pickle', 'rb') as handle:
    data = pickle.load(handle)

PATH_TO_RESULTS = './results/MCAT'

device = 'cuda'
DEV = True

num_elements = 4
iterations = 8

epochs_grid = [10] if DEV else [2000] #[100, 1000, 2000, 4000, 5000]
Ds = [1] if DEV else [1] #, 3, 5, 10, 15] #[1]  #[1, 3, 5]

chosen_scenarios = list(scenarios.keys())
scenario_name = 'full'
set_of_rows = create_sublists_corrected(list(range(len(data['models']))), num_elements)
sampling = {'random_sampling':False if DEV else True,
            'anchor_sampling':False,
            'anchor-irt_sampling':False,
            'disc_sampling':False, 
            'adaptive_sampling': True,
            'adaptive-ki_sampling': False}

for epochs in epochs_grid:
    name_run = f'adaptive_rerun_D1_reweighing_all_samplings' #'baseline_default_pipeline' # adaptive_rerun_D1_reweighing_all_scenarios
    SAVE_NAME = f'{name_run}_{JOB_ID}.pkl'

    accs, results = evaluate_scenarios(data, scenario_name, 
                                                 chosen_scenarios, scenarios, 
                                                 set_of_rows, Ds, iterations, 
                                                 device, bench='irt_helm',
                                                 sampling=sampling,
                                                 epochs=epochs)

    with open(os.path.join(PATH_TO_RESULTS, f'results_{SAVE_NAME}'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(PATH_TO_RESULTS, f'accs_{SAVE_NAME}'), 'wb') as handle:
        pickle.dump(accs, handle, protocol=pickle.HIGHEST_PROTOCOL)
