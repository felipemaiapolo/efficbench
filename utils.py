#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import autograd.numpy as np
from autograd import grad
from tqdm import tqdm


def get_lambda(b, v):
    return (b**2)/(v+(b**2))

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


def item_response_function(xj, theta, a, b):
    """
    Compute the pdf for the Bernoulli distribution of an item response.
    
    Parameters:
    - xj: The response of the subject (0 or 1).
    - theta: The ability parameter of the subject.
    - a: The discrimination parameter of the item.
    - b: The difficulty parameter of the item.
    
    Returns:
    - The pdf value for the given response.
    """
    a = np.array([[[a]]]) if type(a) == np.float64 else a
    b = np.array([[[b]]]) if type(b) == np.float64 else b
    p_correct = item_curve(theta, a, b)
    return np.power(p_correct, xj) * np.power(1 - p_correct, 1 - xj)

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

helm_scenarios = {'cnn':['summarization_cnndm:temperature=0.3,device=cuda,'], 
             'xsum':['summarization_xsum:temperature=0.3,device=cuda,'], 
             'boolq:':['boolq:'],
             'civil_comments':['civil_comments:demographic=LGBTQ,',
                               'civil_comments:demographic=all,',
                               'civil_comments:demographic=black,',
                               'civil_comments:demographic=christian,',
                               'civil_comments:demographic=female,',
                               'civil_comments:demographic=male,',
                               'civil_comments:demographic=muslim,',
                               'civil_comments:demographic=other_religions,',
                               'civil_comments:demographic=white,'],
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

lb_scenarios = {'harness_truthfulqa_mc_0':['harness_truthfulqa_mc_0'],
                 'gsm8k':['harness_gsm8k_5'], 
                 'winogrande':['harness_winogrande_5'], 
                 'arc':['harness_arc_challenge_25'], 
                 'hellaswag':['harness_hellaswag_10'],
                 'mmlu':['harness_hendrycksTest_abstract_algebra_5', 
                         'harness_hendrycksTest_anatomy_5', 
                         'harness_hendrycksTest_astronomy_5', 
                         'harness_hendrycksTest_business_ethics_5', 
                         'harness_hendrycksTest_clinical_knowledge_5', 
                         'harness_hendrycksTest_college_biology_5', 
                         'harness_hendrycksTest_college_chemistry_5', 
                         'harness_hendrycksTest_college_computer_science_5', 
                         'harness_hendrycksTest_college_mathematics_5', 
                         'harness_hendrycksTest_college_medicine_5', 
                         'harness_hendrycksTest_college_physics_5', 
                         'harness_hendrycksTest_computer_security_5', 
                         'harness_hendrycksTest_conceptual_physics_5', 
                         'harness_hendrycksTest_econometrics_5', 
                         'harness_hendrycksTest_electrical_engineering_5', 
                         'harness_hendrycksTest_elementary_mathematics_5', 
                         'harness_hendrycksTest_formal_logic_5', 
                         'harness_hendrycksTest_global_facts_5', 
                         'harness_hendrycksTest_high_school_biology_5', 
                         'harness_hendrycksTest_high_school_chemistry_5', 
                         'harness_hendrycksTest_high_school_computer_science_5', 
                         'harness_hendrycksTest_high_school_european_history_5', 
                         'harness_hendrycksTest_high_school_geography_5', 
                         'harness_hendrycksTest_high_school_government_and_politics_5', 
                         'harness_hendrycksTest_high_school_macroeconomics_5', 
                         'harness_hendrycksTest_high_school_mathematics_5', 
                         'harness_hendrycksTest_high_school_microeconomics_5', 
                         'harness_hendrycksTest_high_school_physics_5', 
                         'harness_hendrycksTest_high_school_psychology_5', 
                         'harness_hendrycksTest_high_school_statistics_5', 
                         'harness_hendrycksTest_high_school_us_history_5', 
                         'harness_hendrycksTest_high_school_world_history_5', 
                         'harness_hendrycksTest_human_aging_5', 
                         'harness_hendrycksTest_human_sexuality_5', 
                         'harness_hendrycksTest_international_law_5', 
                         'harness_hendrycksTest_jurisprudence_5', 
                         'harness_hendrycksTest_logical_fallacies_5', 
                         'harness_hendrycksTest_machine_learning_5', 
                         'harness_hendrycksTest_management_5', 
                         'harness_hendrycksTest_marketing_5', 
                         'harness_hendrycksTest_medical_genetics_5', 
                         'harness_hendrycksTest_miscellaneous_5', 
                         'harness_hendrycksTest_moral_disputes_5', 
                         'harness_hendrycksTest_moral_scenarios_5', 
                         'harness_hendrycksTest_nutrition_5', 
                         'harness_hendrycksTest_philosophy_5', 
                         'harness_hendrycksTest_prehistory_5', 
                         'harness_hendrycksTest_professional_accounting_5', 
                         'harness_hendrycksTest_professional_law_5', 
                         'harness_hendrycksTest_professional_medicine_5', 
                         'harness_hendrycksTest_professional_psychology_5',
                         'harness_hendrycksTest_public_relations_5', 
                         'harness_hendrycksTest_security_studies_5', 
                         'harness_hendrycksTest_sociology_5', 
                         'harness_hendrycksTest_us_foreign_policy_5', 
                         'harness_hendrycksTest_virology_5', 
                         'harness_hendrycksTest_world_religions_5']}

alpaca_scenarios = {#'alpaca_v1':['alpaca_v1'],
                    'alpaca_v2':['alpaca_v2'],
}