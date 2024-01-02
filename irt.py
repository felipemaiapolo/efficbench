import numpy as np
from scipy.optimize import minimize
import jsonlines
import os
import json
from utils import *
     
def create_irt_dataset(responses, dataset_name): #row_to_hide, scenario_name):
    dataset = []
    for i in range(responses.shape[0]):
        aux = {}
        aux_q = {}
        for j in range(responses.shape[1]):
            aux_q['q' + str(j)] = int(responses[i, j])
        aux['subject_id'] = str(i)
        aux['responses'] = aux_q
        dataset.append(aux)
    
    ### save datasets
    with jsonlines.open(dataset_name, mode='w') as writer:
        writer.write_all([dataset[i] for i in range(len(dataset))])

def train_irt_model(dataset_name, model_name, D, hidden, dropout, lr, epochs, device):
    #command=f"py-irt train 'multidim_2pl' {dataset_name} {model_name} --dims {D} --device {device}"
    command=f"py-irt train 'multidim_2pl' {dataset_name} {model_name} --dims {D} --hidden {hidden} --dropout {dropout} --lr {lr} --epochs {epochs} --device {device}"
    with SuppressPrints():
        os.system(command)
        
def load_irt_parameters(model_name):
    with open(model_name+'best_parameters.json') as f:
        params = json.load(f)
    A = np.array(params['disc']).T[None, :, :]
    B = np.array(params['diff']).T[None, :, :]
    Theta = np.array(params['ability'])[:,:,None]
    return A, B, Theta

def estimate_ability_parameters(responses_test, seen_items, A, B, eps=1e-10):

    D = A.shape[1]
    
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]).squeeze()
        log_likelihood = np.sum(responses_test[seen_items] * np.log(P + eps) + (1 - responses_test[seen_items]) * np.log(1 - P + eps))
        return -log_likelihood

    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, np.zeros(D)).x[None,:,None]
    return optimal_theta