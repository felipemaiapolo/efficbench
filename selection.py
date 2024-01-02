import numpy as np
import random
from numba import jit
from sklearn.linear_model import LinearRegression
from sklearn_extra.cluster import KMedoids
from irt import *
from utils import *

def get_seen_unseen_items(scenarios_choosen, scenarios, number_item, subscenarios_position, responses_test):
    
    def shuffle_list(lista):
        return random.sample(lista, len(lista))

    seen_items = []
    for scenario in scenarios_choosen:
        number_items_sub = np.zeros(len(scenarios[scenario])).astype(int)
        number_items_sub += number_item // len(scenarios[scenario])
        number_items_sub[:(number_item - number_items_sub.sum())] += 1
        i = 0
        for sub in shuffle_list(scenarios[scenario]):
            seen_items += random.sample(subscenarios_position[scenario][sub], k=number_items_sub[i])
            i += 1

    # unseen items
    unseen_items = [i for i in range(responses_test.shape[1]) if i not in seen_items]

    return seen_items, unseen_items

def select_initial_adaptive_items(A, B, Theta, number_item, try_size=2000, seed=42):
    random.seed(seed)
    mats = np.stack([np.outer(A[0, :, i], A[0, :, i]) for i in range(A.shape[2])])
    samples = [random.sample(range(A.shape[-1]), number_item) for _ in range(try_size)]
    samples_infos = np.stack([np.linalg.det(np.array([(p * (1 - p))[:, None, None] * mats[s] for p in item_curve(Theta, A[:, :, s], B[:, :, s])]).sum(axis=1)).sum() for s in samples])
    seen_items = samples[np.argmax(samples_infos)]
    unseen_items = [i for i in range(A.shape[-1]) if i not in seen_items]
    return seen_items, unseen_items, mats


def run_adaptive_selection(responses_test, seen_items, unseen_items, scenarios_choosen, scenarios_position, A, B, mats, target_count, balance=False):
    
    assert len(seen_items) <= target_count
    count = len(seen_items)

    while True:
        for scenario in scenarios_choosen:
            if count >= target_count:
                return seen_items, unseen_items 
            
            seen_items, unseen_items = select_next_adaptive_item(responses_test, seen_items, unseen_items, scenario, scenarios_position, A, B, mats, balance)
            count += 1

def select_next_adaptive_item(responses_test, seen_items, unseen_items, scenario, scenarios_position, A, B, mats, balance):
    
    D = A.shape[1]

    if balance:
        unseen_items_scenario = [u for u in unseen_items if u in scenarios_position[scenario]]
    else:
        unseen_items_scenario = unseen_items

    optimal_theta = estimate_ability_parameters(responses_test, seen_items, A, B)
    P = item_curve(optimal_theta, A, B).squeeze()

    # Compute information matrices for seen and unseen items
    I_seen = ((P * (1 - P))[:, None, None] * mats)[seen_items].sum(axis=0)
    I_unseen = ((P * (1 - P))[:, None, None] * mats)[unseen_items_scenario]

    # Select the next item based on the maximum determinant of information
    next_item = unseen_items_scenario[np.argmax(np.linalg.det(I_seen[None, :, :] + I_unseen))]                    
    seen_items.append(next_item)
    unseen_items.remove(next_item)

    return seen_items, unseen_items

             
def anchor(A, B, Theta, number_item):
    P = item_curve(Theta, A, B)
    X = np.log(P/(1-P))

    ###
    kmedoids = KMedoids(n_clusters=number_item, metric='correlation', random_state=0).fit(X.T)
    anchor_points = np.argmax(np.corrcoef(np.hstack((kmedoids.cluster_centers_.T, X)).T)[:number_item,number_item:], axis=1)

    ###
    regs = []
    for j in range(A.shape[-1]):
        regs.append(LinearRegression().fit(X[:, anchor_points[kmedoids.labels_[j]]].reshape(-1,1), X[:, j]))
    
    ###
    anchor_weights = np.array([np.mean(kmedoids.labels_==anchor) for anchor in range(number_item)])

    return anchor_points, kmedoids, anchor_weights, regs

 