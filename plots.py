import pickle
import copy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#from experiments import *
from utils import *

# Define the new renamings according to the latest instructions
rename_mappings = {
    'random_naive': 'random',
    'anchor_naive': 'correct.',
    'anchor-irt_naive': 'IRT',
    'random_pirt': 'random+',
    'anchor_pirt': 'correct.+',
    'anchor-irt_pirt': 'IRT+',
    'random_gpirt': 'random++',
    'anchor_gpirt': 'correct.++',
    'anchor-irt_gpirt': 'IRT++',
    'anchor': 'correct.',
    'anchor-irt':'IRT'
}

color_mappings = {
    'random_naive': '#8c564b',
    'anchor_naive': '#1f77b4',
    'anchor-irt_naive': '#2ca02c',
    'random_gpirt': '#9467bd',
    'anchor_gpirt': '#d62728',
    'anchor-irt_gpirt': '#ff7f0e',
    'anchor': '#1f77b4',
    'anchor-irt': '#2ca02c'
}

benchs = ['lb', 'mmlu', 'helm_lite', 'alpaca','mmlu_fields', 'icl_templates']
titles = {'lb':'Open LLM Leaderboard','mmlu':'MMLU','helm_lite':'HELM','alpaca':'AlpacaEval', 'icl_templates':'ICL consistency'}
splits = {'lb':['iid','noniid'],'mmlu':['iid','noniid'],
          'helm_lite':['iid','noniid'],'alpaca':['iid','noniid'],
          'mmlu_fields':['iid','noniid'],'icl_templates':['iid','noniid','noniid2','noniid3']}

agg_metric = 'avg' #'std' (std=variation across seeds)
methods = ['random_naive', 'anchor_naive', 'anchor-irt_naive',
           #'random_pirt', 'anchor_pirt', 'anchor-irt_pirt']#,
           #'random_cirt','anchor_cirt', 'anchor-irt_cirt']#,
           'random_gpirt', 'anchor_gpirt', 'anchor-irt_gpirt']


style = {"alpha":1, "markersize":3, "markeredgewidth":1, "elinewidth":1, "capsize":3, "linestyle":''}

def plot_perf_lines(table_avg, table_std, title, xlabel, ylabel, ylim,
                    legend=False, error_bar=False, show_title=True, show_xlabel=True, show_ylabel=True, ncols=6, posic=(-1.5, -.35)):
    
    markers = ['o', 'v', '*', 'x', 's','p']
    jitters = [-6.3,-3.7,-1.3,1.3,3.7,6.3]
    colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][:9]
    j=0
    for method, values in table_avg.items():
        x = np.array(list(values.keys()))
        y = np.array(list(values.values()))
        s = np.array(list(table_std[method].values()))
        
        if error_bar: 
            plt.errorbar(x+jitters[j], y, color =color_mappings[method], yerr=s, label=rename_mappings[method], marker=markers[j], **style)
        else: 
            plt.plot(x, y, label=method)

        j+=1
    if show_xlabel: plt.xlabel(xlabel, size=11)
    if show_ylabel: plt.ylabel(ylabel, size=11)
    plt.ylim(ylim[0], ylim[1])
    if show_title:
        plt.title(title)
    else:
        pass
    
    tick_label_size = 10  # Example size, adjust as needed
    plt.tick_params(axis='x', labelsize=tick_label_size)
    plt.tick_params(axis='y', labelsize=tick_label_size)
    
    if legend: plt.legend(loc='upper center', ncols=ncols, bbox_to_anchor=posic)
    plt.grid(alpha=.2)
    #plt.grid(which='major', color='black', linestyle='-')
    #plt.grid(which='minor', color='gray', linestyle=':')
    #plt.show()
   
def winrate(x,axis):
    n = x.shape[axis]
    return(np.argsort(np.argsort(x, axis=axis), axis=axis)/n)

def load_scores(bench, split):
    with open(f'results/accs_{bench}_split-{split}_iterations-5.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    if bench=='mmlu':
        with open(f'data/lb.pickle', 'rb') as handle:
            data2 = pickle.load(handle)
    elif bench=='alpaca':
        with open(f'data/alpaca_v2.pickle', 'rb') as handle:
            data2 = pickle.load(handle)
    else:
        with open(f'data/{bench}.pickle', 'rb') as handle:
            data2 = pickle.load(handle)
    if bench=='lb':scenarios = lb_scenarios
    elif bench=='mmlu':scenarios = {'mmlu':lb_scenarios['mmlu']}
    elif bench=='helm_lite':scenarios = helm_lite_scenarios
    elif bench=='alpaca':scenarios = alpaca_scenarios
    elif bench=='mmlu_fields':scenarios = {'mmlu':lb_scenarios['mmlu']}
    elif bench=='icl_templates':scenarios = icl_templates_scenarios
    else: raise NotImplementedError
    
    scenarios_position, subscenarios_position = prepare_data(scenarios, scenarios, data2)
    scores = create_responses(scenarios, scenarios, data2)
        
    # Balance weights
    balance_weights = np.ones(scores.shape[1]) 
    for scenario in scenarios:
        N = len(scenarios_position[scenario])
        n_sub = len(scenarios[scenario])
        for sub in scenarios[scenario]:
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N/(n_sub*n_i)
            
    scores = balance_weights*scores
    
    scores = np.vstack([scores[:,scenarios_position[scenario]].mean(axis=1) for scenario in scenarios])
    
    return scores[:,list(data.keys())]
