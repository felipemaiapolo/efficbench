import pickle
import copy
import pandas as pd
import argparse
from scipy import stats
from experiments import *
from utils import *

#python run_experiment.py --bench 'lb' --split 'iid' --iterations 10 --device 'cuda'

# ## Definitions

# User input
parser = argparse.ArgumentParser(description='Example script with named arguments.')

parser.add_argument('--bench', type=str, help='Benchmark (helm, lb, mmlu, alpaca)', default = 'lb')
parser.add_argument('--split', type=str, help='iid/noniid', default = 'iid')
parser.add_argument('--iterations', type=int, help='iterations', default = 3)
parser.add_argument('--device', type=str, help='cpu/cuda', default = 'cpu')

args = parser.parse_args()
bench = args.bench
split = args.split
iterations = args.iterations
device = args.device

assert bench in ['helm','lb','mmlu','alpaca','mmlu_fields']
assert split in ['iid','noniid']
assert iterations>0

# Defining other parameters
Ds = [2, 5, 10, 15]
sampling_names = ['random', 'anchor', 'anchor-irt'] #, 'adaptive', 
scenario_name = 'full' #we are evaluating all scenarios at once (this is just a nomination)

# ## Data

# Loading data
if bench in ['lb','mmlu']:
    #data
    with open('data/lb.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    #scenarios
    scenarios = lb_scenarios
    if bench == 'mmlu':
        scenarios = {'mmlu':scenarios['mmlu']}
    
    #split
    if split == 'iid':
        set_of_rows = [list(range(0,len(data['models']),4))]
    else:
        set_of_rows = [list(range(int(len(data['models'])/4)))]
    print(len(set_of_rows[0]), len(data['models']))

elif bench == 'helm':
    #data
    with open('data/helm.pickle', 'rb') as handle:
        data = pickle.load(handle)
        
    #scenarios
    scenarios = helm_scenarios
    
    #split
    if split == 'iid':
        set_of_rows = [[0,5,10,15,20,25], 
                       [1,6,11,16,21,26], 
                       [2,7,12,17,22,27], 
                       [3,8,13,18,23],
                       [4,9,14,19,24]] 
    else:
        set_of_rows = [[0,1,2,3], #ai21
                       [5,6,7,8,9,10,11], #cohere
                       [4,12,13], #anthropic+microsoft
                       [14,15,16,17,18,19,20,21,22], #openai
                       [23,24,25,26,27]] #together
    print(len(set_of_rows[0]), len(data['models']))
    
elif bench == 'alpaca':
    #data
    with open('data/alpaca_v2.pickle', 'rb') as handle:
        data = pickle.load(handle)
 
    #scenarios
    scenarios = alpaca_scenarios
    
    #split
    if split == 'iid':
        set_of_rows = [list(range(0,len(data['models']),4))]
    else:
        set_of_rows = [list(range(int(len(data['models'])/4)))]
    print(len(set_of_rows[0]), len(data['models']))
          
# Loading data
elif bench == 'mmlu_fields':
    
    #data
    with open('data/mmlu_fields.pickle', 'rb') as handle:
        data = pickle.load(handle)
    
    #scenarios
    scenarios = lb_scenarios
    scenarios = {'mmlu':scenarios['mmlu']}
    
    #split
    if split == 'iid':
        k = int(len(data['models'])/40)
        set_of_rows = [list(range(0,len(data['models']),k))]
    else:
        set_of_rows = [list(range(40))]
    print(len(set_of_rows[0]), len(data['models']))
    
else:
    raise NotImplementedError
chosen_scenarios = list(scenarios.keys())


# ## Results
results_full, accs_full, sampling_time_dic = evaluate_scenarios(data, scenario_name, chosen_scenarios, scenarios, set_of_rows, Ds, iterations, device, bench='irt_'+bench, sampling_names = sampling_names)

with open(f'results/results_{bench}_split-{split}_iterations-{iterations}.pickle', 'wb') as handle:
    pickle.dump(results_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'results/accs_{bench}_split-{split}_iterations-{iterations}.pickle', 'wb') as handle:
    pickle.dump(accs_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'results/samplingtime_{bench}_split-{split}_iterations-{iterations}.pickle', 'wb') as handle:
    pickle.dump(sampling_time_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
