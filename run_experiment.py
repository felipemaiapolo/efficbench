import pickle
import copy
import pandas as pd
import argparse
from scipy import stats
from experiments import *
from utils import *

#python run_experiment.py --bench 'mmlu' --split 'iid' --iterations 5 --device 'cuda'
#python run_experiment.py --bench 'helm_lite' --split 'noniid' --iterations 5 --device 'cuda'

def get_data(bench, split):
    # Loading data
    if bench in ['lb','mmlu']:
        #data
        with open('data/lb.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = {'mmlu': lb_scenarios['mmlu']} if bench == 'mmlu' else lb_scenarios

        #split
        if split == 'iid':
            set_of_rows = [list(range(0,len(data['models']),4))]
        elif split == 'noniid':
            set_of_rows = [list(range(int(len(data['models'])/4))),]
        elif split == 'noniid2':
            set_of_rows = [list(range(200))]
        elif split == 'noniid3':
            set_of_rows = [list(range(300))] 

        print(len(set_of_rows[0]), len(data['models']))

    elif bench == 'helm_lite':
        #data
        with open('data/helm_lite.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = helm_lite_scenarios

        #split
        if split == 'iid':
            set_of_rows = [[0,11,22],
                           [1,12,23], 
                           [2,13,24], 
                           [3,14,25],
                           [4,15,26],
                           [5,16,27],
                           [6,17,28],
                           [7,18,29],
                           [8,19],
                           [9,20],
                           [10,21]] 
        else:
            set_of_rows = [[0,1], #AI: Yi
                           [2,3,4], #AlephAlpha_luminous
                           [5,6], #ai21_j2
                           [7,8,9,10], #anthropic_claude
                           [11,12],#cohere
                           [13,14], #google
                           [15,16,17,18], #llama
                           [19,20], #mistral ai
                           [21,22,23,24,25], #openai
                           [26,27], #TII/UAE
                           [28,29]] #writer

        print(len(set_of_rows[0]), len(data['models']))

    elif bench == 'alpaca':
        #data
        with open('data/alpaca_v2.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = alpaca_scenarios

        #split
        if split == 'iid':
            set_of_rows = [list(range(0,len(data['models']),4)),
                           list(range(1,len(data['models'])+1,4)),
                           list(range(2,len(data['models'])+2,4)),
                           list(range(3,len(data['models'])+3,4))]
        elif split == 'noniid':
            set_of_rows = [list(range(int(len(data['models'])/4))),]
        elif split == 'noniid2':
            set_of_rows = [list(range(50))]
        elif split == 'noniid3':
            set_of_rows = [list(range(75))]
 
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

    elif bench == 'icl_templates':
        #data
        with open('data/icl_templates.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = icl_templates_scenarios

        #split
        if split == 'iid':
            import random
            random.seed(42) #0
            list1 = random.sample(range(len(data['models'])), int(len(data['models'])/2))
            list2 = [i for i in range(len(data['models'])) if i not in list1]
            set_of_rows = [list1, list2]

        elif split == 'noniid': #instruction
            templates = [['GPT_3_style','MNLI_crowdsource','always_sometimes_never',
                          'based_on_the_previous_passage','can_we_infer','claim_true_false_inconclusive',
                          'consider_always_sometimes_never','does_it_follow_that'],['does_this_imply',
                          'guaranteed_possible_impossible', 'guaranteed_true', 'justified_in_saying',
                          'must_be_true','should_assume','take_the_following_as_truth']]
            set_of_rows = [[i for i,m in enumerate(data['models']) if np.sum([t in m for t in temp])>0] for temp in templates]

        elif split == 'noniid2': #size
            sizes = [['65b']] 
            set_of_rows = [[i for i,m in enumerate(data['models']) if np.sum([t in m for t in size])>0] for size in sizes]

        elif split == 'noniid3': #same vs cross instr
            cross = [['same_instr'],['cross_instr']]
            set_of_rows = [[i for i,m in enumerate(data['models']) if np.sum([t in m for t in cr])>0] for cr in cross]

        else:
            raise NotImplementedError

        print(len(set_of_rows[0]), len(data['models'])) 

    else:
        raise NotImplementedError
    
    return data, scenarios, set_of_rows


if __name__ == "__main__":
    
    # User input
    parser = argparse.ArgumentParser(description='Example script with named arguments.')

    parser.add_argument('--bench', type=str, help='Benchmark (helm_lite, lb, mmlu, alpaca, icl_templates)', default = 'lb')
    parser.add_argument('--split', type=str, help='iid/noniid/noniid2/noniid3', default = 'iid')
    parser.add_argument('--iterations', type=int, help='iterations', default = 3)
    parser.add_argument('--device', type=str, help='cpu/cuda', default = 'cpu')

    args = parser.parse_args()
    bench = args.bench
    split = args.split
    iterations = args.iterations
    device = args.device

    assert bench in ['helm_lite','lb','mmlu','alpaca','mmlu_fields','icl_templates']
    assert split in ['iid','noniid','noniid2','noniid3']
    assert iterations>0

    # Defining other parameters
    Ds = [2, 5, 10, 15]
    sampling_names = ['random', 'anchor', 'anchor-irt']#, 'adaptive'] 

    scenario_name = 'full' #we are evaluating all scenarios at once (this is just a nomination)

    data, scenarios, set_of_rows = get_data(bench, split)
    
    chosen_scenarios = list(scenarios.keys())


    # ## Results
    results_full, accs_full, sampling_time_dic = evaluate_scenarios(data, scenario_name, chosen_scenarios, scenarios, set_of_rows, Ds, iterations, device, bench='irt_'+bench, split=split, sampling_names = sampling_names)

    with open(f'results/results_{bench}_split-{split}_iterations-{iterations}.pickle', 'wb') as handle:
        pickle.dump(results_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'results/accs_{bench}_split-{split}_iterations-{iterations}.pickle', 'wb') as handle:
        pickle.dump(accs_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'results/samplingtime_{bench}_split-{split}_iterations-{iterations}.pickle', 'wb') as handle:
        pickle.dump(sampling_time_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
