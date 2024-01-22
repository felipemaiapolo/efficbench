import json
import os
from glob import glob 
import pickle as pkl
import numpy as np

RAW_DATA_PATH = "./data/alpaca_results_raw"
ALPACA_EVAL_VERSION = "2.0"
RESULTS_FOLDERS = {"1.0": "alpaca_eval_gpt4",
                   "2.0": "weighted_alpaca_eval_gpt4_turbo",
                   }
SHIFT_VALUE = 1 # 0 for no shifting, 1 for shifting into [0,1]

version = 'alpaca_v2' if ALPACA_EVAL_VERSION == '2.0' else 'alpaca_v1'
directories = glob(os.path.join(RAW_DATA_PATH,  "*"))
models = [path.split(os.sep)[-1] for path in directories]

all_data = []
exceptions = {}

for model in models:
    try:
        results_file = os.path.join(RAW_DATA_PATH, model, RESULTS_FOLDERS[ALPACA_EVAL_VERSION], "annotations.json")
        with open(results_file, 'r') as f:
            data = json.load(f)
        all_data.append(data)
    except FileNotFoundError:
        path = os.path.join(RAW_DATA_PATH, model)
        exceptions[model] = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]

# Delete models w/o data from models
models = [model for model in models if model not in list(exceptions.keys())]
instructions = [all_data[0][j]["instruction"] for j, _ in enumerate(all_data[0])]

data_final =  {version: {
                         "correctness": [],
                         "missing_data": [],
                         },
                        }

for i, _ in enumerate(all_data):
    model_correctness = []
    missing_data = []

    for j, _ in enumerate(all_data[i]):
        # Make sure that all data points are aligned:
        assert all_data[i][j]["instruction"] == instructions[j]
        try:
            correctness = all_data[i][j]["preference"] - SHIFT_VALUE
            missing = 0
        except TypeError:
            correctness = 0
            missing = 1

        missing_data.append(missing)
        model_correctness.append(correctness)
    
    data_final[version]["correctness"].append(model_correctness)
    data_final[version]["missing_data"].append(missing_data)

data_final[version]["correctness"] = np.array(data_final[version]["correctness"])
data_final[version]["missing_data"] = np.array(data_final[version]["missing_data"])

alpaca_eval_results = {"data": data_final,
                       "models": models,
                       "instructions": instructions}

with open(f'{version}.pickle', 'wb') as f:
    pkl.dump(alpaca_eval_results, f)
