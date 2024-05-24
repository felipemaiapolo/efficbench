import json
import os
import subprocess
from glob import glob 
import pickle as pkl
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

RAW_DATA_PATH = "./alpaca_eval/results"
CLONE_PATH = "./alpaca_eval"
ALPACA_EVAL_VERSION = "2.0"
RESULTS_FOLDERS = {"1.0": "alpaca_eval_gpt4",
                   "2.0": "weighted_alpaca_eval_gpt4_turbo",
                   }
SHIFT_VALUE = 1 # 0 for no shifting, 1 for shifting into [0,1]

def get_last_commit_date(file_path: str) -> Optional[str]:
    """Get the last commit date of a file in a git repository.
    Headsup: This function assumes that current working directory is the root of the git repo!

    Args:
        file_path (str): The path to the file.

    Returns:
        Optional[str]: The last commit date or None if an error occurred.
    """
    try:
        # Use '--' to separate the file path
        output = subprocess.check_output(['git', 'log', '-1', '--format=%cd', '--', file_path], cwd='./')
        return output.strip().decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_commit_dates_models(model_names: List[str]) -> Dict[str, datetime]:
    """Get commit dates for models.

    Args:
        model_names (List[str]): List of model names.

    Returns:
        Dict[str, datetime]: Dictionary of model names and their last commit dates.
    """
    dates = {}
    models_path = os.path.join('./', 'results')
    model_names = [name for name in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, name))]

    for model in model_names:
        file_path = os.path.join(models_path, f'{model}/model_outputs.json')
        dates[model]=(datetime.strptime(get_last_commit_date(file_path), '%a %b %d %H:%M:%S %Y %z'))

    return dates

def main() -> None:

    version = 'alpaca_v2' if ALPACA_EVAL_VERSION == '2.0' else 'alpaca_v1'
    directories = glob(os.path.join(RAW_DATA_PATH,  "*"))
    models = [path.split(os.sep)[-1] for path in directories if os.path.isdir(path)]

    # Load data, log in exceptions if data for model is missing
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

    # Exract data from dictionary; shift data; log missing data
    skip_models = []
    for i, _ in enumerate(all_data):
        model_correctness = []
        missing_data = []

        if len(all_data[i])==len(instructions):
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
        else:
            skip_models.append((i, models[i],len(all_data[i])))
            models.remove(models[i])

    data_final[version]["correctness"] = np.array(data_final[version]["correctness"]).T
    data_final[version]["missing_data"] = np.array(data_final[version]["missing_data"]).T

    alpaca_eval_results = {"data": data_final,
                           "models": models,
                           "skip":skip_models,
                           "instructions": instructions}

    # Order data according to last commit date
    try:
        os.chdir(CLONE_PATH)
        dates = get_commit_dates_models(models)
        os.chdir('../')
    except:
        os.chdir('../')

    dates = [dates[model] for model in alpaca_eval_results['models']]
    order = np.argsort(np.array(dates))[::-1]


    alpaca_eval_results['data'][version]['correctness'] = alpaca_eval_results['data'][version]['correctness'][:,order]
    alpaca_eval_results['models'] = np.array(alpaca_eval_results['models'])[order].tolist()

    if ALPACA_EVAL_VERSION == '2.0':
        # Deleting the reference model out of the list
        benchmark_ind = np.argmax([m=='gpt4_1106_preview' for m in alpaca_eval_results['models']])
        alpaca_eval_results['data'][version]['correctness'] = np.delete(alpaca_eval_results['data'][version]['correctness'], benchmark_ind, axis=1)
        alpaca_eval_results['models'].pop(benchmark_ind)

    # Save data
    with open(f'data/{version}.pickle', 'wb') as f:
        pkl.dump(alpaca_eval_results, f)

    print(f"Successfully processed {version} data.")

if __name__ == "__main__":
    main()
