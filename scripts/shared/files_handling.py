import os
import json
import numpy as np
import sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
from metrics import get_roc_metrics, get_precision_recall_metrics
sys.path.pop(0)

def append_change_to_file(output_file, data,data_name):
    data_file = f"{output_file}.raw_data.json"
    # load existing data from the file
    if os.path.exists(data_file):
         with open(data_file, "r") as fin:
            existing_data = json.load(fin)
    else:
        raise FileNotFoundError(f"Data file {data_file} does not exist.")
    # append augmented data
    if data_name in existing_data:
        print("Augmented data already exists. It will be overwritten.")
    existing_data[data_name] = data
    # save updated data back to the file
    with open(data_file, "w") as fout:
        json.dump(existing_data, fout, indent=4)
        print(f"Augmented data appended and saved into {data_file}.")

def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

def save_embedding_results(output_file, name, embedding, predictions):
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    print(f"{embedding} mean/std: {np.mean(predictions['perturb']):.2f}/{np.std(predictions['perturb']):.2f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])

    fpr_model, tpr_model, roc_auc_model = get_roc_metrics(predictions['real'], predictions['perturb'])
    p_model, r_model, pr_auc_model = get_precision_recall_metrics(predictions['real'], predictions['perturb'])
    
    embedding_metrics = {
        'roc_auc': roc_auc_model,
        'pr_auc': pr_auc_model,
        'loss': 1 - pr_auc_model
    }

    print(f"Criterion {name}_threshold ROC AUC sampled: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    print(f"Criterion {name}_threshold ROC AUC {embedding}: {roc_auc_model:.4f}, PR AUC: {pr_auc_model:.4f}")

    # Results
    results_file = f'{output_file}.{name}.json'
    results_data = {
        'name': f'{name}',
        'metrics': {'roc_auc': roc_auc},
        'pr_metrics': {'pr_auc': pr_auc},
        'loss': 1 - pr_auc,
        f'{embedding}_metrics': embedding_metrics
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=4)