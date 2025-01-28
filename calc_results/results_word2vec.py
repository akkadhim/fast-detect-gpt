import os
import json
import csv

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
exp_path = os.path.join(current_directory, "exp_main")
res_path = os.path.join(exp_path, "results/word2vec_perturb")
output_csv = os.path.join(current_directory,"exp_main", "perturb_word2vec_results.csv")

# Define categories to compare
datasets = ["writing", "xsum", "squad"]
source_models = ["gpt2-xl", "opt-2.7b", "gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]
embedding_sources = ["word2vec"]
detect_methods = {
    "fast_detect": ['sampling_discrepancy'],
    # "baseline": ['likelihood','rank','logrank','entropy'],
    }
experiments = ["perturb_word2vec_percent","perturb_word2vec_threshold"]

# Collect results
results = []
counter = 0

for env in ["white", "black"]:
    for dataset in datasets:
        for model in source_models:
            for method, metrics in detect_methods.items():
                if metrics:
                    for metric in metrics:
                        # print(f"Collecting results for {dataset}_{model}.{method}.{metric}.json")
                        counter += 1
                        method_path = os.path.join(res_path, env, method)
                        if not os.path.exists(method_path):
                            print(f"Method path {method_path.replace(exp_path, '')} does not exist. Skipping...")
                            continue
                        for file in os.listdir(method_path):
                            if file.startswith(f"{dataset}_{model}") and (metric == '' or file.endswith(f".{metric}.json")):
                                filepath = os.path.join(method_path, file)
                                print(f"Collecting results for {filepath.replace(exp_path, '')}...")
                                with open(filepath, 'r') as f:
                                    data = json.load(f)

                                # Extract metrics
                                base_metrics  = {
                                    "env": env,
                                    "dataset": dataset,
                                    "model": model,
                                    "method": method,
                                    "metric": metric,
                                    "org_roc_auc": data["metrics"].get("roc_auc", None),
                                    "org_pr_auc": data["pr_metrics"].get("pr_auc", None),
                                    "org_loss": data.get("loss", None),
                                }
                                

                                if "augmentor_metrics" in data:
                                    for experiment in experiments:
                                        experiment_data = data["augmentor_metrics"][experiment]
                                        for value, perturb_data in experiment_data.items():
                                            # Only extract relevant augmentor metrics (roc_auc, pr_auc, loss)
                                            augmentor_metrics = {
                                                "perturb_name": experiment,
                                                "perturb_value": value,
                                                "roc_auc": perturb_data.get("roc_auc", None),
                                                "pr_auc": perturb_data.get("pr_auc", None),
                                                "loss": perturb_data.get("loss", None),
                                            }
                                            results.append({**base_metrics, **augmentor_metrics})
                                else:
                                    print(f"No augmentor metrics found in {file}")

# Write results to CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ["env", "dataset", "model", "method", "metric", "org_roc_auc", "org_pr_auc", "org_loss", "perturb_name", "perturb_value", "roc_auc", "pr_auc", "loss"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow({key: result.get(key, None) for key in fieldnames})

print(f"Results successfully written to {output_csv}")
print(f"Total number of results collected: {counter}")