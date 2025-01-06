import os
import json
import csv

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
exp_path = os.path.join(current_directory, "exp_main")
res_path = os.path.join(exp_path, "eda_tm_results")
output_csv = os.path.join(current_directory, "exp_main", "comparison_eda_tm_results.csv")

# Define categories to compare
datasets = ["xsum", "squad"]
source_models = ["gpt2-xl", "opt-2.7b", "gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]
detect_methods = {
    "fast_detect": ['sampling_discrepancy'],
    "baseline": ['likelihood','rank','logrank','entropy'],
    "dna_gpt": [''],
    "detect_gpt": ['perturbation_100'],
    "detect_llm": ['lrr','npr'],
    }


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
                            continue
                        for file in os.listdir(method_path):
                            if file.startswith(f"{dataset}_{model}") and (metric == '' or file.endswith(f"{metric}.json")):
                                filepath = os.path.join(method_path, file)
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
                                    "aug_roc_auc": data["metrics augmented"].get("roc_auc2", None),
                                    "aug_pr_auc": data["pr_metrics augmented"].get("pr_auc2", None),
                                    "aug_loss": data.get("loss augmented", None),
                                }
                                results.append(base_metrics)

# Write results to CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ["env", "dataset", "model", "method", "metric", "org_roc_auc", "org_pr_auc", "org_loss", "aug_roc_auc", "aug_pr_auc", "aug_loss"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow({key: result.get(key, None) for key in fieldnames})

print(f"Results successfully written to {output_csv}")
print(f"Total number of results collected: {counter}")