import os
import json
import csv

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
exp_path = os.path.join(current_directory, "exp_main")
res_path = os.path.join(exp_path, "results")
output_csv = os.path.join(current_directory, "comparison_results.csv")

# Define categories to compare
datasets = ["writing", "xsum", "squad"]
source_models = ["gpt2-xl", "opt-2.7b", "gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]
embedding_sources = ["glove", "fasttext", "word2vec", "tmae", "elmo", "bert"]
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
                            if file.startswith(f"{dataset}_{model}.{method}") and (metric == '' or file.endswith(".json")):
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
                                }

                                if "augmentor_metrics" in data:
                                    for augmentor, augmentor_data in data["augmentor_metrics"].items():
                                        # Only extract relevant augmentor metrics (roc_auc, pr_auc, loss)
                                        augmentor_metrics = {
                                            "augmentor": augmentor,
                                            "roc_auc": augmentor_data.get("roc_auc", None),
                                            "pr_auc": augmentor_data.get("pr_auc", None),
                                            "loss": augmentor_data.get("loss", None),
                                        }
                                        results.append({**base_metrics, **augmentor_metrics})
                                else:
                                    print(f"No augmentor metrics found in {file}")

# Write results to CSV
with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
    fieldnames = ["env", "dataset", "model", "method", "metric", "org_roc_auc", "org_pr_auc", "org_loss", "augmentor", "roc_auc", "pr_auc", "loss"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow({key: result.get(key, None) for key in fieldnames})

print(f"Results successfully written to {output_csv}")
print(f"Total number of results collected: {counter}")