# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Import model from the parent directory
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics

# Restore the original sys.path to avoid affecting other imports
sys.path.pop(0)

import numpy as np
import torch
import tqdm
import argparse
import json
from data_builder import load_data

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def experiment(args):
    # Load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()

    # Load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    # Evaluate criterion
    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    results = []
    experiments = ["perturb_word2vec_percent","perturb_word2vec_threshold"]
    
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        augmentor_criteria = {}
        
        # ----- Original text -----
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            original_crit = criterion_fn(logits_ref, logits_score, labels)
        
        # ----- Sampled text -----
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels)
        
        # ----- Augmented text -----
        for exp_name in experiments:
            experiment_set = data[exp_name]
            if exp_name not in augmentor_criteria:
                augmentor_criteria[exp_name] = {}
                
            for experiment in experiment_set:
        
                perturbed_text = data[exp_name][experiment][idx]
                tokenized = scoring_tokenizer(perturbed_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                labels = tokenized.input_ids[:, 1:]

                with torch.no_grad():
                    logits_score = scoring_model(**tokenized).logits[:, :-1]
                    if args.reference_model_name == args.scoring_model_name:
                        logits_ref = logits_score
                    else:
                        tokenized = reference_tokenizer(perturbed_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                        assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer mismatch."
                        logits_ref = reference_model(**tokenized).logits[:, :-1]
                    augmentor_criteria[exp_name][experiment] = criterion_fn(logits_ref, logits_score, labels)

        # Append results
        results.append({
            "original_crit": original_crit,
            "sampled_crit": sampled_crit,
            "augmentors": augmentor_criteria
        })
        
    
    # perturbing_percents = [1, 2, 5, 10, 20]
    # similarity_thresholds = ['min', 'mid', 'high']
    # Compute prediction scores for real/sampled/augmented passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
        'augmentors': {exp_name: {
            experiment: [x["augmentors"][exp_name][experiment] for x in results]
            for experiment in data[exp_name]
        } for exp_name in experiments}
    }

    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")

    for exp_name in experiments:
        for experiment in data[exp_name]:
            scores = predictions['augmentors'][exp_name][experiment]
            print(f"{exp_name} ({experiment}) mean/std: {np.mean(scores):.2f}/{np.std(scores):.2f}")

    # Compute metrics
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])

    augmentor_metrics = {}
    for exp_name in experiments:
        augmentor_metrics[exp_name] = {}
        for experiment in data[exp_name]:
            scores = predictions['augmentors'][exp_name][experiment]
            fpr_model, tpr_model, roc_auc_model = get_roc_metrics(predictions['real'], scores)
            p_model, r_model, pr_auc_model = get_precision_recall_metrics(predictions['real'], scores)
            augmentor_metrics[exp_name][experiment] = {
                'roc_auc': roc_auc_model,
                'fpr': fpr_model,
                'tpr': tpr_model,
                'pr_auc': pr_auc_model,
                'precision': p_model,
                'recall': r_model,
                'loss': 1 - pr_auc_model
            }

    print(f"Criterion {name}_threshold ROC AUC sampled: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    for exp_name, experiments in augmentor_metrics.items():
        for experiment, metrics in experiments.items():
            print(f"Criterion {name}_threshold ROC AUC {exp_name} ({experiment}): {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}")

    # Save results
    results_file = f'{args.output_file}.fast_detect.{name}.json'
    results_data = {
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc,
        'augmentor_metrics': augmentor_metrics
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="exp_main/results/word2vec_perturb/white/fast_detect/xsum_gpt2-xl")
    parser.add_argument('--dataset', type=str, default="xsum") 
    parser.add_argument('--dataset_file', type=str, default="exp_main/data/word2vec_perturb/xsum_gpt2-xl")
    parser.add_argument('--reference_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--discrepancy_analytic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
