# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics
from scripts.embedding import Embedding

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

def get_rank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1 # convert to 1-indexed rank
    return -ranks.mean().item()

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_entropy(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()


def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # eval criterions
    criterion_fns = {'likelihood': get_likelihood,
                     'rank': get_rank,
                     'logrank': get_logrank,
                     'entropy': get_entropy}
    augmentation_models = Embedding().MODELS
    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        eval_results = []
        
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]
            augmentor_criteria = {}
            # original text
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits = scoring_model(**tokenized).logits[:, :-1]
                original_crit = criterion_fn(logits, labels)
            # sampled text
            tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits = scoring_model(**tokenized).logits[:, :-1]
                sampled_crit = criterion_fn(logits, labels)
            # augmented text
            for model_name in augmentation_models:
                augmented_text = data[f"augmented_{model_name}"][idx]
                tokenized = scoring_tokenizer(augmented_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits = scoring_model(**tokenized).logits[:, :-1]
                    augmentor_criteria[model_name] = criterion_fn(logits, labels)

            # result
            eval_results.append({
                "original_crit": original_crit,
                "sampled_crit": sampled_crit,
                "augmentors_crit": augmentor_criteria 
            })

        # compute prediction scores for real/sampled passages
        predictions = {
            'real': [x["original_crit"] for x in eval_results],
            'samples': [x["sampled_crit"] for x in eval_results],
            'augmentors_crit': {model: [x["augmentors_crit"][model] for x in eval_results] for model in augmentation_models}
        }

        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        
        augmentor_metrics = {}
        for model in augmentation_models:
            fpr_model, tpr_model, roc_auc_model = get_roc_metrics(predictions['real'], predictions['augmentors_crit'][model])
            p_model, r_model, pr_auc_model = get_precision_recall_metrics(predictions['real'], predictions['augmentors_crit'][model])
            augmentor_metrics[model] = {
                'roc_auc': roc_auc_model,
                'fpr': fpr_model,
                'tpr': tpr_model,
                'pr_auc': pr_auc_model,
                'precision': p_model,
                'recall': r_model,
                'loss': 1 - pr_auc_model
            }

        print(f"Criterion {name}_threshold ROC AUC sampled: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        for model, metrics in augmentor_metrics.items():
            print(f"Criterion {name}_threshold ROC AUC {model}: {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}")

        # log results
        results_file = f'{args.output_file}.baseline.{name}.json'
        results = { 
            'name': f'{name}_threshold',
            'info': {'n_samples': n_samples},
            'predictions': predictions,
            'raw_results': eval_results,
            'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
            'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
            'loss': 1 - pr_auc,
            'augmentor_metrics': augmentor_metrics
        }
        
        with open(results_file, 'w') as fout:
            json.dump(results, fout, indent=4)
            print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--output_file', type=str, default="exp_main/results/white/fast/xsum_gpt2-xl")
    parser.add_argument('--dataset', type=str, default="xsum") 
    parser.add_argument('--dataset_file', type=str, default="exp_main/data/xsum_gpt2-xl")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
