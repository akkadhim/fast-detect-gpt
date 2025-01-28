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
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics
from data_builder import load_data
from scripts.embedding import Embedding

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

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
    return ranks.mean().item()

# Log-Likelihood Log-Rank Ratio
def get_lrr(args, scoring_model, scoring_tokenizer, text, perturbs):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        logits = scoring_model(**tokenized).logits[:, :-1]
        likelihood = get_likelihood(logits, labels)
        logrank = get_logrank(logits, labels)
        return - likelihood / logrank

# Normalized Log-Rank Perturbation
def get_npr(args, scoring_model, scoring_tokenizer, text, perturbs):
    with torch.no_grad():
        tokenized = scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        logits = scoring_model(**tokenized).logits[:, :-1]
        logrank = get_logrank(logits, labels)
        # perturbations
        logranks = []
        for perturb in perturbs:
            tokenized = scoring_tokenizer(perturb, return_tensors="pt", return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            logits = scoring_model(**tokenized).logits[:, :-1]
            logranks.append(get_logrank(logits, labels))
        # npr
        return np.mean(logranks) / logrank

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    
    n_perturbations = args.n_perturbations
    perturb_name = f'perturbation_{n_perturbations}'
    data_perturbation = load_data(f'{args.dataset_file}.{args.mask_filling_model_name}.{perturb_name}')
    
    # eval criterions
    criterion_fns = {'lrr': get_lrr, 'npr': get_npr}
    augmentation_models = Embedding().MODELS
    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        eval_results = []
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]
            
            augmentor_texts = {augmentor: data[f"augmented_{augmentor}"][idx] for augmentor in augmentation_models}
            
            perturb_entry = data_perturbation[idx]
            perturbed_original = perturb_entry["perturbed_original"]
            perturbed_sampled = perturb_entry["perturbed_sampled"]
            perturbed_augmentor_texts = {
                augmentor: perturb_entry[f"perturbed_augmented_{augmentor}"] for augmentor in augmentation_models
            }
            
            original_crit = criterion_fn(args, scoring_model, scoring_tokenizer, original_text, perturbed_original)
            sampled_crit = criterion_fn(args, scoring_model, scoring_tokenizer, sampled_text, perturbed_sampled)
            augmentor_criteria = {
                augmentor: criterion_fn(args, scoring_model, scoring_tokenizer, augmentor_texts[augmentor], 
                                        perturbed_augmentor_texts[augmentor])
                for augmentor in augmentation_models
            }
            # result
            eval_result = {
                "original": {
                    "text": original_text,
                    "crit": original_crit
                },
                "sampled": {
                    "text": sampled_text,
                    "crit": sampled_crit
                },
                "augmentors": {
                    augmentor: {
                        "text": augmentor_texts[augmentor],
                        "crit": augmentor_criteria[augmentor]
                    }
                    for augmentor in augmentation_models
                }
            }          

            for augmentor, crit in augmentor_criteria.items():
                eval_result[f"augmented_{augmentor}"] = augmentor_texts[augmentor]
                eval_result[f"augmented_crit_{augmentor}"] = crit

            eval_results.append(eval_result)

        # compute prediction scores for real/sampled passages
        predictions = {
            'real': [x["original"]["crit"] for x in eval_results],  # Access 'crit' from 'original'
            'samples': [x["sampled"]["crit"] for x in eval_results],  # Access 'crit' from 'sampled'
            'augmentors': {
                augmentor: [
                    x["augmentors"][augmentor]["crit"] for x in eval_results
                ]  # Access 'crit' for each augmentor
                for augmentor in augmentation_models
            }
        }
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        
        # Compute metrics for each augmentor
        augmentor_metrics = {}
        for augmentor in augmentation_models:
            fpr_aug, tpr_aug, roc_auc_aug = get_roc_metrics(predictions['real'], predictions['augmentors'][augmentor])
            p_aug, r_aug, pr_auc_aug = get_precision_recall_metrics(predictions['real'], predictions['augmentors'][augmentor])

            augmentor_metrics[augmentor] = {
                'roc_auc': roc_auc_aug,
                'fpr': fpr_aug,
                'tpr': tpr_aug,
                'pr_auc': pr_auc_aug,
                'precision': p_aug,
                'recall': r_aug,
                'loss': 1 - pr_auc_aug
            }

        # Log results
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        for augmentor, metrics in augmentor_metrics.items():
            print(f"Augmentor {augmentor} ROC AUC: {metrics['roc_auc']:.4f}, PR AUC: {metrics['pr_auc']:.4f}")

        # Save results to file
        results_file = f'{args.output_file}.detect_llm.{name}.json'
        results = {
            'name': f'{name}_threshold',
            'info': {'n_samples': n_samples},
            'predictions': predictions,
            'raw_results': eval_results,
            'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
            'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
            'loss': 1 - pr_auc,
            'augmentor_metrics': augmentor_metrics,
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
    parser.add_argument('--n_perturbations', type=int, default=100)
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-3b")
    args = parser.parse_args()

    experiment(args)
