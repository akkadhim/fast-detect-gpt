# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)
from scripts.shared.model import load_tokenizer, load_model
from scripts.shared.files_handling import load_data
from scripts.shared.files_handling import save_embedding_results
sys.path.pop(0)

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
    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        eval_results = []
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]
            perturbed_text = data[f"perturb_{args.embedding}"][idx]
            
            perturb_entry = data_perturbation[idx]
            perturbed_original = perturb_entry["perturbed_original"]
            perturbed_sampled = perturb_entry["perturbed_sampled"]
            perturbed_embedding = perturb_entry["perturbed_embedding"]
            
            original_crit = criterion_fn(args, scoring_model, scoring_tokenizer, original_text, perturbed_original)
            sampled_crit = criterion_fn(args, scoring_model, scoring_tokenizer, sampled_text, perturbed_sampled)
            perturb_crit = criterion_fn(args, scoring_model, scoring_tokenizer, perturbed_text, perturbed_embedding)

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
                "perturbed": {
                    "text": perturbed_text,
                    "crit": perturb_crit
                }
            }          
            eval_results.append(eval_result)

        # compute prediction scores for real/sampled passages
        predictions = {
            'real': [x["original"]["crit"] for x in eval_results],  # Access 'crit' from 'original'
            'samples': [x["sampled"]["crit"] for x in eval_results],  # Access 'crit' from 'sampled'
            'perturb': [x["perturbed"]["crit"] for x in eval_results],  # Access 'crit' from 'perturbed'
        }
        save_embedding_results(args.output_file, f"detect_llm.{name}", args.embedding, predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="exp_main/results/hybrid/white/detect_llm/xsum_gpt2-xl")
    parser.add_argument('--dataset', type=str, default="xsum") 
    parser.add_argument('--dataset_file', type=str, default="exp_main/data/hybrid/xsum_gpt2-xl")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--n_perturbations', type=int, default=100)
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-3b")
    parser.add_argument('--embedding', type=str, default="word2vec")
    args = parser.parse_args()

    experiment(args)
