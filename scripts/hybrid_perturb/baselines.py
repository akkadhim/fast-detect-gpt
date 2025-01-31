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
# Add parent directory to sys.path
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
    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        eval_results = []
        
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
            original_text = data["original"][idx]
            sampled_text = data["sampled"][idx]
            perturbed_text = data[f"perturb_{args.embedding}"][idx]
            
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
            tokenized = scoring_tokenizer( perturbed_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits = scoring_model(**tokenized).logits[:, :-1]
                perturbed_crit = criterion_fn(logits, labels)

            # result
            eval_results.append({
                "original_crit": original_crit,
                "sampled_crit": sampled_crit,
                "perturbed_crit": perturbed_crit 
            })

        predictions = {
            'real': [x["original_crit"] for x in eval_results],
            'samples': [x["sampled_crit"] for x in eval_results],
            'perturb': [x["perturbed_crit"] for x in eval_results]
        }
        save_embedding_results(args.output_file, f"baseline.{name}", args.embedding, predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    parser.add_argument('--output_file', type=str, default="exp_main/results/hybrid/white/baseline/xsum_gpt2-xl")
    parser.add_argument('--dataset', type=str, default="xsum") 
    parser.add_argument('--dataset_file', type=str, default="exp_main/data/hybrid/xsum_gpt2-xl")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--embedding', type=str, default="word2vec")
    args = parser.parse_args()

    experiment(args)
