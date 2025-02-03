# Copyright (c) Guangsheng Bao & Ahmed Khalid.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import random
import argparse
import os
from tqdm import tqdm
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, parent_dir)
from scripts.shared.embedding import Embedding
from scripts.shared.synonum import *
from scripts.shared.files_handling import *
sys.path.pop(0)

def preprocess_text(text):
    return text


class DataBuilder:
    def __init__(self, args):
        self.args = args
    
    def _trim_to_shorter_length(self, texta, textb, textc = None):
        # truncate to shorter of o and s
        shorter_length = min(len(texta.split(' ')), len(textb.split(' ')), len(textc.split(' ')))
        texta = ' '.join(texta.split(' ')[:shorter_length])
        textb = ' '.join(textb.split(' ')[:shorter_length])
        if textc:
            textc = ' '.join(textc.split(' ')[:shorter_length])
            return texta, textb, textc
        return texta, textb
    
    def _sample_from_embedding(self, texts, embedding):
        changed_count_per_text = 0
        decoded = []
        for doc in texts:
            perturbed_data, changed_count = synonum_by_embedding(doc, embedding, self.args.perturb_percnt)
            decoded.append(perturbed_data)
            changed_count_per_text += changed_count
        return decoded, changed_count_per_text

    def generate_perturb_samples(self, batch_size):
        existing_data = load_data(args.output_file)
        original_data = existing_data.get("original", [])
        sampled_data = existing_data.get("sampled", [])

        if not original_data or not sampled_data:
            raise ValueError("Original and sampled data must be present in the loaded file to generate augmented samples.")
        if len(original_data) != len(sampled_data):
            raise ValueError("Mismatch between original and sampled data lengths.")

        embedding = Embedding(self.args.embedding)
        perturbed_data = []
        changed_count_per_ds = 0
        for batch in tqdm(range(len(original_data) // batch_size), desc="Processing batches"):
            # print('Generating augmented samples for batch', batch, 'of', len(original_data) // batch_size)
            original_batch = original_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_batch = sampled_data[batch * batch_size:(batch + 1) * batch_size]
            perturbed_batch, changed_count = self._sample_from_embedding(sampled_batch, embedding)
            changed_count_per_ds += changed_count
            for o, s, a in zip(original_batch, sampled_batch, perturbed_batch):               
                o, s, a = self._trim_to_shorter_length(o, s, a)
                perturbed_data.append(a)
        
        return perturbed_data, changed_count_per_ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="exp_main/data/hybrid/writing_gpt2-xl")
    parser.add_argument('--dataset', type=str, default="writing")
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--openai_base', type=str, default=None)
    parser.add_argument('--openai_key', type=str, default=None)
    parser.add_argument('--openai_model', type=str, default=None)  # davinci, gpt-3.5-turbo, gpt-4
    parser.add_argument('--base_model_name', type=str, default="gpt2-xl")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    parser.add_argument('--embedding', type=str, default="word2vec")
    parser.add_argument('--perturb_percnt', type=int, default=5)
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    print(f"Using cache dir {args.cache_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading dataset {args.dataset}...')
    print(f'Embedding using {args.embedding}...')
    data_builder = DataBuilder(args)
    perturbed_data, changed_count = data_builder.generate_perturb_samples(batch_size=args.batch_size)
    append_change_to_file(args.output_file, perturbed_data,data_name=f"perturb_{args.embedding}")
    append_change_to_file(args.output_file, changed_count,data_name=f"perturb_{args.embedding}_count")