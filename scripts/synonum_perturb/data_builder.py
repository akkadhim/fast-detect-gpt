# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import argparse
import os
import json
from tqdm import tqdm
import nltk
import sys
from synonum import *
from scripts.embedding import Embedding

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

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
         
    def _sample_from_embedding(self, texts, embedding_model, perturbing_percnt, similarity_threshold):
        decoded = []
        for doc in texts:
            decoded.append(synonum_by_word2vec(doc, embedding_model, perturbing_percnt, similarity_threshold))
        return decoded
    
    def generate_perturbing_samples(self, batch_size):
        existing_data = load_data(args.output_file)
        original_data = existing_data.get("original", [])
        sampled_data = existing_data.get("sampled", [])
        
        if not original_data or not sampled_data:
            raise ValueError("Original and sampled data must be present in the loaded file to generate augmented samples.")
        if len(original_data) != len(sampled_data):
            raise ValueError("Mismatch between original and sampled data lengths.")

        embedding = Embedding(self.args.embedding)
        embedding_model = embedding.models[self.args.embedding]
        ######################################################
        # perturbing_percents = [1, 2, 5, 10, 20]
        perturbing_percents = [5]
        
        similarity_thresholds = ['min', 'mid', 'high']
        # similarity_thresholds = ['mid']
        
        # perturb_data = {percent: [] for percent in perturbing_percents}
        perturb_data = {similarity_threshold: [] for similarity_threshold in similarity_thresholds}
        ######################################################

        for batch in tqdm(range(len(original_data) // batch_size), desc="Processing batches"):
            original_batch = original_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_batch = sampled_data[batch * batch_size:(batch + 1) * batch_size]
            
            # Generate perturbations for each percentage
            for similarity_threshold in similarity_thresholds:
                for percent in perturbing_percents:
                    perturb_batch = self._sample_from_embedding(sampled_batch, embedding_model, percent, similarity_threshold)
                    for o, s, a in zip(original_batch, sampled_batch, perturb_batch):               
                        o, s, a = self._trim_to_shorter_length(o, s, a)
                        ######################################################
                        perturb_data[similarity_threshold].append(a)
                        # perturb_data_by_percent[percent].append(a)
                        ######################################################
                
        return perturb_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="exp_main/data/word2vec_perturb/writing_gpt2-xl")
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
    parser.add_argument('--perturbing_percnt', type=str, default='False')
    parser.add_argument('--similarity_threshold', type=str, default='True')
    parser.add_argument('--embedding', type=str, default="word2vec")
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    print(f"Using cache dir {args.cache_dir}")
    
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading dataset {args.dataset}...')
    data_builder = DataBuilder(args)
    perturbing_data = data_builder.generate_perturbing_samples(batch_size=args.batch_size)
    if args.perturbing_percnt == 'True':
        append_change_to_file(args.output_file, perturbing_data,data_name=f"perturb_{args.embedding}_percent")
    if args.similarity_threshold == 'True':
        append_change_to_file(args.output_file, perturbing_data,data_name=f"perturb_{args.embedding}_threshold")
