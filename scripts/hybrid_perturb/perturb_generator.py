# Copyright (c) Guangsheng Bao & Ahmed Khalid.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time

import numpy as np
import torch
import random
import argparse
import os
import json
import custom_datasets
from eda import *
from tqdm import tqdm
from scripts.embedding import Embedding
from synonum import *

def preprocess_text(text):
    return text

def append_augmented_to_file(output_file, augmented_data, embedding):
    data_file = f"{output_file}.raw_data.json"
    # load existing data from the file
    if os.path.exists(data_file):
         with open(data_file, "r") as fin:
            existing_data = json.load(fin)
    else:
        raise FileNotFoundError(f"Data file {data_file} does not exist.")
    # append augmented data
    augmented_data_name = "augmented_" + embedding
    if augmented_data_name in existing_data:
        print("Augmented data already exists. It will be overwritten.")
    existing_data[augmented_data_name] = augmented_data
    # save updated data back to the file
    with open(data_file, "w") as fout:
        json.dump(existing_data, fout, indent=4)
        print(f"Augmented data appended and saved into {data_file}.")
        
def save_perturb_data(output_file, perturbs):
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(perturbs, fout, indent=4)
        print(f"Prturbs data saved {data_file}.")

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

    def _sample_from_model(self, texts, min_words=55, prompt_tokens=30):
        # encode each text as a list of token ids
        if self.args.dataset == 'pubmed':
            texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        else:
            all_encoded = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
            all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

        if self.args.openai_model:
            # decode the prefixes back into text
            prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)

            decoded = []
            for idx, prefix in enumerate(prefixes):
                while idx >= len(decoded):
                    try:
                        decoded.append(self._openai_sample(prefix))
                    except Exception as ex:
                        print(ex)
                        print('Wait 10 minutes before retry ...')
                        time.sleep(600)

        else:
            self.base_model.eval()
            decoded = ['' for _ in range(len(texts))]

            # sample from the model until we get a sample with at least min_words words for each example
            # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
            tries = 0
            m = 0
            while m < min_words:
                if tries != 0:
                    print()
                    print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")
                    prefixes = self.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
                    for prefix, x in zip(prefixes, decoded):
                        if len(x.split()) == m:
                            print(prefix, '=>', x)

                sampling_kwargs = {}
                if self.args.do_top_p:
                    sampling_kwargs['top_p'] = self.args.top_p
                elif self.args.do_top_k:
                    sampling_kwargs['top_k'] = self.args.top_k
                elif self.args.do_temperature:
                    sampling_kwargs['temperature'] = self.args.temperature
                min_length = 50 if self.args.dataset in ['pubmed'] else 150
                outputs = self.base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True,
                                                   **sampling_kwargs, pad_token_id=self.base_tokenizer.eos_token_id,
                                                   eos_token_id=self.base_tokenizer.eos_token_id)
                decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                m = min(len(x.split()) for x in decoded)
                tries += 1

        return decoded
    
    def _sample_from_embedding(self, texts, embedding):
        decoded = []
        for doc in texts:
            decoded.append(synonum_by_embedding(doc, embedding, self.args.perturb_percnt))
        return decoded

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
        for batch in tqdm(range(len(original_data) // batch_size), desc="Processing batches"):
            # print('Generating augmented samples for batch', batch, 'of', len(original_data) // batch_size)
            original_batch = original_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_batch = sampled_data[batch * batch_size:(batch + 1) * batch_size]
            perturbed_batch = self._sample_from_embedding(sampled_batch, embedding)
            for o, s, a in zip(original_batch, sampled_batch, perturbed_batch):               
                o, s, a = self._trim_to_shorter_length(o, s, a)
                perturbed_data.append(a)
        
        return perturbed_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="exp_main/data/writing_gpt2-xl")
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
    parser.add_argument('--embedding', type=str, default="bert")
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
    perturbed_data = data_builder.generate_perturb_samples(batch_size=args.batch_size)
    append_augmented_to_file(args.output_file, perturbed_data, args.embedding)
