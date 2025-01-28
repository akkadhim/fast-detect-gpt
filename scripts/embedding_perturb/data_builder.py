# Copyright (c) Guangsheng Bao.
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
import scripts.shared.custom_datasets as custom_datasets
from scripts.shared.model import load_tokenizer, load_model
from eda import *
from tqdm import tqdm
from scripts.embedding import Embedding

def preprocess_text(text):
    return text

def save_data(output_file, args, data):
    # write args to file
    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")
    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")

def append_augmented_to_file(output_file, augmented_data, augmentor):
    data_file = f"{output_file}.raw_data.json"
    # load existing data from the file
    if os.path.exists(data_file):
         with open(data_file, "r") as fin:
            existing_data = json.load(fin)
    else:
        raise FileNotFoundError(f"Data file {data_file} does not exist.")
    # append augmented data
    augmented_data_name = "augmented_" + augmentor
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
        if(args.bypass_genearation == "False"):
            self.base_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir)
            self.base_model = None if args.openai_model else load_model(args.base_model_name, args.device, args.cache_dir)

            # trim to shorter length
    
    def _trim_to_shorter_length(self, texta, textb, textc = None):
        # truncate to shorter of o and s
        shorter_length = min(len(texta.split(' ')), len(textb.split(' ')), len(textc.split(' ')))
        texta = ' '.join(texta.split(' ')[:shorter_length])
        textb = ' '.join(textb.split(' ')[:shorter_length])
        if textc:
            textc = ' '.join(textc.split(' ')[:shorter_length])
            return texta, textb, textc
        return texta, textb
    
    def _openai_sample(self, prefix):
        def _drop_last_word(text):
            return ' '.join(text.split(' ')[:-1])

        import openai
        assert self.args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = self.args.openai_key
        if self.args.openai_base is not None:
            openai.api_base = self.args.openai_base

        if self.args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
            prefix = _drop_last_word(prefix)

        # sample from the openai model
        kwargs = {"max_tokens": 200}
        if self.args.do_top_p:
            kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            kwargs['top_k'] = self.args.top_k
        elif self.args.do_temperature:
            kwargs['temperature'] = self.args.temperature

        if self.args.openai_model == 'davinci':
            kwargs["engine"] = self.args.openai_model
            response = openai.Completion.create(prompt=f"{prefix}", **kwargs)
            return prefix + response['choices'][0]['text']

        elif self.args.openai_model in ['gpt-3.5-turbo', 'gpt-4']:
            roles = {'xsum': 'You are a News writer.',
                     'writing': 'You are a Fiction writer.',
                     'pubmed': 'You are a Technical writer.'}
            prompts = {'xsum': 'Please write an article with about 150 words starting exactly with:',
                       'writing': 'Please write an article with about 150 words starting exactly with:',
                       'pubmed': 'Please answer the question in about 50 words.'}
            messages = [
                {'role': 'system', 'content': roles[self.args.dataset]},
                {'role': 'user', 'content': f'{prompts[self.args.dataset]} {prefix}'},
            ]
            kwargs["model"] = self.args.openai_model
            kwargs["messages"] = messages
            response = openai.ChatCompletion.create(**kwargs)
            response = response['choices'][0]['message']['content']
            # ChatGPT may repeat the prefix
            if response.startswith(prefix[:20]):
                return response
            return prefix + ' ' + response

        else:
            raise NotImplementedError

    # sample from base_model using ****only**** the first 30 tokens in each example as context
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
    
    def _sample_from_augmentor(self, texts):
        decoded = []
        augmentor = Embedding(args.augmentor)
        if(args.augmentor in augmentor.MODELS):
            for doc in texts:
                decoded.append(augmentor.document_perturb(doc,args.augment_percnt))
        else:
            alpha_sr = 0
            alpha_ri = 0
            alpha_rs = 0
            alpha_rd = 0 
            for text in tqdm(texts, desc="Augmenting texts using EDA"):
                aug_text = '' 
                sentences = text.split('. ')
                for sentence in sentences:
                    aug_sentence = eda(augmentor, sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd)
                    aug_text = aug_text + aug_sentence + '. '
                decoded.append(aug_text)
        return decoded

    def generate_augmented_samples(self, batch_size):
        existing_data = load_data(args.output_file)
        original_data = existing_data.get("original", [])
        sampled_data = existing_data.get("sampled", [])

        if not original_data or not sampled_data:
            raise ValueError("Original and sampled data must be present in the loaded file to generate augmented samples.")
        if len(original_data) != len(sampled_data):
            raise ValueError("Mismatch between original and sampled data lengths.")

        augmented_data = []
        for batch in tqdm(range(len(original_data) // batch_size), desc="Processing batches"):
            # print('Generating augmented samples for batch', batch, 'of', len(original_data) // batch_size)
            original_batch = original_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_batch = sampled_data[batch * batch_size:(batch + 1) * batch_size]
            augmented_batch = self._sample_from_augmentor(sampled_batch)
            for o, s, a in zip(original_batch, sampled_batch, augmented_batch):               
                o, s, a = self._trim_to_shorter_length(o, s, a)
                augmented_data.append(a)
                
        return augmented_data

    def generate_samples(self, raw_data, batch_size):
        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]
        
        data = {
            "original": [],
            "sampled": []
        }
        for batch in range(len(raw_data) // batch_size):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_text = self._sample_from_model(original_text, min_words=30 if self.args.dataset in ['pubmed'] else 55)

            for o, s in zip(original_text, sampled_text):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)
                    o = o.replace(custom_datasets.SEPARATOR, ' ')
                
                # Trim the original and sampled texts to match lengths
                o, s = self._trim_to_shorter_length(o, s)
                
                # Add to the data
                data["original"].append(o)
                data["sampled"].append(s)
        
        return data

def generate_data(args, dataset, key):
    data_builder = DataBuilder(args)
    # strip newlines from each example; replace one or more newlines with a single space
    def _strip_newlines(text):
        return ' '.join(text.split())

    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, args.cache_dir)
    else:
        data = custom_datasets.load_dataset(dataset, split='train', cache_dir=args.cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the base model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [_strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.shuffle(data)
    data = data[:5_000]

    # keep only examples with <= 512 tokens according to base_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = data_builder.base_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remaining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return data_builder.generate_augmented_samples(data[:args.n_samples], batch_size=args.batch_size)

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
    parser.add_argument('--bypass_genearation', type=str, default="True")
    parser.add_argument('--augmentor', type=str, default="bert")
    parser.add_argument('--augment_percnt', type=int, default=5)
    args = parser.parse_args()

    os.environ["XDG_CACHE_HOME"] = args.cache_dir
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    print(f"Using cache dir {args.cache_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Loading dataset {args.dataset}...')
    dataset_keys = {'xsum': 'document', 'squad': 'context', 'writing': 'document'}
    if(args.bypass_genearation == "False"):
        data = generate_data(args, args.dataset, dataset_keys[args.dataset] if args.dataset in dataset_keys else None)
        save_data(args.output_file, args, data)
    else:
        print(f'Augmenting using {args.augmentor}...')
        data_builder = DataBuilder(args)
        augmented_data = data_builder.generate_augmented_samples(batch_size=args.batch_size)
        append_augmented_to_file(args.output_file, augmented_data, args.augmentor)
