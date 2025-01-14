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
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec

def append_change_to_file(output_file, data):
    data_file = f"{output_file}.raw_data.json"
    # load existing data from the file
    if os.path.exists(data_file):
         with open(data_file, "r") as fin:
            existing_data = json.load(fin)
    else:
        raise FileNotFoundError(f"Data file {data_file} does not exist.")
    # append augmented data
    data_name = "perturb_word2vec"
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

def get_synonyms(word):
    """Fetch synonyms of the given word using WordNet."""
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def calculate_similarity(word, synonyms, model):
    """Calculate similarity between the word and its synonyms using Word2Vec."""
    similarities = {}
    for synonym in synonyms:
        if synonym in model.wv.key_to_index and word in model.wv.key_to_index:
            similarity = model.wv.similarity(word, synonym)
            similarities[synonym] = similarity
    return similarities

class DataBuilder:
    def __init__(self, args):
        self.args = args
        model_path = "embedding_files/datasets/word2vec_1billion/custom_word2vec.model"
        try:
            self.word2vec = Word2Vec.load(model_path)
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            exit()
        
    def _trim_to_shorter_length(self, texta, textb, textc = None):
        # truncate to shorter of o and s
        shorter_length = min(len(texta.split(' ')), len(textb.split(' ')), len(textc.split(' ')))
        texta = ' '.join(texta.split(' ')[:shorter_length])
        textb = ' '.join(textb.split(' ')[:shorter_length])
        if textc:
            textc = ' '.join(textc.split(' ')[:shorter_length])
            return texta, textb, textc
        return texta, textb

    def _synonum_replace(self, doc, percentage):
        aug_text = '' 
        sentences = doc.split('. ')
        for sentence in sentences:
            original_words = sentence.split()  # Keeps original sentence structure
            new_sentence = original_words.copy()

            # Normalize the sentence for replacement logic
            normalized_sentence = text_organizer.get_only_chars(sentence)
            words = normalized_sentence.split()
            words = [word for word in words if word != '']  # Filter out empty strings

            # Get candidate words for replacement
            random_word_list = list(set([word for word in words if word.lower() not in text_organizer.stop_words]))
            random.shuffle(random_word_list)

            # Calculate the number of words to replace based on the percentage
            total_words = len(random_word_list)
            num_to_replace = max(1, int((percentage / 100) * total_words))  # At least one word

            # Replace the calculated number of words with their synonyms
            num_replaced = 0
            for random_word in random_word_list:
                synonyms = get_synonyms(random_word)
                if synonyms:
                    similarities = calculate_similarity(random_word, synonyms, self.word2vec)
                    if similarities:
                        synonym = min(similarities, key=similarities.get)
                        if synonym:
                            # Replace only the matching words, preserving the original format
                            new_sentence = [
                                synonym if word.lower() == random_word else word
                                for word in new_sentence
                            ]
                            num_replaced += 1
                        # Stop after replacing the target number of words
                        if num_replaced >= num_to_replace:
                            break

            aug_text = aug_text + ' '.join(new_sentence) + '. '
        return aug_text
    
    def _sample_from_word2vec(self, texts):
        decoded = []
        for doc in texts:
            decoded.append(self._synonum_replace(doc,args.perturbing_percnt))
        return decoded

    def generate_perturbing_samples(self, batch_size):
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
            augmented_batch = self._sample_from_word2vec(sampled_batch)
            for o, s, a in zip(original_batch, sampled_batch, augmented_batch):               
                o, s, a = self._trim_to_shorter_length(o, s, a)
                augmented_data.append(a)
                
        return augmented_data

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
    parser.add_argument('--perturbing_percnt', type=int, default=5)
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
    append_change_to_file(args.output_file, perturbing_data)
