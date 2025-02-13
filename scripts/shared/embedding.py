import numpy as np
import fasttext
from gensim.models import Word2Vec
import random
import tensorflow as tf
import tensorflow_hub as hub
import torch
from transformers import BertTokenizer, BertModel
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import os
import sys

parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)
import text_organizer
from directories import dicrectories
from tools import tools
sys.path.pop(0)

 
class Embedding:
    MODELS = ['glove', 'fasttext', 'word2vec', 'tmae', 'elmo', 'bert']
    SIMILAR_SIZE = 400
    SIMILAR_INDEX = 5
    
    def __init__(self, model_name = ""):
        self.model_name = model_name
        self.models = {
            "glove": None,
            "fasttext": None,
            "word2vec": None,
            "elmo": None,
            "tmae": None,
            "bert": None,
        }
        if model_name == "glove":
            self.load_glove_embeddings('embedding_files/datasets/glove_imdb_20k/vectors.txt')
        elif model_name == "fasttext":
            self.load_fasttext_model('embedding_files/datasets/fasttext_imdb_20k/fasttext_model.bin')
        elif model_name == "word2vec":
            self.load_word2vec_model('embedding_files/datasets/word2vec_imdb_20k/custom_word2vec.model')
        elif model_name == "elmo":
            self.load_elmo_model("https://tfhub.dev/google/elmo/3")
        elif model_name == "bert":
            self.load_bert_model("embedding_files/datasets/bert_imdb_20k")
        elif model_name == "tmae":
            self.load_tmae_model('embedding_files/datasets/tm_imdb_20k/vectorizer_X.pickle')
      
    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    
    def get_dynamic_index(self, similar_words):
        ratio = self.SIMILAR_INDEX / self.SIMILAR_SIZE
        actual_size = len(similar_words)
        dynamic_index = int(round(ratio * actual_size))
        return dynamic_index
    
    # GloVe
    def load_glove_embeddings(self, glove_file_path):
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        self.models["glove"] = embeddings_index
    
    def glove_word_replacement(self, word):
        glove_vectors = self.models["glove"]
        if word not in glove_vectors:
            return []
        word_vector = glove_vectors[word]
        
        sampled_vocab = random.sample(list(glove_vectors.keys()), self.SIMILAR_SIZE)
        similarities = {}
        for other_word in sampled_vocab:
            if other_word != word:
                other_vector = glove_vectors[other_word]
                similarities[other_word] = self.cosine_similarity(word_vector, other_vector)
        
        similar_words = sorted(similarities.items(), key=lambda item: item[1], reverse=False)
        
        dynamic_index = self.get_dynamic_index(similar_words)
        if 0 <= dynamic_index < len(similar_words):
            return similar_words[dynamic_index][0]
        else:
            return None 

    # FastText
    def load_fasttext_model(self, fasttext_path):
        self.models["fasttext"] = fasttext.load_model(fasttext_path)
    
    def fasttext_word_replacement(self, word):
        fasttext_vectors = self.models["fasttext"]
        if word in fasttext_vectors.words:
            similar_words = fasttext_vectors.get_nearest_neighbors(word, k=self.SIMILAR_SIZE)  
            similar_words = [w for _, w in similar_words]  
            return similar_words[-self.SIMILAR_INDEX - 1]  
        else: 
            return None

    # Word2Vec
    def load_word2vec_model(self, word2vec_path):
        self.models["word2vec"] = Word2Vec.load(word2vec_path)
    
    def word2vec_word_replacement(self, word):
        word2vec_vectors = self.models["word2vec"]
        if word in word2vec_vectors.wv:
            similar_words = word2vec_vectors.wv.most_similar(word, topn=self.SIMILAR_SIZE)
            return similar_words[-self.SIMILAR_INDEX - 1][0]  # Negative indexing
        else:
            return None
  
    # ELMO
    def load_elmo_model(self, elmo_path):
        self.models["elmo"] = hub.load(elmo_path)
        
    def build_elmo_doc_embeddings(self, doc):
        vocabulary = text_organizer.get_only_chars(doc)
        vocabulary = vocabulary.split()
        tokens = [word.lower() for word in vocabulary if word != '']  # Filter out empty strings
        elmo_model = self.models["elmo"]
        
        # tried on Jupyter Notebook, Ubuntu 22.04.4 LTS, NVIDIA A100 
        embeddings = (elmo_model.signatures['default'](tf.constant(tokens))["elmo"]).numpy()
        self.elmo_doc_tokens = tokens
        self.elmo_doc_embeddings = {token: embeddings[i] for i, token in enumerate(tokens)}
    
    def elmo_word_replacement(self, word) :
        if word not in self.elmo_doc_embeddings:
            return None 
        
        word_embedding = self.elmo_doc_embeddings[word][0]
        # Compute cosine similarities for the word with all other words in the document
        similar_words = {}
        for other_word in self.elmo_doc_tokens:
            if other_word != word:  # Do not compare with itself
                other_word_embedding = self.elmo_doc_embeddings[other_word][0]
                similarity = self.cosine_similarity(word_embedding, other_word_embedding)
                similar_words[other_word] = similarity

        sorted_similarities = sorted(similar_words.items(), key=lambda item: item[1], reverse=False)
        dynamic_index = self.get_dynamic_index(sorted_similarities)
        if 0 <= dynamic_index < len(sorted_similarities):
            return sorted_similarities[dynamic_index][0]
        else:
            return None
    
    # TM-AE
    def load_tmae_model(self, path):
        self.vectorizer_X = tools.read_pickle_data(path)
        self.knowledge_directory = dicrectories.knowledge
        
    def tmae_word_replacement(self, word):
        id = self.vectorizer_X.vocabulary_.get(word, None)
        if id is None:
            return None

        file_path = dicrectories.pickle_by_id(self.knowledge_directory, id)
        clauses = tools.read_pickle_data(file_path)
        if not clauses:
            return None
    
        clauses_sorted = sorted((clause for clause in clauses if clause[0] > self.SIMILAR_INDEX), key=lambda x: x[0], reverse=False)
        selected_features = set()
        for clause in clauses_sorted:
            weight = clause[0]
            if weight > self.SIMILAR_INDEX:
                for feature_id in clause[1]:
                    selected_features.add(self.vectorizer_X.get_feature_names_out()[feature_id])
                if len(selected_features) > 0:
                    top_features_list = list(selected_features) # Convert set to list
                    # print("knowledge word:",top_features_list[0])        
                    return top_features_list[0]
        return None
    
    # BERT
    def load_bert_model(self, path):
        self.bert_tokenizer = BertTokenizer.from_pretrained(path)
        self.bert_model = BertModel.from_pretrained(path)
        self.bert_model.eval()
    
    def build_bert_doc_embeddings(self, doc):
        vocabulary = text_organizer.get_only_chars(doc)
        vocabulary = vocabulary.split()
        vocabulary = [word.lower() for word in vocabulary if word != '']  # Filter out empty strings
        
        self.bert_doc_vocabulary = vocabulary
        inputs = self.bert_tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
        token_ids = inputs["input_ids"][0]
        self.bert_doc_tokens = self.bert_tokenizer.convert_ids_to_tokens(token_ids)
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        self.bert_doc_embeddings = outputs.last_hidden_state[0]  # Shape: (sequence_length, hidden_size)

    def bert_word_replacement(self, word):
        try:
            # Identify the target word's position
            target_idx = self.bert_doc_tokens.index(word)
        except ValueError:
            return None
        
        # Get the target embedding
        target_embedding = self.bert_doc_embeddings[target_idx].numpy()

        # Precompute embeddings for all words in the vocabulary if not already done
        if not hasattr(self, "bert_vocab_embeddings"):
            word_inputs = self.bert_tokenizer(self.bert_doc_vocabulary, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                word_embeddings = self.bert_model(**word_inputs).last_hidden_state.mean(dim=1)
            self.bert_vocab_embeddings = word_embeddings.numpy()

        # Compute cosine similarities in bulk
        vocab_embeddings = self.bert_vocab_embeddings
        target_embedding = target_embedding / np.linalg.norm(target_embedding)  # Normalize
        vocab_embeddings = vocab_embeddings / np.linalg.norm(vocab_embeddings, axis=1, keepdims=True)  # Normalize
        similarities = np.dot(vocab_embeddings, target_embedding)

        # Create a mapping of words to their similarity scores
        similar_words = {word: sim for word, sim in zip(self.bert_doc_vocabulary, similarities)}

        # Exclude the target word itself
        similar_words.pop(word, None)

        # Sort by similarity
        sorted_similarities = sorted(similar_words.items(), key=lambda item: item[1], reverse=True)

        # Get dynamic index and retrieve the word
        dynamic_index = self.get_dynamic_index(sorted_similarities)
        if 0 <= dynamic_index < len(sorted_similarities):
            return sorted_similarities[dynamic_index][0]
        else:
            return None

    
    # main function
    def word_replacement(self, word):
        try:
            if self.model_name == "glove":
                return self.glove_word_replacement(word)
            elif self.model_name == "fasttext":
                return self.fasttext_word_replacement(word)
            elif self.model_name == "word2vec":
                return self.word2vec_word_replacement(word)
            elif self.model_name == "tmae":
                return self.tmae_word_replacement(word)
            elif self.model_name == "elmo":
                return self.elmo_word_replacement(word)
            elif self.model_name == "bert":
                return self.bert_word_replacement(word)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
        except KeyError:
            return word

    def document_perturb(self, doc, percentage):
        if (self.model_name == "elmo"):
            self.build_elmo_doc_embeddings(doc)
        elif (self.model_name == "bert"):
            self.build_bert_doc_embeddings(doc)
        
        # print("=============================================================================")
        # print(doc)

        perturbed_text = '' 
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
                synonym = self.word_replacement(random_word)
                # print(random_word," ===== will be =====",synonym)
                if synonym:
                    # Replace only the matching words, preserving the original format
                    new_sentence = [
                        synonym if word.lower() == random_word else word
                        for word in new_sentence
                    ]
                    num_replaced += 1
                if num_replaced >= num_to_replace:  # Stop after replacing the target number of words
                    break

            perturbed_text = perturbed_text + ' '.join(new_sentence) + '. '

        # print("=============================================================================")
        # print(perturbed_text)
        return perturbed_text
    
    