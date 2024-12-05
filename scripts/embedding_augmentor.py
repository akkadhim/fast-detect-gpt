import numpy as np
import fasttext
from gensim.models import Word2Vec
import pickle
from directories import dicrectories
from tools import tools
import text_organizer
import random

class EmbeddingAugmentor:
    MODELS = ['glove', 'fasttext', 'word2vec', 'tmae']
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.models = {
            "glove": None,
            "fasttext": None,
            "word2vec": None,
            "tmae": None
        }
        if model_name == "glove":
            self.load_glove_embeddings('IMDB/vectors.txt')
        elif model_name == "fasttext":
            self.load_fasttext_model('IMDB/fasttext_model.bin')
        elif model_name == "word2vec":
            self.load_word2vec_model('IMDB/custom_word2vec.model')
        elif model_name == "tmae":
            self.vectorizer_X = tools.read_pickle_data("vectorizer_X.pickle")
            self.knowledge_directory = dicrectories.knowledge
        
    def load_glove_embeddings(self, glove_file_path):
        embeddings_index = {}
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        self.models["glove"] = embeddings_index

    def load_fasttext_model(self, fasttext_path):
        self.models["fasttext"] = fasttext.load_model(fasttext_path)

    def load_word2vec_model(self, word2vec_path):
        self.models["word2vec"] = Word2Vec.load(word2vec_path)
        
    def tmae_knowledge_replacement(self, word):
        # print("original word:",word)

        id = self.vectorizer_X.vocabulary_.get(word, None)
        if id is None:
            return None

        file_path = dicrectories.pickle_by_id(self.knowledge_directory, id)
        clauses = tools.read_pickle_data(file_path)
        if not clauses:
            return None
    
        min_weight = 5
        clauses_sorted = sorted((clause for clause in clauses if clause[0] > min_weight), key=lambda x: x[0], reverse=False)
        selected_features = set()
        for clause in clauses_sorted:
            weight = clause[0]
            if weight > min_weight:
                for feature_id in clause[1]:
                    selected_features.add(self.vectorizer_X.get_feature_names_out()[feature_id])
                if len(selected_features) > 0:
                    top_features_list = list(selected_features) # Convert set to list
                    # print("knowledge word:",top_features_list[0])        
                    return top_features_list[0]
        return None
    
    def knowledge_replacement_embeddings(self, word):
        model = self.models.get(self.model_name)
        
        if self.model_name == "glove":
            if word in model:
                similar_words = [(w, np.dot(model[word], model[w])) for w in model.keys()]
                similar_words.sort(key=lambda x: -x[1])
                return similar_words[0][0] if similar_words else word
        elif self.model_name == "fasttext":
            try:
                similar_words = model.get_nearest_neighbors(word)
                return similar_words[0][1] if similar_words else word
            except KeyError:
                return word
        elif self.model_name == "word2vec":
            if word in model.wv:
                similar_words = model.wv.most_similar(word)
                return similar_words[0][0] if similar_words else word
        elif self.model_name == "tmae":
            try:
                return self.tmae_knowledge_replacement(word)
            except KeyError:
                return word
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return word

    def do(self, sentence, percentage):
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
            synonym = self.knowledge_replacement_embeddings(random_word)
            if synonym:
                # Replace only the matching words, preserving the original format
                new_sentence = [
                    synonym if word.lower() == random_word else word
                    for word in new_sentence
                ]
                num_replaced += 1
            if num_replaced >= num_to_replace:  # Stop after replacing the target number of words
                break

        sentence = ' '.join(new_sentence)
        return sentence