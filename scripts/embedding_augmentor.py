import numpy as np
import fasttext
from gensim.models import Word2Vec
import pickle
from directories import dicrectories
from tools import tools

class EmbeddingAugmentor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.models = {
            "glove": None,
            "fasttext": None,
            "word2vec": None,
            "tm-ae": None
        }
        if model_name == "glove":
            self.load_glove_embeddings('IMDB/vectors.txt')
        elif model_name == "fasttext":
            self.load_fasttext_model('IMDB/fasttext_model.bin')
        elif model_name == "word2vec":
            self.load_word2vec_model('IMDB/custom_word2vec.model')
        elif model_name == "tm-ae":
            def preprocess_text(text):
                return text
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
            return word

        id = self.vectorizer_X.vocabulary_[word]
        file_path = dicrectories.pickle_by_id(self.knowledge_directory, id)
        clauses = tools.read_pickle_data(file_path)
        clauses_sorted = sorted(clauses, key=lambda x: x[0], reverse=False)
        
        # Collect top features from high-weight clauses
        top_weight = 5
        top_features = set()
        
        for clause in clauses_sorted:
            weight = clause[0]
            if weight > top_weight:
                for feature_id in clause[1]:
                    top_features.add(self.vectorizer_X.get_feature_names_out()[feature_id])
        
        if len(top_features) > 0:
            top_features_list = list(top_features) # Convert set to list
            # print("knowledge word:",top_features_list[0])        
            return top_features_list[0]
        else:
            return word
    
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
        elif self.model_name == "tm-ae":
            try:
                return self.tmae_knowledge_replacement(word)
            except KeyError:
                return word
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return word
