import text_organizer
import numpy as np
import fasttext
from gensim.models import Word2Vec
from directories import dicrectories
from tools import tools
import random
import tensorflow as tf
import tensorflow_hub as hub
import torch
from transformers import BertTokenizer, BertModel
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
 
class EmbeddingAugmentor:
    MODELS = ['glove', 'fasttext', 'word2vec', 'tmae', 'elmo', 'bert']
    SIMILAR_SIZE = 400
    SIMILAR_INDEX = 5
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.models = {
            "glove": None,
            "fasttext": None,
            "word2vec": None,
            "elmo": None,
            "tmae": None
        }
        if model_name == "glove":
            self.load_glove_embeddings('IMDB/glove_vectors.txt')
        elif model_name == "fasttext":
            self.load_fasttext_model('IMDB/fasttext_model.bin')
        elif model_name == "word2vec":
            self.load_word2vec_model('IMDB/custom_word2vec.model')
        elif model_name == "elmo":
            self.load_elmo_model("https://tfhub.dev/google/elmo/3")
        elif model_name == "bert":
            self.load_bert_model("IMDB/pretrained_bert")
        elif model_name == "tmae":
            self.load_tmae_model('IMDB/tmae_vectorizer_X.pickle')
      
    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    
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
    
    def glove_knowledge_replacement(self, word):
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
        
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=False)
        
        if self.SIMILAR_INDEX < len(sorted_similarities):
            return sorted_similarities[self.SIMILAR_INDEX][0]  
        else:
            return None  

    # FastText
    def load_fasttext_model(self, fasttext_path):
        self.models["fasttext"] = fasttext.load_model(fasttext_path)
    
    def fasttext_knowledge_replacement(self, word):
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
    
    def word2vec_knowledge_replacement(self, word):
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
        
        embeddings = (elmo_model.signatures['default'](tf.constant(tokens))["elmo"]).numpy()
        self.elmo_doc_tokens = tokens
        self.elmo_doc_embeddings = embeddings
    
    def elmo_knowledge_replacement(self, word) :
        word_embedding = self.elmo_doc_embeddings[word]
        # Compute cosine similarities for the word with all other words in the document
        similar_words = {}
        for other_word in self.elmo_doc_tokens:
            if other_word != word:  # Do not compare with itself
                other_word_embedding = self.elmo_doc_embeddings[other_word]
                similarity = self.cosine_similarity(word_embedding, other_word_embedding)
                similar_words[word] = similarity

        sorted_similarities = sorted(similar_words.items(), key=lambda item: item[1], reverse=False)
        if len(sorted_similarities) > 0:
            return sorted_similarities[0][0]  
        else:
            return None 
    
    # TM-AE
    def load_tmae_model(self, path):
        self.vectorizer_X = tools.read_pickle_data(path)
        self.knowledge_directory = dicrectories.knowledge
        
    def tmae_knowledge_replacement(self, word):
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
        # self.vocab_embeddings = np.load(path + "/vocab_embeddings.npy", allow_pickle=True).item()
        self.bert_model.eval()
        
    def bert_knowledge_replacement(self, target_word):       
        # Identify the target word's position
        try:
            target_idx = self.bert_doc_tokens.index(target_word)
        except ValueError:
            return None
        
        target_embedding = self.bert_doc_embeddings[target_idx].numpy()
        # Compute similarity with all words in the vocabulary
        similar_words = {}
        for word in self.bert_doc_vocabulary:
            if target_word != word:
                word_inputs = self.bert_tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    word_embedding = self.bert_model(**word_inputs).last_hidden_state.mean(dim=1).squeeze(0).numpy()
                similarity = self.cosine_similarity(target_embedding, word_embedding)
                similar_words[word] = similarity
        
        sorted_similarities = sorted(similar_words.items(), key=lambda item: item[1], reverse=False)
        
        if len(sorted_similarities) > 0:
            return sorted_similarities[0][0]  
        else:
            return None 

    
    def knowledge_replacement_embeddings(self, word):
        try:
            if self.model_name == "glove":
                return self.glove_knowledge_replacement(word)
            elif self.model_name == "fasttext":
                return self.fasttext_knowledge_replacement(word)
            elif self.model_name == "word2vec":
                return self.word2vec_knowledge_replacement(word)
            elif self.model_name == "tmae":
                return self.tmae_knowledge_replacement(word)
            elif self.model_name == "elmo":
                return self.elmo_knowledge_replacement(word)
            elif self.model_name == "bert":
                return self.bert_knowledge_replacement(word)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
        except KeyError:
            return word

    def do(self, doc, percentage):
        if (self.model_name == "elmo"):
            self.build_elmo_doc_embeddings(doc)
            
        if (self.model_name == "bert"):
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

            aug_text = aug_text + ' '.join(new_sentence) + '. '
        return aug_text
    
    