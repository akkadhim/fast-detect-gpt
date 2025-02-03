#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 
import text_organizer
import random

def calculate_similarity(word, synonyms, model):
    """Calculate similarity between the word and its synonyms using Word2Vec."""
    similarities = {}
    for synonym in synonyms:
        if synonym in model.wv.key_to_index and word in model.wv.key_to_index:
            similarity = model.wv.similarity(word, synonym)
            similarities[synonym] = similarity
    return similarities

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

def synonum_by_word2vec(doc, embedding_model, percentage, similarity_threshold):
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
                similarities = calculate_similarity(random_word, synonyms, embedding_model)
                
                if similarities:
                    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1])
                    if similarity_threshold == 'min':
                        # Select the synonym with the lowest similarity score
                        synonym = sorted_similarities[0][0]
                    elif similarity_threshold == 'mid':
                        # Select the synonym with the mid similarity score
                        mid_index = len(sorted_similarities) // 2
                        synonym = sorted_similarities[mid_index][0]
                    elif similarity_threshold == 'high':
                        # Select the synonym with the highest similarity score
                        synonym = sorted_similarities[-1][0]
                    else:
                        raise ValueError("Invalid level. Choose from 'min', 'mid', 'high'.")
                    
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

def synonum_by_embedding(doc, embedding, percentage):
    if (embedding.model_name == "elmo"):
        embedding.build_elmo_doc_embeddings(doc)
    elif (embedding.model_name == "bert"):
        embedding.build_bert_doc_embeddings(doc)
        
    changed_count = 0
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
                    synonym = random.choice(list(synonyms))
                    if synonym:
                        embedding_word = embedding.word_replacement(synonym)
                        if embedding_word != None and embedding_word != []:
                            # Replace only the matching words, preserving the original format
                            new_sentence = [
                                embedding_word if word.lower() == random_word else word
                                for word in new_sentence
                            ]
                            num_replaced += 1
                            changed_count += 1
                    # Stop after replacing the target number of words
                    if num_replaced >= num_to_replace:
                        break

        aug_text = aug_text + ' '.join(new_sentence) + '. '
    return aug_text, changed_count


