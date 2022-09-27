import os
import numpy as np
import spacy
import string
import logging
import textstat
from pathlib import Path
from scipy.stats import entropy
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
from .process import SentenceTokenizer
from .chunk import Chunk
from utils import load_list_of_lines, save_list_of_lines, get_bookname


class DocBasedFeatureExtractor():
    '''Get features that can be calculated from a single doc.'''
    def __init__(self, 
        language, 
        doc_path, 
        sentences_per_chunk=200,
        processed_sentences=True, 
        unigram_counts=True, 
        bigram_counts=True, 
        trigram_counts=True, 
        raw_text=True, 
        unidecoded_raw_text=True, 
        char_unigram_counts=True):

        self.language = language
        self.sentences_per_chunk = sentences_per_chunk
        self.doc_path = doc_path

        # Parameters for creating chunks
        self.processed_sentences = processed_sentences
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
        self.raw_text = raw_text
        self.unidecoded_raw_text = unidecoded_raw_text
        self.char_unigram_counts = char_unigram_counts

        # Spacy
        if self.language == 'eng':
            self.model_name = 'en_core_web_sm'
        elif self.language == 'ger':
            self.model_name = 'de_core_news_sm'
        else:
            raise Exception(f'Not a valid language {self.language}')
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logging.info(f'Downloading {self.model_name} for Spacy...')
            os.system(f'python3 -m spacy download {self.model_name}')
            logging.info(f'Downloaded {self.model_name} for Spacy.')
            self.nlp = spacy.load(self.model_name)

        # Preprocess or load data
        tokenized_sentences_path = doc_path.replace('/raw_docs', f'/tokenized_sentences')
        if os.path.exists(tokenized_sentences_path):
            self.tokenized_sentences = load_list_of_lines(tokenized_sentences_path, 'str')
        else:
            self.sentence_tokenizer = SentenceTokenizer(self.language)
            self.tokenized_sentences = self.sentence_tokenizer.tokenize(doc_path)
            save_list_of_lines(self.tokenized_sentences, tokenized_sentences_path, 'str')

        sbert_sentence_embeddings_path = doc_path.replace('/raw_docs', f'/sbert_sentence_embeddings') + '.npz'
        if os.path.exists(sbert_sentence_embeddings_path):
            self.sbert_sentence_embeddings = load_list_of_lines(sbert_sentence_embeddings_path, 'np')
        else:
            if self.language == 'eng':
                self.sentence_encoder = SentenceTransformer('stsb-mpnet-base-v2')
            elif self.language == 'ger':
                self.sentence_encoder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
            self.sbert_sentence_embeddings = list(self.sentence_encoder.encode(self.tokenized_sentences))
            save_list_of_lines(self.sbert_sentence_embeddings, sbert_sentence_embeddings_path, 'np')

        doc2vec_chunk_embeddings_path = doc_path.replace('/raw_docs', f'/doc2vec_chunk_embeddings_spc_{self.sentences_per_chunk}') + '.npz'
        if os.path.exists(doc2vec_chunk_embeddings_path):
            self.doc2vec_chunk_embeddings = load_list_of_lines(doc2vec_chunk_embeddings_path, 'np')
        else:
            raise Exception(f'Could not find Doc2Vec chunk embeddings for chunk size {self.sentences_per_chunk}.')

        self.chunks = self.__get_chunks()

    def __get_chunks(self):
        file_name = get_bookname(self.doc_path)
        if self.sentences_per_chunk is None:
            return [Chunk(
                sentences_per_chunk = self.sentences_per_chunk,
                doc_path = self.doc_path,
                chunk_id = 'None',
                tokenized_sentences = self.tokenized_sentences,
                sbert_sentence_embeddings = self.sbert_sentence_embeddings,
                doc2vec_chunk_embedding = self.doc2vec_chunk_embeddings[0],
                processed_sentences = self.processed_sentences,
                unigram_counts = self.unigram_counts,
                bigram_counts = self.bigram_counts,
                trigram_counts = self.trigram_counts,
                raw_text = self.raw_text,
                unidecoded_raw_text = self.unidecoded_raw_text,
                char_unigram_counts = self.char_unigram_counts)]

        else:
            chunks = []
            chunk_id_counter = 0
            for i in range(0, len(self.tokenized_sentences), self.sentences_per_chunk):
                current_sentences = self.tokenized_sentences[i:i+self.sentences_per_chunk]
                current_sentence_embeddings = self.sbert_sentence_embeddings[i:i+self.sentences_per_chunk]
                if (len(current_sentences) == self.sentences_per_chunk) or (i == 0):
                    chunks.append(Chunk(
                        sentences_per_chunk = self.sentences_per_chunk,
                        doc_path = self.doc_path,
                        chunk_id = chunk_id_counter,
                        tokenized_sentences = current_sentences,
                        sbert_sentence_embeddings = current_sentence_embeddings,
                        doc2vec_chunk_embedding = self.doc2vec_chunk_embeddings[chunk_id_counter],
                        processed_sentences = self.processed_sentences,
                        unigram_counts = self.unigram_counts,
                        bigram_counts = self.bigram_counts,
                        trigram_counts = self.trigram_counts,
                        raw_text = self.raw_text,
                        unidecoded_raw_text = self.unidecoded_raw_text,
                        char_unigram_counts = self.char_unigram_counts))
                    chunk_id_counter += 1
            return chunks


    def get_all_features(self):
        chunk_feature_mapping = {
            'ratio_of_punctuation_marks': self.get_ratio_of_punctuation_marks,
            'ratio_of_whitespaces': self.get_ratio_of_whitespaces,
            #'ratio_of_digits': self.get_ratio_of_digits,
            'ratio_of_exclamation_marks': self.get_ratio_of_exclamation_marks,
            'ratio_of_question_marks': self.get_ratio_of_question_marks,
            'ratio_of_commas': self.get_ratio_of_commas,
            'ratio_of_uppercase_letters': self.get_ratio_of_uppercase_letters,
            'average_number_of_words_in_sentence': self.get_average_number_of_words_in_sentence,
            'maximum_number_of_words_in_sentence': self.get_maximum_number_of_words_in_sentence,
            'ratio_of_unique_unigrams': self.get_ratio_of_unique_unigrams,
            'ratio_of_unique_bigrams': self.get_ratio_of_unique_bigrams,
            'ratio_of_unique_trigrams': self.get_ratio_of_unique_trigrams,
            'text_length': self.get_text_length,
            'average_word_length': self.get_average_word_length,
            'bigram_entropy': self.get_bigram_entropy,
            'trigram_entropy': self.get_trigram_entropy,
            'type_token_ratio': self.get_type_token_ratio,
            'flesch_reading_ease_score': self.get_flesch_reading_ease_score,
            'unigram_entropy': self.get_unigram_entropy, # second order redundancy
            'average_paragraph_length': self.get_average_paragraph_length, # structural features
            0: self.get_average_sbert_sentence_embedding, 
            1: self.get_doc2vec_chunk_embedding
        }

        book_feature_mapping = {
            'doc2vec_intra_textual_variance': self.get_doc2vec_intra_textual_variance,
            'sbert_intra_textual_variance': self.get_sbert_intra_textual_variance,
            'doc2vec_stepwise_distance': self.get_doc2vec_stepwise_distance,
            'sbert_stepwise_distance': self.get_sbert_stepwise_distance
        }

        # extract chunk based features
        chunk_features = []
        for chunk in self.chunks:
            if self.sentences_per_chunk is not None:
                chunk_name = chunk.file_name + '_' + str(chunk.chunk_id)
            else:
                chunk_name = chunk.file_name
            current_features = {'file_name': chunk_name}
            for feature_name, feature_function in chunk_feature_mapping.items():
                if isinstance(feature_name, int):
                    current_features.update(feature_function(chunk))
                else:
                    current_features[feature_name] = feature_function(chunk)
            chunk_features.append(current_features)

        # extract book based features
        book_features = None
        if self.sentences_per_chunk is not None:
            book_features = {}
            for feature_name, feature_function in book_feature_mapping.items():
                book_features['file_name'] = self.doc_path.split('/')[-1][:-4]
                book_features[feature_name] = feature_function(self.chunks)

        #Return sbert embeddings by averageing across sentences belonging to a chunk 
        return chunk_features, \
                book_features


    def get_ratio_of_punctuation_marks(self, chunk):
        punctuations = 0
        for character in string.punctuation:
            punctuations += chunk.char_unigram_counts.get(character, 0)
        all_characters = sum(list(chunk.char_unigram_counts.values()))
        return punctuations / all_characters

    def get_ratio_of_digits(self, chunk):
        digits = 0
        all_characters = 0
        for character in [str(i) for i in range(10)]:
            digits += chunk.char_unigram_counts.get(character, 0)
        all_characters = sum(list(chunk.char_unigram_counts.values()))
        return digits / all_characters

    def get_ratio_of_whitespaces(self, chunk):
        return chunk.char_unigram_counts.get(' ', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_exclamation_marks(self, chunk):
        return chunk.char_unigram_counts.get('!', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_question_marks(self, chunk):
        return chunk.char_unigram_counts.get('?', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_commas(self, chunk):
        return chunk.char_unigram_counts.get(',', 0) / sum(list(chunk.char_unigram_counts.values()))

    def get_ratio_of_uppercase_letters(self, chunk):
        num_upper = 0
        num_alpha = 0
        for char in chunk.unidecoded_raw_text:
            if char.isalpha():
                num_alpha += 1
                if char.isupper():
                    num_upper += 1
        return num_upper / num_alpha

    def get_average_paragraph_length(self, chunk):
        split_lengths = [len(curr_split) for curr_split in chunk.raw_text.split('\n')]
        return np.mean(split_lengths)

    def get_average_sbert_sentence_embedding(self, chunk):
        average_sentence_embedding = np.array(chunk.sbert_sentence_embeddings).mean(axis=0)
        average_sentence_embedding_features = dict((f'average_sentence_embedding_{index+1}', embedding_part) for index, embedding_part in enumerate(average_sentence_embedding))
        return average_sentence_embedding_features

    def get_doc2vec_chunk_embedding(self, chunk):
        doc2vec_chunk_embedding_features = dict((f'doc2vec_chunk_embedding_{index+1}', embedding_part) for index, embedding_part in enumerate(chunk.doc2vec_chunk_embedding))
        return doc2vec_chunk_embedding_features

    def get_average_number_of_words_in_sentence(self, chunk):
        sentence_lengths = []
        for processed_sentence in chunk.processed_sentences:
            sentence_lengths.append(len(processed_sentence.split()))
        return np.mean(sentence_lengths)

    def get_maximum_number_of_words_in_sentence(self, chunk):
        sentence_lengths = []
        for processed_sentence in chunk.processed_sentences:
            sentence_lengths.append(len(processed_sentence.split()))
        return np.max(sentence_lengths)

    def get_ratio_of_unique_unigrams(self, chunk):
        return len(chunk.unigram_counts.keys()) / sum(chunk.unigram_counts.values())

    def get_ratio_of_unique_bigrams(self, chunk):
        return len(chunk.bigram_counts.keys()) / sum(chunk.bigram_counts.values())

    def get_ratio_of_unique_trigrams(self, chunk):
        return len(chunk.trigram_counts.keys()) / sum(chunk.trigram_counts.values())

    def get_text_length(self, chunk):
        return len(chunk.unidecoded_raw_text)

    def get_average_word_length(self, chunk):
        word_lengths = []
        for word, count in chunk.unigram_counts.items():
            word_lengths.append(len(word) * count)
        return np.mean(word_lengths)

    def get_unigram_entropy(self, chunk):
        return entropy(list(chunk.unigram_counts.values()))

    def get_bigram_entropy(self, chunk):
        temp_dict = {}
        for bigram, count in chunk.bigram_counts.items():
            bigram = bigram.split()
            left = bigram[0]
            if left in temp_dict.keys():
                temp_dict[left].append(count)
            else:
                temp_dict[left] = [count]
        entropies = []
        for left, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)

    def get_trigram_entropy(self, chunk):
        temp_dict = {}
        for trigram, count in chunk.trigram_counts.items():
            trigram = trigram.split()
            left_and_middle = trigram[0] + ' ' + trigram[1]
            if left_and_middle in temp_dict.keys():
                temp_dict[left_and_middle].append(count)
            else:
                temp_dict[left_and_middle] = [count]
        entropies = []
        for left_and_middle, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)

    def get_type_token_ratio(self, chunk):
        # Type-token ratio according to Algee-Hewitt et al. (2016)
        tokens = sum(chunk.unigram_counts.values())
        types = len(chunk.unigram_counts)
        return types/tokens

    def get_flesch_reading_ease_score(self, chunk):
        return textstat.flesch_reading_ease(chunk.unidecoded_raw_text)

    def get_gunning_fog(self, chunk):
        '''''''''
        Not implemented for German. If we can find 'easy words' in German, then we can implement it ourselves.
        '''
        return textstat.gunning_fog(chunk.unidecoded_raw_text)

    # book-based features
    def __get_intra_textual_variance(self, chunks, embedding_type):
        chunk_embeddings = []
        for chunk in chunks:
            if embedding_type == 'doc2vec':
                chunk_embeddings.append(chunk.doc2vec_chunk_embedding)
            elif embedding_type == 'sbert':
                chunk_embeddings.append(np.array(chunk.sbert_sentence_embeddings).mean(axis=0)) 
            else:
                raise Exception(f'Not a valid embedding type {embedding_type}')
        average_chunk_embedding = np.array(chunk_embeddings).mean(axis=0)
        euclidean_distances = [np.linalg.norm(average_chunk_embedding - chunk_embedding) for chunk_embedding in chunk_embeddings]
        return np.mean(euclidean_distances)

    def get_doc2vec_intra_textual_variance(self, chunks):
        return self.__get_intra_textual_variance(chunks, 'doc2vec')

    def get_sbert_intra_textual_variance(self, chunks):
        return self.__get_intra_textual_variance(chunks, 'sbert')

    def __get_stepwise_distance(self, chunks, embedding_type):
        if len(chunks) == 1:
            return 0
        euclidean_distances = []
        for chunk_idx in range(1, len(chunks)):
            #print('index', chunk_idx)
            if embedding_type == 'doc2vec':
                current_chunk_embedding = chunks[chunk_idx].doc2vec_chunk_embedding
                previous_chunk_embedding = chunks[chunk_idx - 1].doc2vec_chunk_embedding
            elif embedding_type == 'sbert':
                current_chunk_embedding = np.array(chunks[chunk_idx].sbert_sentence_embeddings).mean(axis=0)
                previous_chunk_embedding = np.array(chunks[chunk_idx - 1].sbert_sentence_embeddings).mean(axis=0)
            else:
                raise Exception(f'Not a valid embedding type {embedding_type}')
            #print('Norm:\n', np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
            euclidean_distances.append(np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
        #print('Mean\n: ', np.mean(euclidean_distances))
        return np.mean(euclidean_distances)

    def get_doc2vec_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, 'doc2vec')

    def get_sbert_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, 'sbert')
