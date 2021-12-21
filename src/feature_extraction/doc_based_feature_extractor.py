import os
import numpy as np
import spacy
import string
import logging
import textstat
from scipy.stats import entropy
from process import SentenceTokenizer
from feature_extraction.chunk import Chunk
from sentence_transformers import SentenceTransformer
from utils import load_list_of_lines, save_list_of_lines, unidecode_custom


class DocBasedFeatureExtractor(object):
    def __init__(self, lang, doc_path, sentences_per_chunk=500):
        self.lang = lang
        self.sentences_per_chunk = sentences_per_chunk
        self.doc_path = doc_path

        if self.lang == "eng":
            self.model_name = 'en_core_web_sm'
        elif self.lang == "ger":
            self.model_name = 'de_core_news_sm'
        else:
            raise Exception(f"Not a valid language {self.lang}")

        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logging.info(f"Downloading {self.model_name} for Spacy...")
            os.system(f"python3 -m spacy download {self.model_name}")
            logging.info(f"Downloaded {self.model_name} for Spacy.")
            self.nlp = spacy.load(self.model_name)

        self.stopwords = self.nlp.Defaults.stop_words
        new_stopwords = []
        for stopword in self.stopwords:
            new_stopwords.append(unidecode_custom(stopword))
        self.stopwords = set(new_stopwords)

        ## load sentences
        sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
        if os.path.exists(sentences_path):
            self.sentences = load_list_of_lines(sentences_path, "str")
        else:
            self.sentence_tokenizer = SentenceTokenizer(self.lang)
            self.sentences = self.sentence_tokenizer.tokenize(doc_path)
            save_list_of_lines(self.sentences, sentences_path, "str")

        ## load sbert sentence embeddings
        sbert_sentence_embeddings_path = doc_path.replace("/raw_docs", f"/processed_sbert_sentence_embeddings") + ".npz"
        if os.path.exists(sbert_sentence_embeddings_path):
            self.sbert_sentence_embeddings = load_list_of_lines(sbert_sentence_embeddings_path, "np")
        else:
            if self.lang == "eng":
                self.sentence_encoder = SentenceTransformer('stsb-mpnet-base-v2')
            elif self.lang == "ger":
                self.sentence_encoder = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
            self.sbert_sentence_embeddings = list(self.sentence_encoder.encode(self.sentences))
            save_list_of_lines(self.sbert_sentence_embeddings, sbert_sentence_embeddings_path, "np")

        ## load doc2vec chunk embeddings
        doc2vec_chunk_embeddings_path = doc_path.replace("/raw_docs", f"/processed_doc2vec_chunk_embeddings_spc_{sentences_per_chunk}") + ".npz"
        if os.path.exists(doc2vec_chunk_embeddings_path):
            self.doc2vec_chunk_embeddings = load_list_of_lines(doc2vec_chunk_embeddings_path, "np")
        else:
            raise Exception(f"Could not find Doc2Vec chunk embeddings for chunk size {self.sentences_per_chunk}.")

        self.chunks = self.__get_chunks()

    def __get_chunks(self):
        if self.sentences_per_chunk is None:
            return [Chunk(self.sentences, self.sbert_sentence_embeddings, self.doc2vec_chunk_embeddings)]
        chunks = []
        chunk_id_counter = 0
        for i in range(0, len(self.sentences), self.sentences_per_chunk):
            current_sentences = self.sentences[i:i+self.sentences_per_chunk]
            current_sentence_embeddings = self.sbert_sentence_embeddings[i:i+self.sentences_per_chunk]
            if (len(current_sentences) == self.sentences_per_chunk) or (i == 0):
                chunks.append(Chunk(current_sentences, current_sentence_embeddings, self.doc2vec_chunk_embeddings[chunk_id_counter]))
                chunk_id_counter += 1
        return chunks


    def get_all_features(self):
        chunk_based_feature_mapping = {
            "ratio_of_punctuation_marks": self.get_ratio_of_punctuation_marks,
            "ratio_of_whitespaces": self.get_ratio_of_whitespaces,
            "ratio_of_digits": self.get_ratio_of_digits,
            "ratio_of_exclamation_marks": self.get_ratio_of_exclamation_marks,
            "ratio_of_question_marks": self.get_ratio_of_question_marks,
            "ratio_of_commas": self.get_ratio_of_commas,
            "ratio_of_uppercase_letters": self.get_ratio_of_uppercase_letters,
            "average_number_of_words_in_sentence": self.get_average_number_of_words_in_sentence,
            "maximum_number_of_words_in_sentence": self.get_maximum_number_of_words_in_sentence,
            "ratio_of_unique_word_unigrams": self.get_ratio_of_unique_word_unigrams,
            "ratio_of_unique_word_bigrams": self.get_ratio_of_unique_word_bigrams,
            "ratio_of_unique_word_trigrams": self.get_ratio_of_unique_word_trigrams,
            "text_length": self.get_text_length,
            "average_word_length": self.get_average_word_length,
            "ratio_of_stopwords": self.get_ratio_of_stopwords,
            "bigram_entropy": self.get_word_bigram_entropy,
            "trigram_entropy": self.get_word_trigram_entropy,
            "type_token_ratio": self.get_type_token_ratio,

            ## Features in the list
            "flesch_reading_ease_score": self.get_flesch_reading_ease_score, # readability
            "unigram_entropy": self.get_word_unigram_entropy, # second order redundancy
            "average_paragraph_length": self.get_average_paragraph_length, # structural features
            0: self.get_average_sbert_sentence_embedding,
            1: self.get_doc2vec_chunk_embedding,
            # skipped greetings since this is not e-mail(structural features)
            # skipped types of signature since this is not e-mail(structural features)
            # skipped content specific features. added BERT average sentence embedding instead.

            #######
        }

        book_based_feature_mapping = {
            "doc2vec_intra_textual_variance": self.get_doc2vec_intra_textual_variance,
            "sbert_intra_textual_variance": self.get_sbert_intra_textual_variance,
            "doc2vec_stepwise_distance": self.get_doc2vec_stepwise_distance,
            "sbert_stepwise_distance": self.get_sbert_stepwise_distance
        }

        # extract chunk based features
        chunk_based_features = []
        for chunk in self.chunks:
            current_features = {"book_name": self.doc_path.split("/")[-1][:-4]}
            for feature_name, feature_function in chunk_based_feature_mapping.items():
                if isinstance(feature_name, int):
                    current_features.update(feature_function(chunk))
                else:
                    current_features[feature_name] = feature_function(chunk)
            chunk_based_features.append(current_features)

        # extract book based features
        book_based_features = {}
        for feature_name, feature_function in book_based_feature_mapping.items():
            book_based_features["book_name"] = self.doc_path.split("/")[-1][:-4]
            book_based_features[feature_name] = feature_function(self.chunks)

        return chunk_based_features, \
               book_based_features, \
               [np.array(chunk.sbert_sentence_embeddings).mean(axis=0) for chunk in self.chunks], \
               [chunk.doc2vec_chunk_embedding for chunk in self.chunks]

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
        splitted_lengths = [len(splitted) for splitted in chunk.raw_text.split("\n")]
        return np.mean(splitted_lengths)

    def get_average_sbert_sentence_embedding(self, chunk):
        average_sentence_embedding = np.array(chunk.sbert_sentence_embeddings).mean(axis=0)
        average_sentence_embedding_features = dict((f"average_sentence_embedding_{index+1}", embedding_part) for index, embedding_part in enumerate(average_sentence_embedding))
        return average_sentence_embedding_features

    def get_doc2vec_chunk_embedding(self, chunk):
        doc2vec_chunk_embedding_features = dict((f"doc2vec_chunk_embedding_{index+1}", embedding_part) for index, embedding_part in enumerate(chunk.doc2vec_chunk_embedding))
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

    def get_ratio_of_unique_word_unigrams(self, chunk):
        return len(chunk.word_unigram_counts.keys()) / sum(chunk.word_unigram_counts.values())

    def get_ratio_of_unique_word_bigrams(self, chunk):
        return len(chunk.word_bigram_counts.keys()) / sum(chunk.word_bigram_counts.values())

    def get_ratio_of_unique_word_trigrams(self, chunk):
        return len(chunk.word_trigram_counts.keys()) / sum(chunk.word_trigram_counts.values())

    def get_text_length(self, chunk):
        return len(chunk.unidecoded_raw_text)

    def get_average_word_length(self, chunk):
        word_lengths = []
        for word, count in chunk.word_unigram_counts.items():
            word_lengths.append(len(word) * count)
        return np.mean(word_lengths)

    def get_ratio_of_stopwords(self, chunk):
        number_of_stopwords = 0
        number_of_all_words = 0
        for word, count in chunk.word_unigram_counts.items():
            number_of_all_words += 1
            if word in self.stopwords:
                number_of_stopwords += count
        return number_of_stopwords / number_of_all_words

    def get_word_unigram_entropy(self, chunk):
        return entropy(list(chunk.word_unigram_counts.values()))

    def get_word_bigram_entropy(self, chunk):
        temp_dict = {}
        for word_bigram, count in chunk.word_bigram_counts.items():
            left = word_bigram[0]
            if left in temp_dict.keys():
                temp_dict[left].append(count)
            else:
                temp_dict[left] = [count]
        entropies = []
        for left, counts in temp_dict.items():
            entropies.append(entropy(counts))
        return np.mean(entropies)

    def get_word_trigram_entropy(self, chunk):
        temp_dict = {}
        for word_trigram, count in chunk.word_trigram_counts.items():
            left_and_middle = (word_trigram[0], word_trigram[1])
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
        tokens = sum(chunk.word_unigram_counts.values())
        types = len(chunk.word_unigram_counts)
        return types/tokens

    def get_flesch_reading_ease_score(self, chunk):
        return textstat.flesch_reading_ease(chunk.unidecoded_raw_text)

    def get_gunning_fog(self, chunk):
        """
        Not implemented for German. If we can find "easy words" in German, then we can implement it ourselves.
        """
        return textstat.gunning_fog(chunk.unidecoded_raw_text)

    ######
    ## book based features
    ######
    def __get_intra_textual_variance(self, chunks, embedding_type):
        chunk_embeddings = []
        for chunk in chunks:
            if embedding_type == "doc2vec":
                chunk_embeddings.append(chunk.doc2vec_chunk_embedding)
            elif embedding_type == "sbert":
                chunk_embeddings.append(np.array(chunk.sbert_sentence_embeddings).mean(axis=0))
            else:
                raise Exception(f"Not a valid embedding type {embedding_type}")
        average_chunk_embedding = np.array(chunk_embeddings).mean(axis=0)
        euclidean_distances = [np.linalg.norm(average_chunk_embedding - chunk_embedding) for chunk_embedding in chunk_embeddings]
        return np.mean(euclidean_distances)

    def get_doc2vec_intra_textual_variance(self, chunks):
        return self.__get_intra_textual_variance(chunks, "doc2vec")

    def get_sbert_intra_textual_variance(self, chunks):
        return self.__get_intra_textual_variance(chunks, "sbert")

    def __get_stepwise_distance(self, chunks, embedding_type):
        euclidean_distances = []
        for chunk_idx in range(1, len(chunks)):
            if embedding_type == "doc2vec":
                current_chunk_embedding = chunks[chunk_idx].doc2vec_chunk_embedding
                previous_chunk_embedding = chunks[chunk_idx - 1].doc2vec_chunk_embedding
            elif embedding_type == "sbert":
                current_chunk_embedding = np.array(chunks[chunk_idx].sbert_sentence_embeddings).mean(axis=0)
                previous_chunk_embedding = np.array(chunks[chunk_idx - 1].sbert_sentence_embeddings).mean(axis=0)
            else:
                raise Exception(f"Not a valid embedding type {embedding_type}")
            euclidean_distances.append(np.linalg.norm(current_chunk_embedding - previous_chunk_embedding))
        return np.mean(euclidean_distances)

    def get_doc2vec_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, "doc2vec")

    def get_sbert_stepwise_distance(self, chunks):
        return self.__get_stepwise_distance(chunks, "sbert")
