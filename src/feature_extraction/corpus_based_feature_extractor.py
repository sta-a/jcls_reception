from code import compile_command
from hashlib import new
import os
import sys
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
from pydoc import doc
from xml.sax.handler import feature_namespace_prefixes
from collections import Counter
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, _document_frequency
import scipy
import spacy
from gensim.models import LdaMulticore
from gensim.matutils import Sparse2Corpus
from corpus_toolkit import corpus_tools as ct
from utils import load_list_of_lines, save_list_of_lines, df_from_dict, get_bookname
from .production_rule_extractor import ProductionRuleExtractor
from .doc_based_feature_extractor import DocBasedFeatureExtractor

class CorpusBasedFeatureExtractor():
    '''Get features for which the whole corpus needs to be considered.'''
    def __init__(self, language, doc_paths, sentences_per_chunk=200, nr_features=100):
        self.language = language
        self.doc_paths = doc_paths
        self.sentences_per_chunk = sentences_per_chunk
        self.nr_features = nr_features

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

        self.word_statistics = self.__get_word_statistics()

        self.all_average_sbert_sentence_embeddings = []
        self.all_doc2vec_chunk_embeddings = []
        for doc_chunks in self.__generate_chunks():
            curr_sbert = []
            curr_doc2vec = []
            for chunk in doc_chunks:
                curr_sbert.append(np.array(chunk.sbert_sentence_embeddings).mean(axis=0))
                curr_doc2vec.append(chunk.doc2vec_chunk_embedding)
            self.all_average_sbert_sentence_embeddings.append(curr_sbert)
            self.all_doc2vec_chunk_embeddings.append(curr_doc2vec)

    def __generate_chunks(self,         
            processed_sentences=False, 
            unigram_counts=False, 
            bigram_counts=False, 
            trigram_counts=False, 
            raw_text=False, 
            unidecoded_raw_text=False, 
            char_unigram_counts=False):
        for doc_path in tqdm(self.doc_paths):
            doc_chunks = DocBasedFeatureExtractor(self.language, doc_path, self.sentences_per_chunk, processed_sentences, unigram_counts, bigram_counts, trigram_counts, raw_text, 
                unidecoded_raw_text, char_unigram_counts).chunks
            yield doc_chunks


    def __get_word_statistics(self):
        total_unigram_counts = Counter()
        total_bigram_counts = Counter()
        total_trigram_counts = Counter()
        book_unigram_mapping = {}
        book_bigram_mapping = {}
        book_trigram_mapping = {}

        for doc_chunks in self.__generate_chunks(unigram_counts=True, bigram_counts=True, trigram_counts=True):
            book_unigram_counts = {}
            book_bigram_counts = {}
            book_trigram_counts = {}
            for chunk in doc_chunks:
                file_name = chunk.file_name

                for unigram, counts in chunk.unigram_counts.items():
                    if unigram in book_unigram_counts:
                        book_unigram_counts[unigram] += counts
                    else:
                        book_unigram_counts[unigram] = counts

                for bigram, counts in chunk.bigram_counts.items():
                    if bigram in book_bigram_counts:
                        book_bigram_counts[bigram] += counts
                    else:
                        book_bigram_counts[bigram] = counts

                for trigram, counts in chunk.trigram_counts.items():
                    if trigram in book_trigram_counts:
                        book_trigram_counts[trigram] += counts
                    else:
                        book_trigram_counts[trigram] = counts
            book_unigram_mapping[file_name] = book_unigram_counts
            book_bigram_mapping[file_name] = book_bigram_counts
            book_trigram_mapping[file_name] = book_trigram_counts
            total_unigram_counts.update(book_unigram_counts)
            total_bigram_counts.update(book_bigram_counts)
            total_trigram_counts.update(book_trigram_counts)

        total_unigram_counts = dict(sorted(list(total_unigram_counts.items()), key=lambda x: -x[1])) #all words
        total_bigram_counts = dict(sorted(list(total_bigram_counts.items()), key=lambda x: -x[1])[:2000]) 
        total_trigram_counts = dict(sorted(list(total_trigram_counts.items()), key=lambda x: -x[1])[:2000])
        
        # keep only counts of the 2000 most frequent bi- and trigrams
        book_bigram_mapping_ = {}
        for book, book_dict in book_bigram_mapping.items():
            book_dict_ = {}
            for ngram in set(total_bigram_counts.keys()):
                if ngram in book_dict:
                    book_dict_[ngram] = book_dict[ngram]
            book_bigram_mapping_[book] = book_dict_

        book_trigram_mapping_ = {}
        for book, book_dict in book_trigram_mapping.items():
            book_dict_ = {}
            for ngram in set(total_trigram_counts.keys()):
                if ngram in book_dict:
                    book_dict_[ngram] = book_dict[ngram]
            book_trigram_mapping_[book] = book_dict_

        word_statistics = {
            # {unigram: count}
            'total_unigram_counts': total_unigram_counts,
            'total_bigram_counts': total_bigram_counts,
            'total_trigram_counts': total_trigram_counts,
            # {file_name: {unigram: count}
            'book_unigram_mapping': book_unigram_mapping,
            'book_bigram_mapping': book_bigram_mapping_,
            'book_trigram_mapping': book_trigram_mapping_}
        return word_statistics


    def __tag_chunks(self, tag_type, gram_type):
        def __tag_sentence(sentence_tags, gram_type):
            if gram_type == 'unigram':
                return sentence_tags
            elif gram_type == 'bigram':
                tokens_bigram_temp = ['BOS'] + sentence_tags + ['EOS']
                tokens_bigram = ['_'.join([tokens_bigram_temp[i], tokens_bigram_temp[i+1]]) for i in range(len(tokens_bigram_temp)-1)]
                return tokens_bigram
            elif gram_type == 'trigram':
                tokens_trigram_temp = ['BOS', 'BOS'] + sentence_tags + ['EOS', 'EOS']
                tokens_trigram = ['_'.join([tokens_trigram_temp[i], tokens_trigram_temp[i+1], tokens_trigram_temp[i+2]]) for i in range(len(tokens_trigram_temp)-2)]
                return tokens_trigram
            else:
                raise Exception('Not a valid gram_type')

        def __tag_chunk(chunk, tag_type, gram_type):
            tags_path = chunk.doc_path.replace('/raw_docs', f'/{tag_type}_tags_spc_{self.sentences_per_chunk}').replace('.txt', f'_chunkid_{chunk.chunk_id}.txt')
            if os.path.exists(tags_path):
                all_sentence_tags = [line for line in load_list_of_lines(tags_path, 'str')]
            else:
                all_sentence_tags = []
                # Represent sentences as strings of tags
                for sentence in chunk.tokenized_sentences:
                    doc = self.nlp(sentence)
                    if tag_type == 'pos':
                        sentence_tags = [token.pos_.replace(' ', '') for token in doc]
                    elif tag_type == 'tag':
                        sentence_tags = [token.tag_.replace(' ', '') for token in doc]
                    elif tag_type == 'dep':
                        sentence_tags = [token.dep_.replace(' ', '') for token in doc]
                    else:
                        raise Exception('Not a valid tag_type')
                    all_sentence_tags.append(' '.join(sentence_tags))
                save_list_of_lines(all_sentence_tags, tags_path, 'str')
            
            # Count number of occurrences of tags
            chunk_tag_counter = Counter()
            for sentence_tags in all_sentence_tags:
                sentence_tags = __tag_sentence(sentence_tags.split(), gram_type)
                chunk_tag_counter.update(sentence_tags)
            return chunk_tag_counter

        tagged_chunks = {}
        corpus_tag_counter = Counter()
        for doc_chunks in self.__generate_chunks():
            for chunk in doc_chunks:
                chunk_tag_counter = __tag_chunk(chunk, tag_type, gram_type)
                tagged_chunks[chunk.file_name + '_' + str(chunk.chunk_id)] = chunk_tag_counter
                corpus_tag_counter.update(chunk_tag_counter)

        # get first k tags of corpus_tag_counter
        corpus_tag_counter = sorted([(tag, count) for tag, count in corpus_tag_counter.items()], key=lambda x: -x[1])[:self.nr_features]
        corpus_tag_counter = [tag for tag, count in corpus_tag_counter]

        data = []

        # get first k tags of each chunk_tag_counter
        for chunk_name, tagged_chunk in tagged_chunks.items():
            # create label
            current_chunk_chosen_tag_counts = dict([(tag_type + '_' + gram_type + '_' + tag_name, tagged_chunk[tag_name]) for tag_name in corpus_tag_counter])
            # relative counts
            current_chunk_chosen_tag_counts_sum = sum([count for tag, count in current_chunk_chosen_tag_counts.items()])
            current_chunk_chosen_tag_counts = dict([(tag, count/current_chunk_chosen_tag_counts_sum) for tag, count in current_chunk_chosen_tag_counts.items()])
            current_chunk_chosen_tag_counts['file_name'] = chunk_name
            data.append(current_chunk_chosen_tag_counts)

        df = pd.DataFrame(data)
        return df

    def get_tag_distribution(self):
        result_df = None
        for tag_type in ['pos']:  # ['pos', 'tag', 'dep']:
            for gram_type in ['unigram', 'bigram', 'trigram']:
                current_df = self.__tag_chunks(tag_type, gram_type)
                if result_df is None:
                    result_df = current_df
                else:
                    result_df = result_df.merge(current_df, on='file_name')
        return result_df

    def __get_book_production_counts(self, book, pre):
        chunk_production_counter = Counter()
        for sentence in book:
            sentence_production_counter = pre.get_sentence_production_counter(sentence)
            chunk_production_counter.update(sentence_production_counter)
        return chunk_production_counter

    def get_production_distribution(self):
        '''
        Returns an empty dataframe if the language is German. Reason is explained in
        docstring of ProductionRuleExtractor.
        '''
        if self.language == 'ger':
            raise Exception('Not implemented for German.')
        elif self.language == 'eng':
            pass
        else:
            raise Exception('Not a valid language')

        pre = ProductionRuleExtractor()
        chunk_production_counters = {}
        corpus_production_counter = Counter()

        for doc_chunks in self.__generate_chunks():
            for chunk in doc_chunks:
                chunk_production_counter = self.__get_book_production_counts(chunk.tokenized_sentences, pre)
                chunk_production_counters[chunk.file_name + '_' + str(chunk.chunk_id)] = chunk_production_counter
                corpus_production_counter.update(chunk_production_counter)

        # get first k tags of corpus_tag_counter
        corpus_production_counter = sorted([(tag, count) for tag, count in corpus_production_counter.items()], key=lambda x: -x[1])[:self.nr_features]
        corpus_production_counter = [tag for tag, count in corpus_production_counter]

        data = []

        # get first k tags of each chunk_tag_counter
        for chunk_name, chunk_prodution_counter in chunk_production_counters.items():
            current_chunks_chosen_production_counts = dict([(production_type, chunk_prodution_counter[production_type]) for production_type in corpus_production_counter])
            current_chunks_chosen_production_counts_sum = sum([count for tag, count in current_chunks_chosen_production_counts.items()])
            current_chunks_chosen_production_counts = dict([(tag, count/current_chunks_chosen_production_counts_sum) for tag, count in current_chunks_chosen_production_counts.items()])
            current_chunks_chosen_production_counts['file_name'] = chunk_name
            data.append(current_chunks_chosen_production_counts)

        df = pd.DataFrame(data)
        return df

    def get_overlap_score(self, embedding_type):
        if embedding_type == 'doc2vec':
            all_embeddings = self.all_doc2vec_chunk_embeddings
        elif embedding_type == 'sbert':
            all_embeddings = self.all_average_sbert_sentence_embeddings
        else:
            raise Exception(f'Not a valid embedding_type {embedding_type}.')

        cluster_means = []
        for index, current_list_of_embeddings in enumerate(all_embeddings):
            cluster_means.append(np.array(current_list_of_embeddings).mean(axis=0)) #average of all sentenfces

        labels = []
        predictions = []
        for label_index, current_list_of_embeddings in list(enumerate(all_embeddings)):
            for current_embedding in current_list_of_embeddings:
                labels.append(label_index)
                best_cluster = None
                smallest_distance = np.inf
                for prediction_index, cluster_mean in enumerate(cluster_means):
                    current_distance = np.linalg.norm(current_embedding - cluster_mean)
                    if current_distance < smallest_distance:
                        smallest_distance = current_distance
                        best_cluster = prediction_index
                predictions.append(best_cluster)
        labels = np.array(labels)
        predictions = np.array(predictions)

        file_names = []
        overlap_scores = []
        for label_index, doc_path in enumerate(self.doc_paths):
            file_name = get_bookname(doc_path)
            indices = np.argwhere(labels == label_index).ravel()
            current_predictions = predictions[indices]
            incorrect_prediction_indices = np.argwhere(current_predictions != label_index)
            # fraction of nearest chunks that are part of other books
            overlap_score = len(incorrect_prediction_indices) / len(current_predictions)
            file_names.append(file_name)
            overlap_scores.append(overlap_score)
        return pd.DataFrame.from_dict({'file_name': file_names, f'overlap_score_{embedding_type}': overlap_scores})

    def get_overlap_score_doc2vec(self):
        return self.get_overlap_score('doc2vec')

    def get_overlap_score_sbert(self):
        return self.get_overlap_score('sbert')

    def get_outlier_score(self, embedding_type):
        if embedding_type == 'doc2vec':
            all_embeddings = self.all_doc2vec_chunk_embeddings
        elif embedding_type == 'sbert':
            all_embeddings = self.all_average_sbert_sentence_embeddings
        else:
            raise Exception(f'Not a valid embedding_type {embedding_type}.')

        cluster_means = []
        for index, current_list_of_embeddings in enumerate(all_embeddings):
            cluster_means.append(np.array(current_list_of_embeddings).mean(axis=0))

        outlier_scores = []
        file_names = []
        for current_index, current_cluster_mean in enumerate(cluster_means):
            doc_path = self.doc_paths[current_index]
            file_name = get_bookname(doc_path)
            nearest_distance = np.inf
            for other_index, other_cluster_mean in enumerate(cluster_means):
                if current_index == other_index:
                    continue
                current_distance = np.linalg.norm(current_cluster_mean - other_cluster_mean)
                if current_distance < nearest_distance:
                    nearest_distance = current_distance
            outlier_scores.append(nearest_distance)
            file_names.append(file_name)
        return pd.DataFrame.from_dict({'file_name': file_names, f'outlier_score_{embedding_type}': outlier_scores})
        
    def get_outlier_score_doc2vec(self):
        return self.get_outlier_score('doc2vec')

    def get_outlier_score_sbert(self):
        return self.get_outlier_score('sbert')

    def filter_document_term_matrix(self, dtm, min_nr_documents=None, min_percent_documents=None, max_nr_documents=None, max_percent_documents=None):
        if min_nr_documents is None and min_percent_documents is None and max_nr_documents is None and max_percent_documents is None:
            raise Exception('Specify at least one filtering criterion.')
        min_columns = []
        max_columns = []
        document_frequency = dtm.astype(bool).sum(axis=0)

        #Filter minimum
        if min_nr_documents is not None and min_percent_documents is not None:
            raise Exception('Specify either the the minimum number or the minimum percentage of documents in which a term must occur.')
        elif min_percent_documents is not None:
            min_nr_documents = round(min_percent_documents/100 * dtm.shape[0])
        if min_nr_documents is not None:
            min_columns = [dtm.columns[x] for x in range(0,len(dtm.columns)) if document_frequency[x]>=min_nr_documents]

        #Filter maximum
        if max_nr_documents is not None and max_percent_documents is not None:
            raise Exception('Specify either the the maximum number or the maximum percentage of documents in which a term can occur.')
        elif max_percent_documents is not None:
            max_nr_documents = round(max_percent_documents/100 * dtm.shape[0])
        if max_nr_documents is not None:
            max_columns = [dtm.columns[x] for x in range(0,len(dtm.columns)) if document_frequency[x]<=max_nr_documents]

        if min_columns and max_columns:
            dtm_reduced = dtm[list(set(min_columns).intersection(max_columns))]
        elif min_columns:
            dtm_reduced = dtm[min_columns]
        else:
            dtm_reduced = dtm[max_columns]
        return dtm_reduced

    def get_distance_from_corpus(self, ngram_type, min_nr_documents=None, min_percent_documents=None, max_nr_documents=None, max_percent_documents=None):
        dtm = pd.DataFrame(self.word_statistics[f'book_{ngram_type}_mapping']).fillna(0).T
        dtm = self.filter_document_term_matrix(dtm, min_nr_documents, min_percent_documents, max_nr_documents, max_percent_documents)
        dtm = list(set(dtm.columns.tolist()))
        corpus_counts = self.word_statistics[f'total_{ngram_type}_counts']
        corpus_vector = [corpus_counts[key] if key in corpus_counts else 0 for key in dtm]

        distances = {}
        for doc_chunks in self.__generate_chunks(unigram_counts=True, bigram_counts=True, trigram_counts=True):
            for chunk in doc_chunks:
                
                if ngram_type == 'unigram':
                    chunk_counts = chunk.unigram_counts
                elif ngram_type == 'bigram':
                    chunk_counts = chunk.bigram_counts
                elif ngram_type == 'trigram':
                    chunk_counts = chunk.trigram_counts

                chunk_vector = [chunk_counts[key] if key in chunk_counts else 0 for key in dtm]
                # Corpus counts without the counts from the chunk
                curr_corpus_vector = list(np.subtract(np.array(corpus_vector), np.array(chunk_vector)))
                cosine_distance = scipy.spatial.distance.cosine(curr_corpus_vector, chunk_vector)
                distances[chunk.file_name + '_' + str(chunk.chunk_id)] = cosine_distance
        distances = df_from_dict(d=distances, keys_as_index=False, keys_column_name='file_name', values_column_value=f'{ngram_type}_distance')
        return distances 

    def get_unigram_distance(self):
        distances = self.get_distance_from_corpus(ngram_type='unigram', min_nr_documents=2)
        return distances

    def get_unigram_distance_limited(self):
        #Filter for mid-frequency words
        distances = self.get_distance_from_corpus(ngram_type='unigram', min_percent_documents=5, max_percent_documents=50)
        distances = distances.rename(columns={'unigram_distance': 'unigram_distance_limited'})
        return distances

    def get_bigram_distance(self):
        distances = self.get_distance_from_corpus(ngram_type='bigram', min_nr_documents=2)
        return distances

    def get_trigram_distance(self):
        distances = self.get_distance_from_corpus(ngram_type='trigram', min_nr_documents=2)
        return distances

    def get_all_features(self):
        corpus_chunk_feature_mapping = [
                                self.get_unigram_distance,
                                self.get_unigram_distance_limited,
                                self.get_bigram_distance,
                                self.get_trigram_distance,
                                self.get_tag_distribution]
        if self.language == 'eng':
            corpus_chunk_feature_mapping.append(self.get_production_distribution)
        corpus_book_feature_mapping = [
                                self.get_overlap_score_doc2vec,
                                self.get_overlap_score_sbert,
                                self.get_outlier_score_doc2vec,
                                self.get_outlier_score_sbert]

        corpus_chunk_features = None
        for feature_function in corpus_chunk_feature_mapping:
            print(feature_function)
            if  corpus_chunk_features is None:
                corpus_chunk_features = feature_function()
            else:
                corpus_chunk_features = corpus_chunk_features.merge(feature_function(), how='inner', on='file_name', validate='one_to_one')
        if self.sentences_per_chunk is None:
            corpus_chunk_features['file_name'] = corpus_chunk_features['file_name'].str.split('_').str[:4].str.join('_')

        corpus_book_features = None
        if self.sentences_per_chunk is not None:
            for feature_function in corpus_book_feature_mapping:
                if corpus_book_features is None:
                    corpus_book_features = feature_function()
                else:
                    corpus_book_features = corpus_book_features.merge(feature_function(), how='inner', on='file_name', validate='one_to_one')
        return corpus_chunk_features, corpus_book_features
