import os
import spacy
import re
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, _document_frequency
import scipy
from gensim.models import LdaMulticore
from gensim.matutils import Sparse2Corpus
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
import pickle
from collections import Counter
from utils import load_list_of_lines, save_list_of_lines, unidecode_custom
from feature_extraction.production_rule_extractor import ProductionRuleExtractor
from corpus_toolkit import corpus_tools as ct
logging.basicConfig(level=logging.DEBUG)


class CorpusBasedFeatureExtractor(object):
    def __init__(self, lang, doc_paths, all_average_sbert_sentence_embeddings, all_doc2vec_chunk_embeddings):
        self.lang = lang
        self.doc_paths = doc_paths
        self.word_statistics = self.__get_word_statistics()
        self.all_average_sbert_sentence_embeddings = all_average_sbert_sentence_embeddings
        self.all_doc2vec_chunk_embeddings = all_doc2vec_chunk_embeddings

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

        if self.lang == "eng":
            self.spell_checker = SpellChecker(language='en')
        elif self.lang == "ger":
            self.spell_checker = SpellChecker(language='de')
        else:
            raise Exception(f"Not a valid language {self.lang}")

        self.stopwords = self.nlp.Defaults.stop_words
        new_stopwords = []
        for stopword in self.stopwords:
            new_stopwords.append(unidecode_custom(stopword))
        self.stopwords = set(new_stopwords)

    def __find_unigram_counts(self, processed_sentences):
        word_unigram_counts = {}
        for processed_sentence in processed_sentences:
            for word_unigram in processed_sentence.split():
                if word_unigram in word_unigram_counts.keys():
                    word_unigram_counts[word_unigram] += 1
                else:
                    word_unigram_counts[word_unigram] = 1
        return word_unigram_counts

    def __find_bigram_counts(self, processed_sentences):
        processed_text = "<BOS> " + " <EOS> <BOS> ".join(processed_sentences) + " <EOS>"
        split_processed_text = processed_text.split()
        word_bigram_counts = {}
        for i in range(len(split_processed_text) - 1):
            current_bigram = split_processed_text[i] + " " + split_processed_text[i+1]
            if current_bigram in word_bigram_counts:
                word_bigram_counts[current_bigram] += 1
            else:
                word_bigram_counts[current_bigram] = 1
        return word_bigram_counts

    def __find_trigram_counts(self, processed_sentences):
        processed_text = "<BOS> <BOS> " + " <EOS> <EOS> <BOS> <BOS> ".join(processed_sentences) + " <EOS> <EOS>"
        split_processed_text = processed_text.split()
        word_trigram_counts = {}
        for i in range(len(split_processed_text) - 2):
            current_trigram = split_processed_text[i] + " " + split_processed_text[i+1] + " " + split_processed_text[i+2]
            if current_trigram in word_trigram_counts.keys():
                word_trigram_counts[current_trigram] += 1
            else:
                word_trigram_counts[current_trigram] = 1
        return word_trigram_counts

    def __preprocess_sentences(self, sentences):
        def __preprocess_sentences_helper(text):
            text = text.lower()
            text = unidecode_custom(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in sentences]

    def __tag_books(self, tag_type, gram_type, k):
        def __tag_sentence(tokens_unigram, gram_type):
            if gram_type == "unigram":
                return tokens_unigram
            elif gram_type == "bigram":
                tokens_bigram_temp = ["BOS"] + tokens_unigram + ["EOS"]
                tokens_bigram = ["_".join([tokens_bigram_temp[i], tokens_bigram_temp[i+1]]) for i in range(len(tokens_bigram_temp)-1)]
                return tokens_bigram
            elif gram_type == "trigram":
                tokens_trigram_temp = ["BOS", "BOS"] + tokens_unigram + ["EOS", "EOS"]
                tokens_trigram = ["_".join([tokens_trigram_temp[i], tokens_trigram_temp[i+1], tokens_trigram_temp[i+2]]) for i in range(len(tokens_trigram_temp)-2)]
                return tokens_trigram
            else:
                raise Exception("Not a valid gram_type")

        def __tag_book(doc_path, tag_type, gram_type):
            tags_path = doc_path.replace("/raw_docs", f"/{tag_type}_tags")
            if os.path.exists(tags_path):
                all_tokens_unigram = [line for line in load_list_of_lines(tags_path, "str")]
            else:
                sentences_path = doc_path.replace("/raw_docs", "/processed_sentences")
                sentences = load_list_of_lines(sentences_path, "str")
                all_tokens_unigram = []
                for sentence in sentences:
                    doc = self.nlp(sentence)
                    if tag_type == "pos":
                        tokens_unigram = [token.pos_.replace(" ", "") for token in doc]
                    elif tag_type == "tag":
                        tokens_unigram = [token.tag_.replace(" ", "") for token in doc]
                    elif tag_type == "dep":
                        tokens_unigram = [token.dep_.replace(" ", "") for token in doc]
                    else:
                        raise Exception("Not a valid tag_type")
                    all_tokens_unigram.append(" ".join(tokens_unigram))
                save_list_of_lines(all_tokens_unigram, tags_path, "str")

            book_tag_counter = Counter()
            for tokens_unigram in all_tokens_unigram:
                sentence_tags = __tag_sentence(tokens_unigram.split(), gram_type)
                book_tag_counter.update(sentence_tags)
            return book_tag_counter

        tagged_books = {}
        corpus_tag_counter = Counter()
        for doc_path in self.doc_paths:
            book_name = doc_path.split("/")[-1][:-4]
            book_tag_counter = __tag_book(doc_path, tag_type, gram_type)
            tagged_books[book_name] = book_tag_counter
            corpus_tag_counter.update(book_tag_counter)

        # get first k tags of corpus_tag_counter
        corpus_tag_counter = sorted([(tag, count) for tag, count in corpus_tag_counter.items()], key=lambda x: -x[1])[:k]
        corpus_tag_counter = [tag for tag, count in corpus_tag_counter]

        data = []

        # get first k tags of each book_tag_counter
        for book_name, tagged_book in tagged_books.items():
            current_books_chosen_tag_counts = dict([(tag_type + "_" + gram_type + "_" + tag_name, tagged_book[tag_name]) for tag_name in corpus_tag_counter])
            current_books_chosen_tag_counts_sum = sum([count for tag, count in current_books_chosen_tag_counts.items()])
            current_books_chosen_tag_counts = dict([(tag, count/current_books_chosen_tag_counts_sum) for tag, count in current_books_chosen_tag_counts.items()])
            current_books_chosen_tag_counts["book_name"] = book_name
            data.append(current_books_chosen_tag_counts)

        df = pd.DataFrame(data)
        return df

    def get_tag_distribution(self, k):
        result_df = None
        for tag_type in ['pos']:  # ['pos', 'tag', 'dep']:
            for gram_type in ['unigram', 'bigram', 'trigram']:
                current_df = self.__tag_books(tag_type, gram_type, k)
                if result_df is None:
                    result_df = current_df
                else:
                    result_df = result_df.merge(current_df, on='book_name')
        return result_df

    def get_spelling_error_distribution(self):
        def __get_spelling_error_count_in_sentence(sentence):
            misspelled = self.spell_checker.unknown(sentence.split())
            return len(misspelled)

        def __get_spelling_error_rate_in_book(book):
            error_counter = sum([__get_spelling_error_count_in_sentence(sentence) for sentence in book])
            error_rate = error_counter / len(book)
            return error_rate

        data = []
        for doc_path in self.doc_paths:
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", "/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            error_rate = __get_spelling_error_rate_in_book(sentences)
            data.append({"book_name": book_name,
                         "error_rate": error_rate})
        df = pd.DataFrame(data)
        return df

    def __get_book_production_counts(self, book, pre):
        book_production_counter = Counter()
        for sentence in book:
            sentence_production_counter = pre.get_sentence_production_counter(sentence)
            book_production_counter.update(sentence_production_counter)
        return book_production_counter

    def get_production_distribution(self, k):
        """
        Returns an empty dataframe if the language is German. Reason is explained in
        docstring of ProductionRuleExtractor.
        """
        if self.lang == "ger":
            return pd.DataFrame(data=[doc_path.split("/")[-1][:-4] for doc_path in self.doc_paths], columns=["book_name"])
        elif self.lang == "eng":
            pass
        else:
            raise Exception("Not a valid language")

        pre = ProductionRuleExtractor()
        book_production_counters = {}
        corpus_production_counter = Counter()

        for doc_path in self.doc_paths:
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", "/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            book_production_counter = self.__get_book_production_counts(sentences, pre)
            book_production_counters[book_name] = book_production_counter
            corpus_production_counter.update(book_production_counter)

        # get first k tags of corpus_tag_counter
        corpus_production_counter = sorted([(tag, count) for tag, count in corpus_production_counter.items()], key=lambda x: -x[1])[:k]
        corpus_production_counter = [tag for tag, count in corpus_production_counter]

        data = []

        # get first k tags of each book_tag_counter
        for book_name, book_prodution_counter in book_production_counters.items():
            current_books_chosen_production_counts = dict([(production_type, book_prodution_counter[production_type]) for production_type in corpus_production_counter])
            current_books_chosen_production_counts_sum = sum([count for tag, count in current_books_chosen_production_counts.items()])
            current_books_chosen_production_counts = dict([(tag, count/current_books_chosen_production_counts_sum) for tag, count in current_books_chosen_production_counts.items()])
            current_books_chosen_production_counts["book_name"] = book_name
            data.append(current_books_chosen_production_counts)

        df = pd.DataFrame(data)
        return df

    def __get_word_statistics(self):
        # get total counts over all documents
        all_unigram_counts = Counter()
        book_unigram_mapping_abs = {}
        book_unigram_mapping_rel = {}

        for doc_path in tqdm(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
            sentences = load_list_of_lines(sentences_path, "str")
            processed_sentences = self.__preprocess_sentences(sentences)
            unigram_counts = self.__find_unigram_counts(processed_sentences)
            all_unigram_counts.update(unigram_counts)            
            total_unigram_count = sum(unigram_counts.values())# get counts per document            
            book_unigram_mapping_abs[book_name] = unigram_counts#absolute frequency            
            book_unigram_mapping_rel[book_name] = dict((unigram, count / total_unigram_count) for unigram, count in unigram_counts.items() if unigram in all_unigram_counts.keys()) #relative frequencies

        all_unigram_counts = dict(sorted(list(all_unigram_counts.items()), key=lambda x: -x[1])) #all words
        word_statistics = {
            "all_unigram_counts": all_unigram_counts,
            "book_unigram_mapping_abs": book_unigram_mapping_abs,
            "book_unigram_mapping_rel": book_unigram_mapping_rel,
        }
        return word_statistics

    def __add_bigrams_trigrams_statistics(self):
        if "all_bigram_counts" in self.word_statistics:
            return self.word_statistics
        else:
            all_bigram_counts = Counter()
            all_trigram_counts = Counter()
            book_bigram_mapping_abs = {}
            book_trigram_mapping_abs = {}
            book_bigram_mapping_rel = {}
            book_trigram_mapping_rel = {}

            for doc_path in tqdm(self.doc_paths):
                book_name = doc_path.split("/")[-1][:-4]
                sentences_path = doc_path.replace("/raw_docs", f"/processed_sentences")
                sentences = load_list_of_lines(sentences_path, "str")
                processed_sentences = self.__preprocess_sentences(sentences)
                
                bigram_counts = self.__find_bigram_counts(processed_sentences)
                trigram_counts = self.__find_trigram_counts(processed_sentences)
                all_bigram_counts.update(bigram_counts)
                all_trigram_counts.update(trigram_counts)
                total_bigram_count = sum(bigram_counts.values())
                total_trigram_count = sum(trigram_counts.values())
                book_bigram_mapping_abs[book_name] = bigram_counts
                book_trigram_mapping_abs[book_name] = trigram_counts
                book_bigram_mapping_rel[book_name] = dict((bigram, count / total_bigram_count) for bigram, count in bigram_counts.items() if bigram in all_bigram_counts.keys())
                book_trigram_mapping_rel[book_name] = dict((trigram, count / total_trigram_count) for trigram, count in trigram_counts.items() if trigram in all_trigram_counts.keys())

            all_bigram_counts = dict(sorted(list(all_bigram_counts.items()), key=lambda x: -x[1])[:2000])
            all_trigram_counts = dict(sorted(list(all_trigram_counts.items()), key=lambda x: -x[1])[:2000])

            #filter dicts so that they only contain most frequent n-grams
            for mapping in [book_bigram_mapping_abs, book_bigram_mapping_rel]:
                for book, book_dict in mapping.items():
                    for ngram in list(book_dict.keys()):
                        if ngram not in list(all_bigram_counts.keys()):
                            del book_dict[ngram]
            for mapping in [book_trigram_mapping_abs, book_trigram_mapping_rel]:
                for book, book_dict in mapping.items():
                    for ngram in list(book_dict.keys()):
                        if ngram not in list(all_trigram_counts.keys()):
                            del book_dict[ngram]

            word_stat_dir = self.word_statistics
            word_stat_dir["all_bigram_counts"] = all_bigram_counts
            word_stat_dir["all_trigram_counts"] = all_trigram_counts
            word_stat_dir["book_bigram_mapping_abs"] = book_bigram_mapping_abs
            word_stat_dir["book_trigram_mapping_abs"] = book_trigram_mapping_abs
            word_stat_dir["book_bigram_mapping_rel"] = book_bigram_mapping_rel
            word_stat_dir["book_trigram_mapping_rel"] = book_trigram_mapping_rel
            return word_stat_dir


    def get_k_most_common_ngram_counts(self, k, n, include_stopwords):
        if n == 1:
            dct1 = self.word_statistics["all_unigram_counts"]
            dct2 = self.word_statistics["book_unigram_mapping_rel"]
        elif n == 2 or n == 3:
            self.word_statistics = self.__add_bigrams_trigrams_statistics()
            if n == 2:
                dct1 = self.word_statistics["all_bigram_counts"]
                dct2 = self.word_statistics["book_bigram_mapping_rel"]
            else:
                dct1 = self.word_statistics["all_trigram_counts"]
                dct2 = self.word_statistics["book_trigram_mapping_rel"]
        else:
            raise Exception(f"Not a valid n: {n}")
        if include_stopwords:
            # words that are the most common in the whole corpus
            most_common_k_ngrams = [ngram for ngram, count in sorted(list(dct1.items()), key=lambda x: -x[1])[:k]]
        else:
            filtered_ngrams = []
            for ngram, count in dct1.items():
                split_ngram = ngram.split()
                exclude = False
                for word in split_ngram:
                    if word in self.stopwords:
                        exclude = True
                if exclude:
                    continue
                else:
                    filtered_ngrams.append((ngram, count))
            most_common_k_ngrams = [ngram for ngram, count in sorted(filtered_ngrams, key=lambda x: -x[1])[:k]]
        result = []
        for book_name, ngram_counts in dct2.items():
            # get freq of k most common n-grams in current document
            dct = dict((f"{k}_most_common_{n}gram_stopword_{include_stopwords}_{common_ngram}", dct2[book_name].get(common_ngram, 0)) for common_ngram in most_common_k_ngrams)
            dct["book_name"] = book_name
            result.append(dct)
        result = pd.DataFrame(result)
        return result

    def get_k_most_common_unigram_counts_including_stopwords(self, k):
        return self.get_k_most_common_ngram_counts(k, 1, True)

    def get_k_most_common_bigram_counts_including_stopwords(self, k):
        return self.get_k_most_common_ngram_counts(k, 2, True)

    def get_k_most_common_trigram_counts_including_stopwords(self, k):
        return self.get_k_most_common_ngram_counts(k, 3, True)

    def get_k_most_common_unigram_counts_excluding_stopwords(self, k):
        return self.get_k_most_common_ngram_counts(k, 1, False)

    def get_k_most_common_bigram_counts_excluding_stopwords(self, k):
        return self.get_k_most_common_ngram_counts(k, 2, False)

    def get_k_most_common_trigram_counts_excluding_stopwords(self, k):
        return self.get_k_most_common_ngram_counts(k, 3, False)

    def get_overlap_score(self, embedding_type):
        if embedding_type == "doc2vec":
            all_embeddings = self.all_doc2vec_chunk_embeddings
        elif embedding_type == "sbert":
            all_embeddings = self.all_average_sbert_sentence_embeddings
        else:
            raise Exception(f"Not a valid embedding_type {embedding_type}.")

        cluster_means = []
        for index, current_list_of_embeddings in enumerate(all_embeddings):
            cluster_means.append(np.array(current_list_of_embeddings).mean(axis=0))

        labels = []
        predictions = []
        for label_index, current_list_of_embeddings in tqdm(list(enumerate(all_embeddings))):
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

        book_names = []
        overlap_scores = []
        for label_index, doc_path in enumerate(self.doc_paths):
            book_name = doc_path.split("/")[-1][:-4]
            indices = np.argwhere(labels == label_index).ravel()
            current_predictions = predictions[indices]
            incorrect_prediction_indices = np.argwhere(current_predictions != label_index)
            overlap_score = len(incorrect_prediction_indices) / len(current_predictions)
            book_names.append(book_name)
            overlap_scores.append(overlap_score)
        return pd.DataFrame.from_dict({"book_name": book_names, f"overlap_score_{embedding_type}": overlap_scores})

    def get_overlap_score_doc2vec(self):
        return self.get_overlap_score("doc2vec")

    def get_overlap_score_sbert(self):
        return self.get_overlap_score("sbert")

    def get_outlier_score(self, embedding_type):
        if embedding_type == "doc2vec":
            all_embeddings = self.all_doc2vec_chunk_embeddings
        elif embedding_type == "sbert":
            all_embeddings = self.all_average_sbert_sentence_embeddings
        else:
            raise Exception(f"Not a valid embedding_type {embedding_type}.")

        cluster_means = []
        for index, current_list_of_embeddings in enumerate(all_embeddings):
            cluster_means.append(np.array(current_list_of_embeddings).mean(axis=0))

        outlier_scores = []
        book_names = []
        for current_index, current_cluster_mean in enumerate(cluster_means):
            doc_path = self.doc_paths[current_index]
            book_name = doc_path.split("/")[-1][:-4]
            nearest_distance = np.inf
            for other_index, other_cluster_mean in enumerate(cluster_means):
                if current_index == other_index:
                    continue
                current_distance = np.linalg.norm(current_cluster_mean - other_cluster_mean)
                if current_distance < nearest_distance:
                    nearest_distance = current_distance
            outlier_scores.append(nearest_distance)
            book_names.append(book_name)
        return pd.DataFrame.from_dict({"book_name": book_names, f"outlier_score_{embedding_type}": outlier_scores})

    def get_outlier_score_doc2vec(self):
        return self.get_outlier_score("doc2vec")

    def get_outlier_score_sbert(self):
        return self.get_outlier_score("sbert")

    def get_lda_topic_distribution(self):
        num_topics = 10

        documents = []
        for doc_path in self.doc_paths:
            with open(doc_path, "r") as reader:
                documents.append(reader.read().strip())

        if self.lang == "eng":
            stop_words = spacy.lang.en.stop_words.STOP_WORDS
        elif self.lang == "ger":
            stop_words = spacy.lang.de.stop_words.STOP_WORDS
        else:
            raise Exception(f"Not a valid language {self.lang}")

        vect = CountVectorizer(min_df=20, max_df=0.2, stop_words=stop_words,
                               token_pattern='(?u)\\b\\w\\w\\w+\\b')
        X = vect.fit_transform(documents)
        corpus = Sparse2Corpus(X, documents_columns=False)
        id_map = dict((v, k) for k, v in vect.vocabulary_.items())
        lda_model = LdaMulticore(corpus=corpus, id2word=id_map, passes=2, random_state=42, num_topics=num_topics, workers=3)

        topic_distributions = []
        book_names = []
        for doc_path, document in zip(self.doc_paths, documents):
            book_name = doc_path.split("/")[-1][:-4]
            string_input = [document]
            X = vect.transform(string_input)
            corpus = Sparse2Corpus(X, documents_columns=False)
            output = list(lda_model[corpus])[0]
            full_output = [0] * num_topics
            for topic_id, ratio in output:
                full_output[topic_id] = ratio
            topic_distributions.append(full_output)
            book_names.append(book_name)
        topic_distributions = pd.DataFrame(topic_distributions, columns=[f"lda_topic_{i+1}" for i in range(num_topics)])
        topic_distributions["book_name"] = book_names
        return topic_distribution

    def filter_document_term_matrix(self, df, min_nr_documents=None, min_percent_documents=None, max_nr_documents=None, max_percent_documents=None):
        if min_nr_documents==None and min_percent_documents==None and max_nr_documents==None and max_percent_documents==None:
            raise Exception("Specify at least one filtering criterion.")

        min_columns = []
        max_columns = []
        #Filter minimum
        if min_nr_documents!=None and min_percent_documents!=None:
            raise Exception("Specify either the the minimum number of documents or the minimum percentage of documents in which a term must occur.")
        elif min_percent_documents!=None:
            min_nr_documents = round(min_percent_documents/100 * df.shape[0])
        elif min_nr_documents!=None:
            document_frequency = df.astype(bool).sum(axis=0)
            min_columns = [df.columns[x] for x in range(0,len(df.columns)) if document_frequency[x]>=min_nr_documents]

        #Filter maximum
        if max_nr_documents!=None and max_percent_documents!=None:
            raise Exception("Specify either the the maximum number of documents or the maximum percentage of documents in which a term can occur.")
        elif max_percent_documents!=None:
            max_nr_documents = round(max_percent_documents/100 * df.shape[0])
        elif max_nr_documents!=None:
            document_frequency = df.astype(bool).sum(axis=0)
            max_columns = [df.columns[x] for x in range(0,len(df.columns)) if document_frequency[x]<=max_nr_documents]

        df_reduced = df[list(set(min_columns + max_columns))]
        return df_reduced

    def get_tfidf(self, k=50):
        document_term_matrix = pd.DataFrame.from_dict(self.word_statistics['book_unigram_mapping_abs']).fillna(0).T
        # Tfidf
        t = TfidfTransformer(norm='l1', use_idf=True, smooth_idf=True)
        tfidf = pd.DataFrame.sparse.from_spmatrix(t.fit_transform(document_term_matrix), columns=document_term_matrix.columns, index=document_term_matrix.index)
        tfidf_reduced = self.filter_document_term_matrix(tfidf, min_percent_documents=10)
        # From remaining words, keep only those that are in the top k for at least one book
        all_top_k_words = []
        for index, row in tfidf_reduced.iterrows():
            top_k_words = row.nlargest(n=k, keep='all')
            all_top_k_words.extend(top_k_words.index.to_list())
        all_top_k_words = list(set(all_top_k_words))
        tfidf_top_k = tfidf_reduced[all_top_k_words]
        tfidf_top_k.columns = [f"tfidf_{column}" for column in tfidf_top_k.columns]
        tfidf_top_k = tfidf_top_k.reset_index().rename(columns={'level_0':'book_name', 'index':'book_name'}) #automatically created column name can be 'index' or 'level_0'
        return tfidf_top_k

    def get_distance_from_corpus(self, df, column_name, min_nr_documents=None, min_percent_documents=None, max_nr_documents=None, max_percent_documents=None):
        df_reduced = self.filter_document_term_matrix(df, min_nr_documents=2)
        distances = []
        for index, document in tqdm(df.iterrows()):
            df_reduced = df.drop(labels=document.name, axis=0, inplace=False)
            mean_rel_frequencies = df_reduced.mean(axis=0)
            cosine_distance = scipy.spatial.distance.cosine(document, mean_rel_frequencies)
            distances.append(cosine_distance)
        if min_nr_documents!=None:
            column_name = "distance_" + column_name + "_minnrdocs_" + str(min_nr_documents) + "_maxnrdocs_" + str(max_nr_documents)
        else: 
            column_name = "distance_" + column_name + "_minpercdocs_" + str(min_percent_documents) + "_maxpercdocs_" + str(max_percent_documents)
        distances = pd.DataFrame(distances, columns=[column_name])
        distances['book_name'] = df.index
        return distances

    def get_unigram_distance(self):
        document_term_matrix_relative = pd.DataFrame.from_dict(self.word_statistics['book_unigram_mapping_rel']).fillna(0).T 
        distances = self.get_distance_from_corpus(document_term_matrix_relative, column_name="unigram", min_nr_documents=2)
        return distances

    def get_unigram_distance_limited(self):
        #Filter for mid-frequency words
        document_term_matrix_relative = pd.DataFrame.from_dict(self.word_statistics['book_unigram_mapping_rel']).fillna(0).T 
        distances = self.get_distance_from_corpus(document_term_matrix_relative, column_name="unigram_limited", min_percent_documents=5, max_percent_documents=50)
        return distances

    def get_bigram_distance(self):
        if not "all_bigram_counts" in self.word_statistics:
            self.word_statistics = self.__add_bigrams_trigrams_statistics()
        document_bigram_matrix_relative = pd.DataFrame.from_dict(self.word_statistics['book_bigram_mapping_rel']).fillna(0).T 
        distances = self.get_distance_from_corpus(document_bigram_matrix_relative, column_name="bigram", min_nr_documents=2)
        return distances

    def get_trigram_distance(self):
        if not "all_trigram_counts" in self.word_statistics:
            self.word_statistics = self.__add_bigrams_trigrams_statistics()
        document_trigram_matrix_relative = pd.DataFrame.from_dict(self.word_statistics['book_trigram_mapping_rel']).fillna(0).T 
        distances = self.get_distance_from_corpus(document_trigram_matrix_relative, column_name="trigram", min_nr_documents=2)
        return distances

    def get_all_features(self, k=100):
        ''' Get corpus-based features

        Args:
            k (int): number of features to return

        Returns:
            pd.DataFrame of corpus-based features
        '''
        result = None
        for feature_function in [# self.get_lda_topic_distribution,
                                self.get_k_most_common_unigram_counts_including_stopwords(k=k),
                                self.get_k_most_common_bigram_counts_including_stopwords(k=k),
                                self.get_k_most_common_trigram_counts_including_stopwords(k=k),

                                self.get_overlap_score_doc2vec(),
                                self.get_overlap_score_sbert(),
                                self.get_outlier_score_doc2vec(),
                                self.get_outlier_score_sbert(),
                                self.get_unigram_distance(),
                                self.get_unigram_distance_limited(),
                                self.get_bigram_distance(),
                                self.get_trigram_distance(),
                                self.get_tag_distribution(k=30),
                                self.get_spelling_error_distribution(),
                                self.get_production_distribution(k=30),  # this returns an empty dataframe if language is German
                                ]:
            if result is None:
                result = feature_function
            else:
                result = result.merge(feature_function, on="book_name")
        return result
