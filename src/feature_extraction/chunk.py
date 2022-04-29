import numpy as np
from unidecode import unidecode
from utils import get_bookname, preprocess_sentences_helper

class Chunk():
    def __init__(self, 
        sentences_per_chunk,
        doc_path, 
        chunk_id, 
        tokenized_sentences, 
        sbert_sentence_embeddings, 
        doc2vec_chunk_embedding, 
        processed_sentences=False, 
        unigram_counts=False, 
        bigram_counts=False, 
        trigram_counts=False, 
        raw_text=False, 
        unidecoded_raw_text=False, 
        char_unigram_counts=False):

        self.sentences_per_chunk = sentences_per_chunk
        self.doc_path = doc_path
        self.file_name = get_bookname(self.doc_path)
        self.chunk_id = chunk_id
        self.tokenized_sentences = tokenized_sentences
        self.sbert_sentence_embeddings = sbert_sentence_embeddings
        self.doc2vec_chunk_embedding = doc2vec_chunk_embedding
        self.processed_sentences = processed_sentences
        self.unigram_counts = unigram_counts
        self.bigram_counts = bigram_counts
        self.trigram_counts = trigram_counts
        self.raw_text = raw_text
        self.unidecoded_raw_text = unidecoded_raw_text
        self.char_unigram_counts = char_unigram_counts

        if self.processed_sentences == True:
            self.processed_sentences = self.__preprocess_sentences()
        if self.unigram_counts == True:
            self.unigram_counts = self.__find_unigram_counts()
        if self.bigram_counts == True:
            self.bigram_counts = self.__find_bigram_counts()
        if self.trigram_counts == True:
            self.trigram_counts = self.__find_trigram_counts()
        if self.raw_text == True:
            self.raw_text = self.__get_raw_text()
        if self.unidecoded_raw_text == True:
            self.unidecoded_raw_text = self.__unidecode_raw_text()
        if self.char_unigram_counts == True:
            self.char_unigram_counts = self.__find_char_unigram_counts()

    def __preprocess_sentences(self):
        return [preprocess_sentences_helper(sentence) for sentence in self.tokenized_sentences]

    def __find_unigram_counts(self):
        if type(self.processed_sentences) == bool:
            self.processed_sentences = self.__preprocess_sentences()
        unigram_counts = {}
        for processed_sentence in self.processed_sentences:
            for unigram in processed_sentence.split():
                if unigram in unigram_counts.keys():
                    unigram_counts[unigram] += 1
                else:
                    unigram_counts[unigram] = 1
        return unigram_counts

    def __find_bigram_counts(self):
        if type(self.processed_sentences) == bool:
            self.processed_sentences = self.__preprocess_sentences()
        processed_text = '<BOS> ' + ' <EOS> <BOS> '.join(self.processed_sentences) + ' <EOS>'
        processed_text_split = processed_text.split()
        bigram_counts = {}
        for i in range(len(processed_text_split) - 1):
            current_bigram = processed_text_split[i] + ' ' + processed_text_split[i+1]
            if current_bigram in bigram_counts:
                bigram_counts[current_bigram] += 1
            else:
                bigram_counts[current_bigram] = 1
        return bigram_counts

    def __find_trigram_counts(self):
        if type(self.processed_sentences) == bool:
            self.processed_sentences = self.__preprocess_sentences()
        processed_text = '<BOS> <BOS> ' + ' <EOS> <EOS> <BOS> <BOS> '.join(self.processed_sentences) + ' <EOS> <EOS>'
        processed_text_split = processed_text.split()
        trigram_counts = {}
        for i in range(len(processed_text_split) - 2):
            current_trigram = processed_text_split[i] + ' ' + processed_text_split[i+1] + ' ' + processed_text_split[i+2]
            if current_trigram in trigram_counts.keys():
                trigram_counts[current_trigram] += 1
            else:
                trigram_counts[current_trigram] = 1
        return trigram_counts

    def __get_raw_text(self):
        return ' '.join(self.tokenized_sentences)

    def __unidecode_raw_text(self):
        if type(self.raw_text) == bool:
            self.raw_text = self.__get_raw_text()
        return unidecode(self.raw_text)

    def __find_char_unigram_counts(self):
        if type(self.unidecoded_raw_text) == bool:
            self.unidecoded_raw_text = self.__unidecode_raw_text()

        char_unigram_counts = {}
        for character in self.unidecoded_raw_text:
            if character in char_unigram_counts.keys():
                char_unigram_counts[character] += 1
            else:
                char_unigram_counts[character] = 1
        return char_unigram_counts

    def __eq__(self, other):
        return (self.sentences_per_chunk == other.sentences_per_chunk) and \
                (self.doc_path == other.doc_path) and \
                (self.file_name == other.file_name) and \
                (self.chunk_id == other.chunk_id) and \
                (self.tokenized_sentences == other.tokenized_sentences) and \
                (all([np.array_equal(x,y) for x,y in zip(self.sbert_sentence_embeddings, other.sbert_sentence_embeddings)])) and \
                (all([np.array_equal(x,y) for x,y in zip(self.all_doc2vec_chunk_embeddings, other.all_doc2vec_chunk_embeddings)])) and \
                (self.processed_sentences == other.processed_sentences) and \
                (self.unigram_counts == other.unigram_counts) and \
                (self.bigram_counts == other.bigram_counts) and \
                (self.trigram_counts == other.trigram_counts) and \
                (self.char_unigram_counts == other.char_unigram_counts) and \
                (self.raw_text == other.raw_text) and \
                (self.unidecoded_raw_text == other.unidecoded_raw_text)