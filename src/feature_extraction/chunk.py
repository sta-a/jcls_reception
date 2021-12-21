import re
from utils import unidecode_custom


class Chunk(object):
    def __init__(self, sentences, sbert_sentence_embeddings, doc2vec_chunk_embedding):
        self.sentences = sentences
        self.sbert_sentence_embeddings = sbert_sentence_embeddings
        self.raw_text = " ".join(sentences)
        self.doc2vec_chunk_embedding = doc2vec_chunk_embedding
        self.unidecoded_raw_text = unidecode_custom(self.raw_text)
        self.processed_sentences = self.__preprocess_sentences()
        self.word_unigram_counts = self.__find_word_unigram_counts()
        self.word_bigram_counts = self.__find_word_bigram_counts()
        self.word_trigram_counts = self.__find_word_trigram_counts()
        self.char_unigram_counts = self.__find_char_unigram_counts()

    def __preprocess_sentences(self):
        def __preprocess_sentences_helper(text):
            text = text.lower()
            text = unidecode_custom(text)
            text = re.sub("[^a-zA-Z]+", " ", text).strip()
            text = text.split()
            text = " ".join(text)
            return text
        return [__preprocess_sentences_helper(sentence) for sentence in self.sentences]

    def __find_word_unigram_counts(self):
        word_unigram_counts = {}
        for processed_sentence in self.processed_sentences:
            for word_unigram in processed_sentence.split():
                if word_unigram in word_unigram_counts.keys():
                    word_unigram_counts[word_unigram] += 1
                else:
                    word_unigram_counts[word_unigram] = 1
        return word_unigram_counts

    def __find_word_bigram_counts(self):
        processed_text = "<BOS> " + " <EOS> <BOS> ".join(self.processed_sentences) + " <EOS>"
        splitted_processed_text = processed_text.split()
        word_bigram_counts = {}
        for i in range(len(splitted_processed_text) - 1):
            current_word_bigram = (splitted_processed_text[i], splitted_processed_text[i+1])
            if current_word_bigram in word_bigram_counts:
                word_bigram_counts[current_word_bigram] += 1
            else:
                word_bigram_counts[current_word_bigram] = 1
        return word_bigram_counts

    def __find_word_trigram_counts(self):
        processed_text = "<BOS> <BOS> " + " <EOS> <EOS> <BOS> <BOS> ".join(self.processed_sentences) + " <EOS> <EOS>"
        splitted_processed_text = processed_text.split()
        word_trigram_counts = {}
        for i in range(len(splitted_processed_text) - 2):
            current_word_trigram = (splitted_processed_text[i], splitted_processed_text[i+1], splitted_processed_text[i+2])
            if current_word_trigram in word_trigram_counts.keys():
                word_trigram_counts[current_word_trigram] += 1
            else:
                word_trigram_counts[current_word_trigram] = 1
        return word_trigram_counts

    def __find_char_unigram_counts(self):
        char_unigram_counts = {}
        for character in self.unidecoded_raw_text:
            if character in char_unigram_counts.keys():
                char_unigram_counts[character] += 1
            else:
                char_unigram_counts[character] = 1
        return char_unigram_counts
