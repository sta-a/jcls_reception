import os
import logging
logging.basicConfig(level=logging.DEBUG)
from sklearn.utils import shuffle
from transformers import BertTokenizer
from multiprocessing import cpu_count
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from .process import SentenceTokenizer
from utils import load_list_of_lines, save_list_of_lines


class Doc2VecChunkVectorizer():
    def __init__(self,
                 lang,
                 sentences_per_chunk=500,
                 dm=1,
                 dm_mean=1,
                 seed=42,
                 n_cores=-1):
        self.lang = lang
        if lang == 'eng':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif lang == 'ger':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        else:
            raise Exception(f'Not a valid language {lang}')
        self.sentences_per_chunk = sentences_per_chunk
        self.dm = dm
        self.dm_mean = dm_mean
        self.seed = seed
        if n_cores == -1 or n_cores is None:
            self.n_cores = cpu_count()
        else:
            self.n_cores = n_cores

    def fit_transform(self, doc_paths):
        tagged_chunks = []
        chunk_id_counter = 0
        doc_path_to_chunk_ids = {}
        logging.info('Preparing data for Doc2VecChunkVectorizer...')
        for doc_id, doc_path in enumerate(doc_paths): # fix this
            tokenized_sentences_path = doc_path.replace('/raw_docs', '/tokenized_sentences')
            if os.path.exists(tokenized_sentences_path):
                self.sentences = load_list_of_lines(tokenized_sentences_path, 'str')
            else:
                self.sentence_tokenizer = SentenceTokenizer(self.lang)
                self.sentences = self.sentence_tokenizer.tokenize(doc_path)
                save_list_of_lines(self.sentences, tokenized_sentences_path, 'str')

            if self.sentences_per_chunk is None:
                words = self.tokenizer.tokenize(' '.join(self.sentences))
                tagged_chunks.append(TaggedDocument(words=words, tags=[f'chunk_{chunk_id_counter}']))
                if doc_path in doc_path_to_chunk_ids.keys():
                    doc_path_to_chunk_ids[doc_path].append(chunk_id_counter)
                else:
                    doc_path_to_chunk_ids[doc_path] = [chunk_id_counter]
                chunk_id_counter += 1
            else:
                for i in range(0, len(self.sentences), self.sentences_per_chunk):
                    current_sentences = self.sentences[i:i+self.sentences_per_chunk]
                    if (len(current_sentences) == self.sentences_per_chunk) or (i == 0):
                        words = self.tokenizer.tokenize(' '.join(current_sentences))
                        tagged_chunks.append(TaggedDocument(words=words, tags=[f'chunk_{chunk_id_counter}']))
                        if doc_path in doc_path_to_chunk_ids.keys():
                            doc_path_to_chunk_ids[doc_path].append(chunk_id_counter)
                        else:
                            doc_path_to_chunk_ids[doc_path] = [chunk_id_counter]
                        chunk_id_counter += 1

        logging.info('Prepared data for Doc2VecChunkVectorizer.')

        logging.info('Fitting Doc2VecChunkVectorizer...')
        self.d2v_model = Doc2Vec(shuffle(tagged_chunks), #vector_size=100 by default
                                 window=10,
                                 dm=self.dm,
                                 dm_mean=self.dm_mean,
                                 workers=self.n_cores,
                                 seed=self.seed)
        logging.info('Fitted Doc2VecChunkVectorizer.')

        logging.info('Saving chunk vectors...')
        for doc_path in doc_paths:
            chunk_vectors = [self.d2v_model.dv[f'chunk_{chunk_id}'] for chunk_id in doc_path_to_chunk_ids[doc_path]]
            save_list_of_lines(chunk_vectors, doc_path.replace('/raw_docs', f'/doc2vec_chunk_embeddings_spc_{self.sentences_per_chunk}'), 'np')
        logging.info('Saved chunk vectors.')
