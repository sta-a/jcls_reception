# %%
# %load_ext autoreload
# %autoreload 2
from_commandline = True

import argparse
import os
import numpy as np
import pandas as pd
import time
from itertools import repeat
from multiprocessing import Pool, cpu_count
from feature_extraction.doc2vec_chunk_vectorizer import Doc2VecChunkVectorizer
from feature_extraction.doc_based_feature_extractor import DocBasedFeatureExtractor
from feature_extraction.corpus_based_feature_extractor import CorpusBasedFeatureExtractor
from utils import get_doc_paths


if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default='eng')
    parser.add_argument('--data_dir', default='../data')
    args = parser.parse_args()
    language = args.language
    data_dir = args.data_dir
else:
    # Don't use defaults because VSC interactive can't handle command line arguments
    language = 'ger'
    data_dir = '../data/'

# Select number of texts to work with
nr_texts = None # None for all texts
raw_docs_dir = os.path.join(data_dir, 'raw_docs', language)
features_dir = os.path.join(data_dir, f'features_{nr_texts}', language)
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

doc_paths = get_doc_paths(raw_docs_dir)[:nr_texts]
print(len(doc_paths))
sents_per_chunk = 200


def _create_doc2vec_embeddings():
    for curr_sentences_per_chunk in [sents_per_chunk, None]:
        doc2vec_chunk_embeddings_dir = raw_docs_dir.replace('/raw_docs', f'/doc2vec_chunk_embeddings_spc_{curr_sentences_per_chunk}')
        if not os.path.exists(doc2vec_chunk_embeddings_dir):
            d2vcv = Doc2VecChunkVectorizer(language, curr_sentences_per_chunk)
            d2vcv.fit_transform(doc_paths)


def _get_doc_features_helper(doc_path, language, sentences_per_chunk):
    fe = DocBasedFeatureExtractor(language, doc_path, sentences_per_chunk=sentences_per_chunk)
    chunk_features, book_features = fe.get_all_features() 
    return [chunk_features, book_features]
    

def _get_doc_features(sentences_per_chunk):      
    all_chunk_features = []
    all_book_features = [] 

    nr_processes = max(cpu_count() - 2, 1)
    with Pool(processes=nr_processes) as pool:
        res = pool.starmap(_get_doc_features_helper, zip(doc_paths, repeat(language), repeat(sentences_per_chunk)))
        for doc_features in res:
            all_chunk_features.extend(doc_features[0])
            all_book_features.append(doc_features[1])
            
    print(len(all_chunk_features), len(all_book_features))
    # Save book features only once (not when running with fulltext chunks)
    return all_chunk_features, all_book_features


def _get_corpus_features(sentences_per_chunk):
    cbfe = CorpusBasedFeatureExtractor(language, doc_paths, sentences_per_chunk=sentences_per_chunk, nr_features=100)
    chunk_features, book_features = cbfe.get_all_features()
    # Save book features only once (not when running with fulltext chunks)
    return chunk_features, book_features


def _merge_features(doc_chunk_features, 
                    doc_book_features, 
                    doc_chunk_features_fulltext, 
                    corpus_chunk_features, 
                    corpus_book_features, 
                    corpus_chunk_features_fulltext):
    # Book features
    doc_book_features = pd.DataFrame(doc_book_features)
    doc_chunk_features_fulltext = pd.DataFrame(doc_chunk_features_fulltext)

    print('doc_book_features: ', doc_book_features.shape, 'doc_chunk_features_fulltext: ', doc_chunk_features_fulltext.shape, 'corpus_book_features: ', corpus_book_features.shape, 'corpus_chunk_features_fulltext: ', corpus_chunk_features_fulltext.shape)

    book_df = doc_book_features\
                .merge(right=doc_chunk_features_fulltext, on='file_name', how='outer', validate='one_to_one')\
                .merge(right=corpus_book_features, on='file_name', validate='one_to_one')\
                .merge(right=corpus_chunk_features_fulltext, on='file_name', validate='one_to_one')
    book_df.columns = [col + '_fulltext' if col != 'file_name' else col for col in book_df.columns]

    # Chunk features
    doc_chunk_features = pd.DataFrame(doc_chunk_features)
    chunk_df = doc_chunk_features.merge(right=corpus_chunk_features, on='file_name', how='outer', validate='one_to_one')
    # Remove chunk id from file_name
    chunk_df['file_name'] = chunk_df['file_name'].str.split('_').str[:4].str.join('_')
    chunk_df.columns = [col + '_chunk' if col != 'file_name' else col for col in chunk_df.columns]

    # Combine book features and averages of chunksaveraged chunk features
    # baac = book and averaged chunk
    baac_df = book_df.merge(chunk_df.groupby('file_name').mean().reset_index(drop=False), on='file_name', validate='one_to_many')
    # cacb = chunk and copied book
    cacb_df = chunk_df.merge(right=book_df, on='file_name', how='outer', validate='many_to_one')
    print(book_df.shape, chunk_df.shape, baac_df.shape, cacb_df.shape)

    dfs = {'book': book_df, 'baac': baac_df, 'chunk': chunk_df, 'cacb': cacb_df}
    for name, df in dfs.items():
        print(name)
        df = df.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
        file_path = os.path.join(features_dir, f'{name}_features.csv')
        df.to_csv(file_path, index=False)
    return dfs

            
if __name__ == '__main__':
    start = time.time()
    _create_doc2vec_embeddings()
    # doc-based features
    doc_chunk_features, doc_book_features = _get_doc_features(sents_per_chunk)
    # Recalculate the chunk features for the whole book, which is treated as one chunk
    doc_chunk_features_fulltext, _ = _get_doc_features(None)
    
    # Corpus-based features
    corpus_chunk_features, corpus_book_features = _get_corpus_features(sents_per_chunk)
    # Recalculate the chunk features for the whole book, which is considered as one chunk
    corpus_chunk_features_fulltext, _ = _get_corpus_features(None)

    dfs = _merge_features(doc_chunk_features, 
                    doc_book_features, 
                    doc_chunk_features_fulltext, 
                    corpus_chunk_features, 
                    corpus_book_features, 
                    corpus_chunk_features_fulltext)

    runtime = time.time() - start
    print('Runtime with multiprocessing for all texts:', runtime)
    # with open('runtime_tracker.txt', 'a') as f:
        # f.write(f'nr_texts,runtime\n')
        # f.write(f'{nr_texts},{round(runtime, 2)}\n')