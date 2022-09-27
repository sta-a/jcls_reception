from cProfile import label
import os
from ast import literal_eval
from pathlib import Path
import numpy as np
import pandas as pd
import statistics
from unidecode import unidecode
import re


def get_bookname(doc_path):
    return doc_path.split('/')[-1][:-4]

def get_doc_paths(docs_dir, lang):
    doc_paths = [os.path.join(docs_dir, doc_name) for doc_name in os.listdir(docs_dir) if doc_name[-4:] == '.txt']
    return doc_paths

def preprocess_sentences_helper(text):
    text = text.lower()
    text = unidecode(text)
    text = re.sub('[^a-zA-Z]+', ' ', text).strip()
    text = text.split()
    text = ' '.join(text)
    return text

def df_from_dict(d, keys_as_index, keys_column_name, values_column_value):
    '''Turn both keys and values of a dict into columns of a df.'''
    df = pd.DataFrame(d.items(), columns=[keys_column_name, values_column_value])
    if keys_as_index == True:
        df = df.set_index(keys=keys_column_name)
    return df

    
def load_list_of_lines(path, line_type):
    if line_type == 'str':
        with open(path, 'r') as reader:
            lines = [line.strip() for line in reader]
    elif line_type == 'np':
        lines = list(np.load(path)['arr_0'])
    else:
        raise Exception(f'Not a valid line_type {line_type}')
    return lines


def save_list_of_lines(lst, path, line_type):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    if line_type == 'str':
        with open(path, 'w') as writer:
            for item in lst:
                writer.write(str(item) + '\n')
    elif line_type == 'np':
        np.savez_compressed(path, np.array(lst))
    else:
        raise Exception(f'Not a valid line_type {line_type}')


def read_canon_labels(labels_dir):
    labels = pd.read_csv(os.path.join(labels_dir, '210907_regression_predict_02_setp3_FINAL.csv'), sep=';')[['file_name', 'm3']]
    labels = labels.rename(columns={'m3': 'y'})
    labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
    return labels


def get_labels(lang, sentiscores_dir, metadata_dir, label_type):
    if label_type == 'library':
        if lang == 'eng':
            library_file = 'ENG_texts_circulating-libs.csv'
        else:
            library_file = 'GER_texts_circulating-libs.csv'
        labels = pd.read_csv(metadata_dir + library_file, sep=';', header=0)[['file_name', 'sum_libraries']]
        labels = labels.rename(columns={'sum_libraries': 'classified'})
        labels['classified'] = labels['classified'].apply(lambda x: 1 if x!=0 else 0)

    else:

        # Combine all labels and filenames in one file
        # File with sentiment scores for both tools
        if lang == 'eng':
            scores_file = 'ENG_reviews_senti_FINAL.csv'
        else:
            scores_file = 'GER_reviews_senti_FINAL.csv'
        score_labels = pd.read_csv(sentiscores_dir + scores_file, sep=';', header=0)

        # File with class labels
        if lang == 'eng':
            class_file = 'ENG_reviews_senti_classified.csv'
        else:
            class_file = 'GER_reviews_senti_classified.csv'
        class_labels = pd.read_csv(sentiscores_dir + class_file, sep=';', header=0)[['textfile', 'journal', 'file_name', 'classified']]

        labels = score_labels.merge(right=class_labels, 
                                    on=['textfile', 'journal', 'file_name'], 
                                    how='outer', 
                                    suffixes=('', '_scorelabels'), 
                                    validate='one_to_one')
        labels = labels[['sentiscore_average', 'sentiment_Textblob', 'file_name', 'classified']]

        # Sentiment Scores
        if (label_type == 'textblob') or (label_type == 'sentiart'):
            def _average_scores(group):
                textblob_value = group['sentiment_Textblob'].mean()
                sentiart_value = group['sentiscore_average'].mean()
                group['sentiment_Textblob'] = textblob_value
                group['sentiscore_average'] = sentiart_value
                return group        
            # Get one label per book
            labels = labels.groupby('file_name').apply(_average_scores)

        elif label_type == 'combined':
            def _aggregate_scores(row):
                if row['classified'] == 'positive':
                    score = row['sentiment_Textblob']
                elif row['classified'] == 'negative':
                    score = row['sentiscore_average']
                elif row['classified'] == 'not_classified':
                    score = statistics.mean([row['sentiment_Textblob'], row['sentiscore_average']]) 
                return score
            labels['combined'] = labels.apply(lambda row: _aggregate_scores(row), axis=1)
            #labels['combined'].sort_values().plot(kind='bar', figsize=(10, 5))

        elif (label_type == 'multiclass') or (label_type == 'twoclass'):
            #Assign one class label per work
            grouped_docs = labels.groupby('file_name')
            single_review = grouped_docs.filter(lambda x: len(x)==1)
            # If work has multiple reviews, keep labels only if reviews are not opposing (positive and negative)
            multiple_reviews = grouped_docs.filter(lambda x: len(x)>1 and not('negative' in x['classified'].values and 'positive' in x['classified'].values))
            #opposed_reviews = grouped_docs.filter(lambda x: len(x)>1 and ('negative' in x['classified'].values and 'positive' in x['classified'].values))  
            def _select_label(group):
                # Keep label with higher count, keep more extreme label if counts are equal
                count = group['classified'].value_counts().reset_index().rename(columns={'index': 'classified', 'classified': 'count'})
                if count.shape[0]>1:
                    if count.iloc[0,1] == count.iloc[1,1]:
                        grouplabel = count['classified'].max()
                    else:
                        grouplabel = count.iloc[0,0]
                    group['classified'] = grouplabel
                return group
            multiple_reviews = multiple_reviews.groupby('file_name').apply(_select_label)
            labels = pd.concat([single_review, multiple_reviews])
            labels['classified'] = labels['classified'].replace(to_replace={'positive': 3, 'not_classified': 2, 'negative': 1})

            if label_type =='twoclass':
                # Create label reviewed/not reviewed
                labels['classified'] = 1
            
    labels = labels.sort_values(by='file_name', axis=0, ascending=True, na_position='first')
    labels = labels.drop_duplicates(subset='file_name')
    if label_type == 'textblob':
        labels = labels[['file_name', 'sentiment_Textblob']].rename(columns={'sentiment_Textblob': 'y'})
    elif label_type == 'sentiart':
        labels = labels[['file_name', 'sentiscore_average']].rename(columns={'sentiscore_average': 'y'})
    elif (label_type == 'multiclass') or (label_type == 'twoclass'):
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    elif label_type == 'combined':
        labels = labels[['file_name', 'combined']].rename(columns={'combined': 'y'})
    elif label_type == 'library':
        labels = labels[['file_name', 'classified']].rename(columns={'classified': 'y'})
    return labels

