import os
from pathlib import Path
import numpy as np
import pandas as pd
from unidecode import unidecode
import re


def get_bookname(doc_path):
    return Path(doc_path).stem

def get_doc_paths(docs_dir):
    doc_paths = [os.path.join(docs_dir, doc_name) for doc_name in os.listdir(docs_dir) if Path(doc_name).suffix == '.txt']
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if line_type == 'str':
        with open(path, 'w') as writer:
            for item in lst:
                writer.write(str(item) + '\n')
    elif line_type == 'np':
        np.savez_compressed(path, np.array(lst))
    else:
        raise Exception(f'Not a valid line_type {line_type}')
