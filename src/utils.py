import os
from ast import literal_eval
from pathlib import Path
import numpy as np
import pandas as pd
from unidecode import unidecode
import statistics

german_special_chars = {'Ä':'Ae', 'Ö':'Oe', 'Ü':'Ue', 'ä':'ae', 'ö':'oe', 'ü':'ue', 'ß':'ss'}


def get_doc_paths(docs_dir, lang):
    doc_paths = [os.path.join(docs_dir, lang, doc_name) for doc_name in os.listdir(os.path.join(docs_dir, lang)) if doc_name[-4:] == ".txt"]
    return doc_paths


def load_list_of_lines(path, line_type):
    if line_type == "str":
        with open(path, "r") as reader:
            lines = [line.strip() for line in reader]
    elif line_type == "np":
        lines = list(np.load(path)["arr_0"])
    else:
        raise Exception(f"Not a valid line_type {line_type}")
    return lines


def save_list_of_lines(lst, path, line_type):
    os.makedirs(str(Path(path).parent), exist_ok=True)
    if line_type == "str":
        with open(path, "w") as writer:
            for item in lst:
                writer.write(str(item) + "\n")
    elif line_type == "np":
        np.savez_compressed(path, np.array(lst))
    else:
        raise Exception(f"Not a valid line_type {line_type}")


def read_labels(labels_dir):
    labels_df = pd.read_csv(os.path.join(labels_dir, "210907_regression_predict_02_setp3_FINAL.csv"), sep=";")[["file_name", "m3"]]
    print(type(labels_df))
    labels = dict(labels_df.values)
    return labels


def read_sentiment_scores(sentiment_dir, canonization_labels_dir, lang):
    if lang == "eng":
        file_name = "ENG_reviews_senti_classified.csv"
    else:
        file_name = "GER_reviews_senti_classified.csv"
    canonization_scores = pd.read_csv(canonization_labels_dir + "210907_regression_predict_02_setp3_FINAL.csv", sep=';', header=0)[["id", "file_name"]]
    labels = pd.read_csv(sentiment_dir + file_name, sep=";", header=0)[["text_id", "sentiscore_average", "sentiment_Textblob","classified"]]
    labels = labels.merge(right=canonization_scores, how="left", right_on="id", left_on="text_id", validate="many_to_one")    
    labels = labels.rename(columns={"classified": "c", "file_name": "book_name"})

    def _aggregate_scores(row):
        if row["c"] == "positive":
            y = row["sentiment_Textblob"]
        elif row["c"] == "negative":
            y = row["sentiscore_average"]
        elif row["c"] == "not_classified":
            y = statistics.mean([row["sentiment_Textblob"], row["sentiscore_average"]]) 
        return y
    labels["y"] = labels.apply(lambda row: _aggregate_scores(row), axis=1)
    labels["y"].sort_values().plot(kind="bar", figsize=(10, 5))

    #Aggregate works with multiple reviews
    #Assign one label per work
    single_review = labels.groupby("book_name").filter(lambda x: len(x)==1)
    #multiple_reviews = labels.groupby("book_name").apply(lambda x: print(("negative" in x["c"].values and "positive" in x["c"].values)))
    multiple_reviews = labels.groupby("book_name").filter(lambda x: len(x)>1 and not("negative" in x["c"].values and "positive" in x["c"].values))
    opposed_reviews = labels.groupby("book_name").filter(lambda x: len(x)>1 and ("negative" in x["c"].values and "positive" in x["c"].values))  
    def _select_label(group):
        count = group["c"].value_counts().reset_index().rename(columns={'index': 'c', "c": "count"})
        #take label with the highest count, take more extreme label if counts are equal
        if count.shape[0]>1:
            if count.iloc[0,1] == count.iloc[1,1]:
                grouplabel = count["c"].max()
            else:
                grouplabel = count.iloc[0,0]
            group["c"] = grouplabel
        return group
    multiple_reviews = multiple_reviews.groupby("book_name").apply(_select_label)
    multiple_reviews = multiple_reviews.drop_duplicates(subset="book_name")
    labels = pd.concat([single_review, multiple_reviews])
    
    labels["c"] = labels["c"].replace(to_replace={"positive": 3, "not_classified": 2, "negative": 1})
    return labels[["book_name", "y", "c"]]


def read_library_scores(sentiment_dir, canonization_labels_dir, lang):
    if lang == "eng":
        file_name = "ENG_texts_circulating-libs.csv"
    else:
        file_name = "GER_texts_circulating-libs.csv"
    canonization_scores = pd.read_csv(canonization_labels_dir + "210907_regression_predict_02_setp3_FINAL.csv", sep=';', header=0)[["id", "file_name"]]
    scores = pd.read_csv(sentiment_dir + file_name, sep=";", header=0)[["id", "sum_libraries"]]
    scores = scores.merge(right=canonization_scores, how="left", on="id", validate="one_to_one")
    scores = scores.rename(columns={"sum_libraries": "y", "file_name": "book_name"})[["book_name", "y"]]
    return scores

def unidecode_custom(text):
    for char, replacement in german_special_chars.items():
        text = text.replace(char, replacement)
    text = unidecode(text)
    return text

def read_sentiart_textblob_scores(sentiment_dir, canonization_labels_dir, lang):
    if lang == "eng":
        file_name = "ENG_reviews_senti_FINAL.csv"
    else:
        file_name = "GER_reviews_senti_FINAL.csv"
    canonization_scores = pd.read_csv(canonization_labels_dir + "210907_regression_predict_02_setp3_FINAL.csv", sep=';', header=0)[["id", "file_name"]]
    labels = pd.read_csv(sentiment_dir + file_name, sep=";", header=0)[["text_id", "sentiscore_average", "sentiment_Textblob"]]
    labels = labels.merge(right=canonization_scores, how="left", right_on="id", left_on="text_id", validate="many_to_one")    
    labels = labels.rename(columns={"file_name": "book_name"})
    print(labels)

    print(labels.shape)
    def _aggregate_scores(group):
        textblob_value = group["sentiment_Textblob"].mean()
        sentiart_value = group["sentiscore_average"].mean()
        group["sentiment_Textblob"] = textblob_value
        group["sentiscore_average"] = sentiart_value
        return group
    labels = labels.groupby("book_name").apply(_aggregate_scores)
    print(labels.shape)

    labels = labels.drop_duplicates(subset="book_name")
    print(labels.shape)
    return labels[["book_name", "sentiscore_average", "sentiment_Textblob"]]