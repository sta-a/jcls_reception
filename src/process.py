import os
import re
import logging
logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
import pickle
import spacy
from transformers import BertTokenizer
from unidecode import unidecode


class SentenceTokenizer(object):
    def __init__(self, lang):
        self.lang = lang
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

    def tokenize(self, doc_path):
        with open(doc_path, "r") as reader:
            doc = reader.read().strip()
        i = 0
        all_sentences = []
        while True:
            current_doc = doc[i:i+500000]
            current_doc = current_doc.replace("\r\n", " ")
            current_doc = current_doc.replace("\n", " ")
            current_doc = current_doc.replace("   ", " ")
            current_doc = current_doc.replace("  ", " ")
            current_sentences = [(sent.text, sent.text_with_ws) for sent in self.nlp(current_doc).sents]
            if len(current_sentences) == 1:
                all_sentences.extend([sents[0].strip() for sents in current_sentences])
                break
            else:
                all_sentences.extend([sents[0].strip() for sents in current_sentences[:-1]])
                i += len(''.join([sents[1] for sents in current_sentences[:-1]]))
        all_sentences = [sent for sent in all_sentences if len(sent) > 3]
        return all_sentences


class Doc2VecProcessor(object):
    def __init__(self, lang, processed_chunk_sentence_count=500, stride=500):
        self.lang = lang
        self.sentence_tokenizer = SentenceTokenizer(self.lang)
        self.processed_chunk_sentence_count = processed_chunk_sentence_count
        self.stride = stride

    def process(self, doc_paths):
        logging.info("Processing texts...")
        if self.processed_chunk_sentence_count is not None:
            os.makedirs("/".join(doc_paths[0].split("/")[:-1]).replace("/raw_docs", f"/processed_docs_sc_{self.processed_chunk_sentence_count}_st_{self.stride}"), exist_ok=True)
        else:
            os.makedirs("/".join(doc_paths[0].split("/")[:-1]).replace("/raw_docs", f"/processed_doc2vec_full"), exist_ok=True)

        for doc_path in tqdm(doc_paths):
            with open(doc_path, "r") as doc_reader:
                doc = doc_reader.read()

            def _process_text(text):
                text = text.lower()
                text = unidecode(text)
                text = re.sub("[^a-zA-Z]+", " ", text).strip()
                text = " ".join(text.split())
                return text

            if self.processed_chunk_sentence_count is not None:
                if os.path.exists(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle"):
                    sentences = pickle.load(open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "rb"))
                else:
                    sentences = self.sentence_tokenizer.tokenize(doc)
                    pickle.dump(sentences, open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "wb"))
                sentences = [_process_text(sentence) for sentence in sentences]

                for i in range(0, len(doc), self.stride):
                    current_chunk = sentences[i:i+self.processed_chunk_sentence_count]
                    if (len(current_chunk) < self.processed_chunk_sentence_count) and i != 0:
                        break
                    processed_doc_path = doc_path[:-4].replace("/raw_docs", f"/processed_doc2vec_sc_{self.processed_chunk_sentence_count}_st_{self.stride}") +  f"_pt_{i}.txt"
                    with open(processed_doc_path, "w") as doc_writer:
                        doc_writer.write(" ".join(current_chunk))
            else:
                doc = _process_text(doc)
                processed_doc_path = doc_path.replace("/raw_docs", "/processed_doc2vec_full")
                with open(processed_doc_path, "w") as doc_writer:
                    doc_writer.write(doc)
        logging.info("Processed texts.")


class BertProcessor(object):
    def __init__(self, lang, pad):
        self.lang = lang
        self.pad = pad
        if self.lang == "eng":
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif self.lang == "ger":
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        else:
            raise Exception(f"Not a valid language {self.lang}")
        self.sentence_tokenizer = SentenceTokenizer(self.lang)

    def process(self, doc_paths):
        logging.info("Processing texts...")
        os.makedirs("/".join(doc_paths[0].split("/")[:-1]).replace("/raw_docs", f"/processed_bert_512_tokens"), exist_ok=True)
        os.makedirs("/".join(doc_paths[0].split("/")[:-1]).replace("/raw_docs", f"/processed_bert_sentence_tokens"), exist_ok=True)
        os.makedirs("/".join(doc_paths[0].split("/")[:-1]).replace("/raw_docs", f"/processed_sentences"), exist_ok=True)

        for doc_path in tqdm(doc_paths):
            with open(doc_path, "r") as doc_reader:
                doc = doc_reader.read()
            if os.path.exists(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle"):
                sentences = pickle.load(open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "rb"))
            else:
                sentences = self.sentence_tokenizer.tokenize(doc)
                pickle.dump(sentences, open(doc_path[:-4].replace("/raw_docs", f"/processed_sentences") + ".pickle", "wb"))

            current_paragraph = ""
            current_token_count = 0
            tokenized_paragraphs = []
            tokenized_sentences = []
            for current_sentence in sentences:
                tokenized_sentence = self.bert_tokenizer(current_sentence, return_tensors='pt')
                if tokenized_sentence['input_ids'].shape[1] > 512:
                    continue
                current_tokenized_length = tokenized_sentence['input_ids'].shape[1] - 2
                if self.pad:
                    tokenized_sentence = self.bert_tokenizer(current_sentence, return_tensors='pt', padding=True, truncation=True)
                tokenized_sentences.append(tokenized_sentence)
                if current_token_count + current_tokenized_length <= 510:
                    current_token_count += current_tokenized_length
                    if current_paragraph == "":
                        current_paragraph = current_sentence
                    else:
                        current_paragraph += " " + current_sentence
                else:
                    if self.pad:
                        tokenized_paragraph = self.bert_tokenizer(current_paragraph, return_tensors='pt', padding=True, truncation=True)
                    else:
                        tokenized_paragraph = self.bert_tokenizer(current_paragraph, return_tensors='pt', padding=False, truncation=False)
                    tokenized_paragraphs.append(tokenized_paragraph)
                    current_paragraph = current_sentence
                    current_token_count = current_tokenized_length
            for tokenized_paragraph in tokenized_paragraphs:
                if tokenized_paragraph['input_ids'].shape[1] > 512:
                    print("Long paragraph detected:", doc_path.split("/")[-1])
                    break
            
            for tokenized_sentence in tokenized_sentences:
                if tokenized_sentence['input_ids'].shape[1] > 512:
                    print("Long sentence detected:", doc_path.split("/")[-1])
                    break
            
            if self.pad:
                pickle_path = doc_path[:-4].replace("/raw_docs", f"/processed_bert_512_tokens_padded") + ".pickle"
            else:
                pickle_path = doc_path[:-4].replace("/raw_docs", f"/processed_bert_512_tokens_not_padded") + ".pickle"
            pickle.dump(tokenized_paragraphs, open(pickle_path, "wb"))
            
            if self.pad:
                pickle_path = doc_path[:-4].replace("/raw_docs", f"/processed_bert_sentence_tokens_padded") + ".pickle"
            else:
                pickle_path = doc_path[:-4].replace("/raw_docs", f"/processed_bert_sentence_tokens_not_padded") + ".pickle"
            pickle.dump(tokenized_sentences, open(pickle_path, "wb"))
            
        logging.info("Processed texts.")
