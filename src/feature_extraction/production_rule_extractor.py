import nltk
from collections import Counter
from nltk.corpus import conll2000
from nltk.chunk.util import tree2conlltags, conlltags2tree
from nltk.tag import UnigramTagger, BigramTagger
from nltk.chunk import ChunkParserI


class ProductionRuleExtractor(ChunkParserI):
    '''''''''
    Code taken from:
    https://www.kdnuggets.com/2018/08/understanding-language-syntax-and-structure-practitioners-guide-nlp-3.html

    According to here: https://www.reddit.com/r/Python/comments/fackw/setting_up_a_german_chunked_language_corpus_to/
                       it is not easy to find a chunker for German.
    '''
    def __init__(self, tagger_classes=[UnigramTagger, BigramTagger]):
        train_sent_tags = self.conll_tag_chunks(conll2000.chunked_sents())
        self.chunk_tagger = self.combined_tagger(train_sent_tags, tagger_classes)

    def conll_tag_chunks(self, chunk_sents):
        tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
        return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

    def combined_tagger(self, train_data, taggers, backoff=None):
        for tagger in taggers:
            backoff = tagger(train_data, backoff=backoff)
        return backoff

    def parse(self, sentence):
        tagged_sentence = nltk.pos_tag(sentence.split())
        pos_tags = [tag for word, tag in tagged_sentence]
        chunk_pos_tags = self.chunk_tagger.tag(pos_tags)
        chunk_tags = [chunk_tag for (pos_tag, chunk_tag) in chunk_pos_tags]
        wpc_tags = [(word, pos_tag, chunk_tag) for ((word, pos_tag), chunk_tag) in zip(tagged_sentence, chunk_tags)]
        chunk_tree = conlltags2tree(wpc_tags)
        return chunk_tree

    def get_productions(self, sentence):
        chunk_tree = self.parse(sentence)
        productions = []
        for production in chunk_tree.productions():
            lhs = str(production.lhs())
            rhs = [a[1] if type(a) == tuple else a for a in production.rhs()]
            production_str = lhs + '->' + '_'.join([str(a) for a in rhs])
            productions.append(production_str)
        return productions

    def get_sentence_production_counter(self, sentence):
        productions = self.get_productions(sentence)
        sentence_production_counter = Counter(productions)
        return sentence_production_counter
