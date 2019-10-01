"""
Sentence Probability by unigrams or bigrams in a corpus
Usage: python3 language_models.py
"""
__author__ = "Miquel Puig Mena"

import math
import regex as re
from mutual_info import count_bigrams
from mutual_info import count_unigrams


class LanguageModels:
    DEF_start_tag = "<s>"
    DEF_end_tag = "</s>"
    UNIGRAM = "Unigram Model"
    BIGRAM = "Bigram Model"

    def __init__(self, file_text, ngram):
        self.processed_by_ngrams = {}
        self.corpus_size = 0
        self.ngram = ngram
        self.normalize_corpus(file_text)

    @staticmethod
    def tokenize(text):
        """
        Splits string text by non-letter symbols except for [<,>,/].
        Those symbols are reserved for sentence splitting meaning that they will from a token by them self.

        :param text: String. Text to split.
        :return: String[].
        """
        return re.split(r'[^\p{L}\<\>\/]+', text)

    @staticmethod
    def coliding_tuples(list):
        """
        Contains pair of tuples containing coliding elements in priginal list.
        If original_list was length N, tuples_list will be length N-1 containing N-1 tuples.

        :param list: List containing elements.
        :return: Tuple[]
        """
        tuples_list = []
        for iteration, element in enumerate(list[:-1]):
            tuples_list.append((element, list[iteration+1]))
        return tuples_list

    def increase_corpus_size(self, to_add):
        self.corpus_size += to_add

    def get_corpus_size(self):
        return self.corpus_size

    def set_ngram(self, ngram):
        self.ngram = ngram

    def get_ngram(self):
        return self.ngram

    def normalize_corpus(self, text, start_tag=DEF_start_tag, end_tag=DEF_end_tag):
        """
        Splits text by sentences and encapsulates with <s>sentence</s>.
        Processes text by counting ngrams of sentenced_text and persists in model.

        :param text:        String. Text to use as Dataset.
        :param start_tag:   String. Encapsulate start tag.
        :param end_tag:     String. Encapsulate end tag.
        """
        # Text comma filtered:
        #   Remove all commas in it's usual use from text.
        text_comma_filtered = re.sub(r'\,(\s\p{L})',
                                     r'\1',
                                     text)
        # Sentenced text:
        #   Find all punctuation symbols followed by N spaces and a Capital letter.
        #   Replace by </s>\n<s>[FoundCapitalLetter]. Note that punctuation symbol is dropped.
        sentenced_text = re.sub(r'[\p{P}][\p{P}|\s]*(\p{Lu})',
                                r' {end} {start} \1'.format(end=end_tag, start=start_tag),
                                text_comma_filtered)
        # Add tags at beginning and end of file respectively
        sentenced_text_lower = start_tag + sentenced_text.lower() + end_tag

        tokenized_text = self.tokenize(sentenced_text_lower)
        self.process_text(tokenized_text)

    def process_text(self, tok_text):
        self.increase_corpus_size(len(tok_text))
        ngram = self.get_ngram()
        _count_unigrams = count_unigrams(tok_text)
        for key, key_freq in _count_unigrams.items():
            _count = {key: {'freq': key_freq, 'prob': key_freq / self.get_corpus_size()}}
            self.processed_by_ngrams.update(_count)
        if ngram == self.BIGRAM:
            _count_bigrams = count_bigrams(tok_text)
            for key, key_freq in _count_bigrams.items():
                _freq_i = self.processed_by_ngrams[key[0]]['freq']
                _count = {key: {'freq_tuple': key_freq, 'freq_i': _freq_i, 'prob': key_freq / _freq_i}}
                self.processed_by_ngrams.update(_count)
        elif ngram == self.UNIGRAM:
            pass
        else:
            raise Exception("Unknown counting type.")

    def prob_sentence(self, sentence):
        ngram = self.get_ngram()
        to_print = ["-" * 100, ngram, "=" * 100]
        print("\n".join(to_print))
        prob = 1
        if ngram == self.UNIGRAM:
            print("Word\t\tFreq\t\tSize\t\tProb")
            print("=" * 100)
            for word in sentence:
                word_prob = self.processed_by_ngrams[word]['prob']
                print("{word}\t\t{count_word}\t\t{size}\t\t{prob}"
                      .format(word=word,
                              count_word=self.processed_by_ngrams[word]['freq'],
                              size=self.get_corpus_size(),
                              prob=self.processed_by_ngrams[word]['prob']))
                prob = prob*word_prob
        elif ngram == self.BIGRAM:
            print("Word0\t\tWord1\t\tCountTuple\tCount_1\t\tP(wi+1|wi)")
            print("=" * 100)
            for tuple in self.coliding_tuples(sentence):
                if tuple in self.processed_by_ngrams:
                    freq_tuple = self.processed_by_ngrams[tuple]['freq_tuple']
                    tuple_prob = self.processed_by_ngrams[tuple]['prob']
                    tuple_prob_show = tuple_prob
                else:
                    freq_tuple = 0
                    tuple_prob = self.processed_by_ngrams[tuple[1]]['prob']
                    tuple_prob_show = "*Backoff: {}".format(self.processed_by_ngrams[tuple[1]]['prob'])
                freq_i = self.processed_by_ngrams[tuple[0]]['freq']
                print("{tuple0}\t\t{tuple1}\t\t{count_tuple}\t\t{count0}\t\t{prob_tuple}"
                      .format(tuple0=tuple[0],
                              tuple1=tuple[1],
                              count_tuple=freq_tuple,
                              count0=freq_i,
                              prob_tuple=tuple_prob_show))
                prob = prob * tuple_prob
        return prob

    @staticmethod
    def geometric_mean_prob(prob, N):
        return prob**(1/N)

    @staticmethod
    def entropy_rate(prob, N):
        return - math.log(prob, 2)/N

    @staticmethod
    def perplexity(entropy):
        return 2**entropy


if __name__ == '__main__':
    file = open("Selma.txt", 'r')
    file_txt = file.read()
    file.close()

    #sentence_unigram = ["det", "var", "en", "gång", "en", "katt", "som", "hette", "nils", "</s>"]
    #sentence_bigram = ["<s>", "det", "var", "en", "gång", "en", "katt", "som", "hette", "nils", "</s>"]
    sentence_unigram = ["han", "kunde", "inte", "förstå", "att", "de", "blevo", "så", "glada", "åt", "honom", "sådan", "som", "han", "var", "</s>"]
    sentence_bigram = ["<s>", "han", "kunde", "inte", "förstå", "att", "de", "blevo", "så", "glada",  "åt", "honom", "sådan", "som", "han", "var", "</s>"]

    model_unigram = LanguageModels(file_txt, LanguageModels.UNIGRAM)
    prob_unigram = model_unigram.prob_sentence(sentence_unigram)
    geom_mean_unigram = model_unigram.geometric_mean_prob(prob_unigram, len(sentence_unigram))
    entropy_unigram = model_unigram.entropy_rate(prob_unigram, len(sentence_unigram))
    perplexity_unigram = model_unigram.perplexity(entropy_unigram)
    to_print = [
        "=" * 100,
        "Prob. Unigram: {}".format(prob_unigram),
        "Geometric mean prob.: {}".format(geom_mean_unigram),
        "Entropy rate: {}".format(entropy_unigram),
        "Perplexity: {}".format(perplexity_unigram),
        "-" * 100,
        "\n"
    ]
    print("\n".join(to_print))
    model_bigram = LanguageModels(file_txt, LanguageModels.BIGRAM)
    prob_bigram = model_bigram.prob_sentence(sentence_bigram)
    geom_mean_bigram = model_bigram.geometric_mean_prob(prob_bigram, len(sentence_bigram)-1)
    entropy_bigram = model_bigram.entropy_rate(prob_bigram, len(sentence_bigram)-1)
    perplexity_bigram = model_bigram.perplexity(entropy_bigram)
    to_print = [
        "=" * 100,
        "Prob. Bigram: {}".format(prob_bigram),
        "Geometric mean prob.: {}".format(geom_mean_bigram),
        "Entropy rate: {}".format(entropy_bigram),
        "Perplexity: {}".format(perplexity_bigram),
        "-" * 100
    ]
    print("\n".join(to_print))


