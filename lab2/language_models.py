import regex as re
from mutual_info import count_bigrams
from mutual_info import count_unigrams


def decorate_title(f):
    def wrapper(a, b, ngram):
        print(ngram)
        print("=" * 100)
        print("Word\tFreq\tSize\tProb")
        f(a, b, ngram)
    return wrapper


class LanguageModels:
    DEF_start_tag = "<s>"
    DEF_end_tag = "</s>"
    UNIGRAM = "Unigram Model"
    BIGRAM = "Bigram Model"

    def __init__(self, file_text):
        self.normalized_text = self.normalize_corpus(file_text)
        self.processed_by_ngrams = {}

    @staticmethod
    def tokenize(text):
        """uses the nonletters to break the text into words
        returns a list of words"""
        erased_punct = re.sub(r'[^\P{P}\/]+', '', text)
        multi_line_text = re.sub(r'(\p{L}+|\<s\>|\<\/s\>)', r'\1\n', erased_punct)
        words_intro_separated = re.sub(r'\n+', '\n', multi_line_text)
        words = words_intro_separated.split()
        return words

    def normalize_corpus(self, text, start_tag=DEF_start_tag, end_tag=DEF_end_tag):
        """
        Splits text by sentences and encapsulates with <s>sentence</s>.

        :param start_tag:   String. Encapsulate start tag.
        :param end_tag:     String. Encapsulate end tag.
        :return:            String. Text formatted in the form of: EmptyString.join(List(<s>sentence</s>)).
        """
        # Text  no enter:
        #   Removed all \n from text
        text_no_enter = re.sub(r'\n', r'', text)
        # Sentenced text:
        #   Find all punctuation symbols followed by N spaces and a Capital letter.
        #   Replace by </s>\n<s>[FoundCapitalLetter]. Note that punctuation symbol is dropped.
        sentenced_text = re.sub(r'(\.|\?|\!)(\s)+(\p{Lu})',
                                r'{end}{start}\3'.format(end=end_tag, start=start_tag),
                                text_no_enter)
        # Return:
        #   Add Start and End tags on start of file and end of file respectively.
        return start_tag + sentenced_text.lower() + end_tag

    @decorate_title
    def process_text(self, text, ngram):
        self.processed_by_ngrams.clear()
        tok_text = self.tokenize(text)
        size=len(tok_text)
        if ngram == self.UNIGRAM:
            count = count_unigrams(tok_text)
        elif ngram == self.BIGRAM:
            count = count_bigrams(tok_text)
        else:
            raise Exception("Unknown counting type.")
        for key, key_freq in count.items():
            self.processed_by_ngrams.update({key: {'freq': key_freq, 'size': size, 'prob': key_freq / size}})

    @staticmethod
    def coliding_tuples(list):
        """
        Contains pair of tuples containing coliding elements in priginal list.
        If original_list was length N, tuples_list will be length N-1 containing N-1 tuples.

        :param list: List containing elements.
        :return: List.
        """
        tuples_list = []
        for iteration, element in enumerate(list[:-1]):
            tuples_list.append((element, list[iteration+1]))
        return tuples_list

    def prob_sentence(self, sentence, ngram):
        self.process_text(self.normalized_text, ngram)
        prob = 1
        if ngram == self.UNIGRAM:
            for word in sentence:
                print("{word}\t{count_word}\t{size}\t{prob}"
                      .format(word=word,
                              count_word=self.processed_by_ngrams[word]['freq'],
                              size=self.processed_by_ngrams[word]['size'],
                              prob=self.processed_by_ngrams[word]['prob']))
                prob = prob*self.processed_by_ngrams[word]['prob']
        elif ngram == self.BIGRAM:
            for tuple in self.coliding_tuples(sentence):
                prob = prob * self.processed_by_ngrams[tuple]['prob']
        return prob


if __name__ == '__main__':
    file = open("Selma.txt", 'r')
    file_txt = file.read()
    file.close()
    model = LanguageModels(file_txt)
    print("*"*100)
    print("*"*100)
    sentence = ["det", "var", "en", "gång", "en", "katt", "som", "hette", "nils", "</s>"]
    prob_unigram = model.prob_sentence(sentence, LanguageModels.UNIGRAM)
    prob_bigram = model.prob_sentence(sentence, LanguageModels.BIGRAM)
    print(prob_unigram)
    print(prob_bigram)
