import os
import pickle
import regex as re
import numpy as np

from math import log10, sqrt

SELMA_PATH = './Selma/'
INDEX_FILE = './master_index.json'


class CosineSimilarity:
    @staticmethod
    def solve(array_data_1, array_data_2):
        np1 = np.array(list(array_data_1.values()))
        np2 = np.array(list(array_data_2.values()))
        numerator = np.sum(np1*np2)
        denominator = sqrt(np.sum(np1*np1))*sqrt(np.sum(np2*np2))
        return numerator / denominator


class Tfidf:
    def __init__(self):
        self.files_representation = {}
        self.files_length = {}

    def get_files_representation(self):
        return self.files_representation

    def get_files_length(self):
        return self.files_length

    def set_file_length(self, file_name, length):
        self.files_length.update({file_name: length})

    def set_word_metric_by_file(self, file_name, word, metric):
        if file_name not in self.files_representation:
            self.files_representation[file_name] = {}
        self.files_representation[file_name].update({word: metric})

    def TF_by_word_repetitions(self, file_name, word_repetitions):
        return word_repetitions / self.files_length[file_name]

    def IDF_by_word_appearances(self, document_appearances):
        return log10(len(self.files_length) / document_appearances)

    def metric_by_word_info(self, file_name, word_repetitions, document_appearances):
        tf = self.TF_by_word_repetitions(file_name, word_repetitions)
        idf = self.IDF_by_word_appearances(document_appearances)
        return tf * idf


class MasterIndex:
    def __init__(self):
        self.myDict = {}

    def persist_dict(self, file_name, word, start_position):
        if word in self.myDict:
            if file_name in self.myDict[word]:
                self.myDict[word][file_name].append(start_position)
            else:
                self.myDict[word].update({file_name: [start_position]})
        else:
            self.myDict[word] = {file_name: [start_position]}

    def dump_index_to_file(self, file_name):
        pickle.dump(self.myDict, open(file_name, "wb"))

    def set_metrics(self, TFIDF):
        for file_name in TFIDF.get_files_length():
            for word, word_overview_by_document in self.myDict.items():
                word_repetitions_in_document = 0
                if file_name in word_overview_by_document:
                    word_repetitions_in_document = len(word_overview_by_document[file_name])
                metric = TFIDF.metric_by_word_info(file_name=file_name,
                                                   word_repetitions=word_repetitions_in_document,
                                                   document_appearances=len(word_overview_by_document))
                TFIDF.set_word_metric_by_file(file_name, word, metric)


class Indexer:
    def __init__(self):
        self.MI = MasterIndex()
        self.TFIDF = Tfidf()

    @staticmethod
    def get_files(dir, suffix):
        """
        Returns all the files in a folder ending with suffix
        :param dir:
        :param suffix:
        :return: the list of file names
        """
        files = []
        for file in os.listdir(dir):
            if file.endswith(suffix):
                files.append(file)
        return files

    @staticmethod
    def build_symetric_matrix(list_of_titles):
        matrix = {}
        for deep1title in list_of_titles:
            matrix[deep1title] = {}
            aux_list = list_of_titles.copy()
            aux_list.remove(deep1title)
            for deep2title in aux_list:
                matrix[deep1title].update({deep2title: None})
        return matrix

    @staticmethod
    def get_max_from_matrix(dict_matrix):
        max = {'confidence': 0, 'coordinates': (None, None)}
        for key_row, row in  dict_matrix.items():
            for key_col, col in row.items():
                if max['confidence'] < col:
                    max.update({'coordinates': (key_row, key_col)})
                    max.update({'confidence': col})
        return max

    def populate_master_index_by_file(self, file_name):
        """
        Extracts all words from a file and persists them in MasterIndex Dictionary
        :param file_name:
        """
        file = open(SELMA_PATH + file_name, encoding='UTF8').read().lower()
        words_in_file = re.finditer('\p{L}+', file)
        for iteration, word_appearances in enumerate(words_in_file):
            self.MI.persist_dict(file_name, word_appearances.group(), word_appearances.start())
        self.TFIDF.set_file_length(file_name, iteration)

    def compute_corpus(self, file_names):
        for file_name in file_names:
            print("Computing file: {} ... ".format(file_name))
            self.populate_master_index_by_file(file_name=file_name)
            print("File {} successfully parsed :) ".format(file_name))
        self.MI.dump_index_to_file(INDEX_FILE)

    def compare_corpus(self, file_names):
        self.MI.set_metrics(self.TFIDF)
        aux_file_names = file_names.copy()
        similarity_matrix = self.build_symetric_matrix(file_names)
        tfidf_computed = self.TFIDF.get_files_representation().copy()
        for principal in file_names:
            aux_file_names.remove(principal)
            for secondary in aux_file_names:
                similarity = CosineSimilarity.solve(tfidf_computed[principal],
                                                    tfidf_computed[secondary])
                similarity_matrix[principal][secondary] = similarity
                similarity_matrix[secondary][principal] = similarity
        print(similarity_matrix)
        return similarity_matrix

    def run(self):
        file_names = self.get_files(SELMA_PATH, '.txt')
        self.compute_corpus(file_names)
        similarity_matrix = self.compare_corpus(file_names)
        max_confidence_similar_novels = self.get_max_from_matrix(similarity_matrix)

        print("="*100)
        print("Indexer found most similarity between {novels} novels with a confidence of {confidence}"
              .format(novels=max_confidence_similar_novels['coordinates'],
                      confidence=max_confidence_similar_novels['confidence']))
        print("="*100)


if __name__ == "__main__":
    Indexer().run()




