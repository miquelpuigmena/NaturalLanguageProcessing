class Indexer:
    # ...
    def populate_master_index_by_file(self, file_name):
        file = open(SELMA_PATH + file_name, encoding='UTF8')
                    .read()
                    .lower()
        words_in_file = re.finditer('\p{L}+', file)
        for iteration, word_appearances in enumerate(words_in_file):
            self.MI.persist_dict(file_name, 
                                 word_appearances.group(), 
                                 word_appearances.start())
class MasterIndex:
    # ...
    def persist_dict(self, file_name, word, start_position):
        if word in self.myDict:
            if file_name in self.myDict[word]:
                self.myDict[word][file_name].append(start_position)
            else:
                self.myDict[word].update({file_name: [start_position]})
        else:
            self.myDict[word] = {file_name: [start_position]}
