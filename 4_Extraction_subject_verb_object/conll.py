import os
import regex as re

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    f = f.lower()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split('\t'))) for row in rows if row[0] != '#' and not isRange(row)]
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def get_most_freq(dict_db, N):
    return sorted(dict_db.items(), key=lambda x: x[1], reverse=True)[:N]


def find_verb_relation_by(sentence, center_pos, TAG):
    for word in sentence:
        if word.get('deprel') == TAG:
            if word.get('head') == center_pos:
                break
            #deep_2_head = sentence[int(word.get('head'))].get('head')
            #if deep_2_head == center_pos:
            #    break
    else:
        return None
    return word


def isRange(id):
    pattern = re.compile("^\d+-\d+")
    return pattern.match(id)


def persist_key_to(dict_db, key):
    freq = 1
    if key in dict_db:
        freq = dict_db.get(key) + 1
    dict_db.update({key: freq})


def tupleize_by_SV(dict_db, corpus, mode='conll'):
    if mode == 'conll':
        TAG_SUBJECT = 'SS'.lower()
    elif mode == 'conllu':
        TAG_SUBJECT = 'nsubj'.lower()
    else:
        print("Unknown mode")
        exit(1)
    for sentence in corpus:
        for word in sentence:
            if word.get('deprel') == TAG_SUBJECT:
                form = word.get('form')
                verb = sentence[int(word.get('head'))].get('form')
                persist_key_to(dict_db, (form, verb))


def tupleize_by_SVO(dict_db, corpus, mode='conll'):
    if(mode == 'conll'):
        TAG_SUBJECT = 'SS'.lower()
        TAG_OBJECT = 'OO'.lower()
    elif(mode == 'conllu'):
        TAG_SUBJECT = 'nsubj'.lower()
        TAG_OBJECT = 'obj'.lower()
    else:
        print("Unknown mode")
        exit(1)
    for sentence in corpus:
        for word in sentence:
            if word.get('deprel') == TAG_SUBJECT:
                verb_pos = word.get('head')
                found_object = find_verb_relation_by(sentence.copy(), verb_pos, TAG_OBJECT)
                if found_object != None:
                    form_subject = word.get('form')
                    verb = sentence[int(word.get('head'))].get('form')
                    form_object = found_object.get('form')
                    persist_key_to(dict_db, (form_subject, verb, form_object))


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()


if __name__ == '__main__':
    print("*"*25 + 'CONLL' + "*"*25)
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test.conll'

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)

    dict_db = dict()
    tupleize_by_SV(dict_db, formatted_corpus)
    print("Total tuples found: {}".format(sum(dict_db.values())))
    print(get_most_freq(dict_db, N=5))

    dict_db = dict()
    tupleize_by_SVO(dict_db, formatted_corpus)
    print("Total tuples found: {}".format(sum(dict_db.values())))
    print(get_most_freq(dict_db, N=5))

    print("*"*25 + 'CONLLU' + "*"*25)
    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']
    files = get_files('corpus_universal',
                      'train.conllu')
    for train_file in files:
        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_u)
        print("-"*25 + train_file + "-"*25, len(formatted_corpus))

        dict_db = dict()
        tupleize_by_SV(dict_db, formatted_corpus, mode='conllu')
        print("Total tuples found: {}".format(sum(dict_db.values())))
        print(get_most_freq(dict_db, N=3))

        dict_db = dict()
        tupleize_by_SVO(dict_db, formatted_corpus, mode='conllu')
        print("Total tuples found: {}".format(sum(dict_db.values())))
        print(get_most_freq(dict_db, N=3))

