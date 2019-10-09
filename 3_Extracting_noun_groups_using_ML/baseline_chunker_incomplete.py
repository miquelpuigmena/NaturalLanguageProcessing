"""
CoNLL 2000 file reader
"""
__author__ = "Pierre Nugues"


def persist_dict(file, data):
    f = open(file, "w+")
    for line in data:
        f.write("{word} {pos} {chunk} {predicted}\n"
                .format(word=line.get('form'),
                        pos=line.get('pos'),
                        chunk=line.get('chunk'),
                        predicted=line.get('predicted')))


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
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
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows]
        new_sentences.append(sentence)
    return new_sentences


def train(train_data):
    distribution = {}
    for sentence in train_data:
        for word in sentence:
            PoS = word.get('pos')
            chunk = word.get('chunk')
            if PoS not in distribution:
                distribution.update({PoS: {}})
            PoS_distribution = distribution.get(PoS)
            if chunk not in PoS_distribution:
                PoS_distribution.update({chunk: 1})
            else:
                PoS_distribution.update({chunk: PoS_distribution.get(chunk) + 1})
    return distribution


def highest_chunk_freq_by_pos(distribution):
    distribution_highest = {}
    for PoS in distribution.keys():
        chunks_of_pos = distribution.get(PoS)
        highest = max(chunks_of_pos, key=chunks_of_pos.get)
        distribution_highest.update({PoS: {highest: chunks_of_pos.get(highest)}})
    return distribution_highest


def predict(test_data, predictions):
    predicted_data = []
    for sentence in test_data:
        for word in sentence:
            prediction_by_pos = predictions.get(word.get('pos'))
            copy_word = word.copy()
            copy_word.update({'predicted': list(prediction_by_pos).pop()})
            predicted_data.append(copy_word)
    return predicted_data


if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'
    eval_file = 'eval.txt'
    column_names = ['form', 'pos', 'chunk']

    # TRAIN
    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names)
    trained_distribution = train(formatted_corpus)
    baseline_distribution = highest_chunk_freq_by_pos(trained_distribution)

    # TEST
    sentences_test = read_sentences(test_file)
    formatted_corpus_test = split_rows(sentences_test, column_names)
    formatted_corpus_predicted = predict(formatted_corpus_test, baseline_distribution)

    # PRESENT EVALUATION FILE
    persist_dict(eval_file, formatted_corpus_predicted)
