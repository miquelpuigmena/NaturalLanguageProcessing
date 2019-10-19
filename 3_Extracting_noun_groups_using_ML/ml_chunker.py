"""
Machine learning chunker for CoNLL 2000
"""
__author__ = "Pierre Nugues"

import time
import conll_reader
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model, tree
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


def extract_features(sentences, w_size, feature_names):
    """
    Builds X matrix and y vector
    X is a list of dictionaries and y is a list
    :param sentences:
    :param w_size:
    :return:
    """
    X_l = []
    y_l = []
    for sentence in sentences:
        X, y = extract_features_sent(sentence, w_size, feature_names)
        X_l.extend(X)
        y_l.extend(y)
    return X_l, y_l


def extract_features_sent(sentence, w_size, feature_names):
    """
    Extract the features from one sentence
    returns X and y, where X is a list of dictionaries and
    y is a list of symbols
    :param sentence: string containing the CoNLL structure of a sentence
    :param w_size:
    :return:
    """

    # We pad the sentence to extract the context window more easily
    start = "BOS BOS BOS\n"
    end = "\nEOS EOS EOS"
    start *= w_size
    end *= w_size
    sentence = start + sentence
    sentence += end

    # Each sentence is a list of rows
    sentence = sentence.splitlines()
    padded_sentence = list()
    for line in sentence:
        line = line.split()
        padded_sentence.append(line)

    # We extract the features and the classes
    # X contains is a list of features, where each feature vector is a dictionary
    # y is the list of classes
    X = list()
    y = list()
    for i in range(len(padded_sentence) - 2 * w_size):
        # x is a row of X
        x = list()
        # The words in lower case
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j][0].lower())
        # The POS
        for j in range(2 * w_size + 1):
            x.append(padded_sentence[i + j][1])
        # The chunks (Up to the word)
        for j in range(w_size):
            x.append(padded_sentence[i + j][2])

        # We represent the feature vector as a dictionary
        X.append(dict(zip(feature_names, x)))
        # The classes are stored in a list
        y.append(padded_sentence[i + w_size][2])
    return X, y


def predict(test_sentences, feature_names, f_out):
    y_test = list()
    y_test_predicted = list()
    for test_sentence in test_sentences:
        X_test_dict, y_test_local = extract_features_sent(test_sentence, w_size, feature_names)
        chunk_n2 = 'BOS'
        chunk_n1 = 'BOS'
        y_test_predicted_local = []
        for iteration in range(len(X_test_dict)):
            # Update previous chunk predictions
            X_test_dict[iteration]['chunk_n1'] = chunk_n1
            X_test_dict[iteration]['chunk_n2'] = chunk_n2
            # Vectorize the test sentence and one hot encoding
            X_test = vec.transform(X_test_dict[iteration])
            # Predicts the chunks and returns numbers
            y_iteration_predict = classifier.predict(X_test)[0]
            y_test_predicted_local.append(y_iteration_predict)
            # Persist chunk predictions
            chunk_n1 = y_iteration_predict
            chunk_n2 = chunk_n1

        # Appends the predicted chunks as a last column and saves the rows
        rows = test_sentence.splitlines()
        rows = [rows[i] + ' ' + y_test_predicted_local[i] for i in range(len(rows))]
        for row in rows:
            f_out.write(row + '\n')
        f_out.write('\n')
        # Persist predictions and real results for each sentence
        y_test_predicted.extend(y_test_predicted_local)
        y_test.extend(y_test_local)
    f_out.close()
    return y_test_predicted, y_test


if __name__ == '__main__':

    start_time = time.clock()
    train_corpus = 'train.txt'
    test_corpus = 'test.txt'
    w_size = 2  # The size of the context window to the left and right of the word
    feature_names = ['word_n2', 'word_n1', 'word', 'word_p1', 'word_p2',
                     'pos_n2', 'pos_n1', 'pos', 'pos_p1', 'pos_p2',
                     'chunk_n2', 'chunk_n1']

    train_sentences = conll_reader.read_sentences(train_corpus)

    print("Extracting the features...")
    X_dict, y = extract_features(train_sentences, w_size, feature_names)

    print("Encoding the features...")
    # Vectorize the feature matrix and carry out a one-hot encoding
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    # The statement below will swallow a considerable memory
    # X = vec.fit_transform(X_dict).toarray()
    # print(vec.get_feature_names())

    training_start_time = time.clock()
    print("Training the model...")
    classifier = linear_model.LogisticRegression(multi_class='auto', solver='lbfgs')
    #classifier = linear_model.Perceptron(penalty='l2')
    #classifier = tree.DecisionTreeClassifier()
    #classifier = linear_model.SGDClassifier(penalty='l2')

    model = classifier.fit(X, y)
    print(model)

    # Predicting the test set
    test_start_time = time.clock()
    # We apply the model to the test set
    test_sentences = list(conll_reader.read_sentences(test_corpus))

    """
    # Here we carry out a chunk tag prediction and we report the per tag error
    # This is done for the whole corpus without regard for the sentence structure
    print("Predicting the chunks in the test set...")
    X_test_dict, y_test = extract_features(test_sentences, w_size, feature_names)
    # Vectorize the test set and one-hot encoding
    X_test = vec.transform(X_test_dict)  # Possible to add: .toarray()
    y_test_predicted = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_test_predicted)))
    """
    print("Predicting the test set...")
    f_out = open('out', 'w')
    y_test_predicted, y_test = predict(test_sentences, feature_names, f_out)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_test_predicted)))

    end_time = time.clock()
    print("Training time:", (test_start_time - training_start_time) / 60)
    print("Test time:", (end_time - test_start_time) / 60)