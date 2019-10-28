import transition
import conll
import time
from features import extract_mode_1, extract_mode_2, extract_mode_3
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import classification_report


def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'


def extract_features(formatted_corpus, mode, test_mode=False, vec=None, classifier=None):
    # EXTRACT FEATURES
    feature_names_1 = ['stack0_POS', 'stack0_word',
                       'queue0_POS', 'queue0_word',
                       'can-re', 'can-la']
    feature_names_2 = ['stack1_POS', 'stack1_word',
                       'queue1_POS', 'queue1_word']
    feature_names_3 = ['left_POS', 'left_word',
                       'right_POS', 'right_word']

    feature_names = {'mode1': feature_names_1, 'mode2': feature_names_2, 'mode3': feature_names_3}
    X = list()
    transitions = list()
    sent_cnt = 0
    for sentence in formatted_corpus:
        sent_cnt += 1
        # if sent_cnt % 1000 == 0:
        #    print(sent_cnt, 'sentences on', len(formatted_corpus), flush=True)
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        while queue:
            if mode == 3:
                X_row = extract_mode_3(stack, queue, graph, feature_names, sentence)
            elif mode == 2:
                X_row = extract_mode_2(stack, queue, graph, feature_names, sentence)
            elif mode == 1:
                X_row = extract_mode_1(stack, queue, graph, feature_names, sentence)
            if not test_mode:
                stack, queue, graph, trans = reference(stack, queue, graph)
            elif test_mode:
                X_row_vec = vec.transform(X_row)
                trans_nr = classifier.predict(X_row_vec)
                stack, queue, graph, trans = parse_ml(stack, queue, graph, trans_nr)
            X.append(X_row)
            transitions.append(trans)
        stack, graph = transition.empty_stack(stack, graph)
        # print('Equal graphs:', transition.equal_graphs(sentence, graph))

        # Poorman's projectivization to have well-formed graphs.
        if test_mode:
            for word in sentence:
                word['head'] = graph['heads'][word['id']]
                word['deprel'] = graph['deprels'][word['id']]
        # print(graph)
    for pos, e in enumerate(X[:6]):
        print("x = {}, y= {}".format(e, transitions[pos]))
    # print(X)
    # print(transitions)
    if test_mode:
        conll.save('out_{}_mode_{}.conll'.format("test", mode), formatted_corpus, column_names_2006)
    return X, transitions


def parse_ml(stack, queue, graph, trans):
    #right arc
    if stack and trans[:2] == 'ra' and transition.can_rightarc(stack):
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    #left arc
    if stack and trans[:2] == 'la' and transition.can_leftarc(stack, graph):
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'
    #reduce
    if stack and trans[:2] == 're' and transition.can_reduce(stack, graph):
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'
    #shift
    if stack and trans[:2] == 'sh':
        stack, queue, graph = transition.shift(stack, queue, graph)
        return stack, queue, graph, 'sh'
    #action not possible -> shift
    else:
        stack, queue, graph = transition.shift(stack, queue, graph)
        return stack, queue, graph, 'sh'


def train_model(X_dict, y, classifier, vec):
    start_time = time.clock()
    X = vec.fit_transform(X_dict)
    print("Training the model...")
    model = classifier.fit(X, y)
    # print(model)
    y_pred = classifier.predict(X)
    print(classification_report(y, y_pred))
    stop_time = time.clock()
    print("Training time:", (stop_time - start_time) / 60)


def test_model(X_test_dict, y_test_dict, classifier, vec):
    start_time = time.clock()
    X_test = vec.transform(X_test_dict)
    y_test_predicted = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test_dict, y_test_predicted)))
    stop_time = time.clock()
    print("Testing time:", (stop_time - start_time) / 60)


if __name__ == '__main__':
    train_file = 'swedish_talbanken05_train.conll'
    test_file = 'swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    #for i in range(1, 4):
    vec = DictVectorizer(sparse=True)
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear', verbose=1)
    # classifier = linear_model.Perceptron(penalty='l2')
    # classifier = tree.DecisionTreeClassifier()
    # classifier = linear_model.SGDClassifier(penalty='l2')
    sentences = conll.read_sentences(train_file)
    sentences_test = conll.read_sentences(test_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)
    formatted_corpus_test = conll.split_rows(sentences_test, column_names_2006_test)

    X, y = extract_features(formatted_corpus, mode=3)

    train_model(X, y, classifier, vec)

    X_test, y_test = extract_features(formatted_corpus_test, mode=3, test_mode=True, vec=vec, classifier=classifier)
    test_model(X_test, y_test, classifier, vec)





