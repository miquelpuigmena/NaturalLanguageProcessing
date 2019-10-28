import transition

SIZE_1 = 6
SIZE_2 = 10
SIZE_3 = 14


def extract_mode_1(stack, queue, graph, feature_names, sentence=None):
    features = list()
    if stack:
        stack0 = stack[0]
        features.extend([stack0.get('postag'), stack0.get('form')])
    else:
        features.extend(['nil', 'nil'])
    if queue:
        queue0 = queue[0]
        features.extend([queue0.get('postag'), queue0.get('form')])
    else:
        features.extend(['nil', 'nil'])

    features.append(transition.can_reduce(stack, graph))
    features.append(transition.can_leftarc(stack, graph))
    return dict(zip(feature_names.get('mode1'), features))


def extract_mode_2(stack, queue, graph, feature_names, sentence=None):
    extracted_features = dict()
    dict_mode_1 = extract_mode_1(stack, queue, graph, feature_names)
    extracted_features.update(dict_mode_1)
    features_mode_2 = list()
    if len(stack) > 1:
        stack1 = stack[1]
        features_mode_2.extend([stack1.get('postag'), stack1.get('form')])
    else:
        features_mode_2.extend(['nil', 'nil'])
    if len(queue) > 1:
        queue1 = queue[1]
        features_mode_2.extend([queue1.get('postag'), queue1.get('form')])
    else:
        features_mode_2.extend(['nil', 'nil'])
    extracted_features.update(dict(zip(feature_names.get('mode2'), features_mode_2)))
    return extracted_features


def extract_mode_3(stack, queue, graph, feature_names, sentence=None):
    extracted_features = dict()
    dict_mode_2 = extract_mode_2(stack, queue, graph, feature_names)
    extracted_features.update(dict_mode_2)
    features_mode_3 = list(['nil']*4)
    if stack:
        id0 = int(stack[0].get('id'))
        for pos, word in enumerate(sentence):
            if word.get('id') == str(id0-1):
                features_mode_3[:2] = [word.get('postag'), word.get('form')]
            elif word.get('id') == str(id0+1):
                features_mode_3[2:] = [word.get('postag'), word.get('form')]
    extracted_features.update(dict(zip(feature_names.get('mode3'), features_mode_3)))
    return extracted_features