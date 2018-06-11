# source from: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
import os
import re

def get_data(path, label_dic=None):
    '''
    Load text files from 20newsgroups folder and return data with labels
    :param path: path to 20newsgroups folder e.g '20_newsgroup/train'
    :return: docs, labels and labels_index list
    '''
    # list of text news
    texts = []
    
    dig_token = "dgt"
    # pattern to find the digit numbers
    dig_pattern = r'\b[0-9]*\.*[0-9]+\b'
    # list of cleaned texts
    docs = []
    #dictionary mapping label name to numeric id
    label_index = {}

    #list of label ids
    labels = []

    # help from: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isdir(p):
            if label_dic == None:
                label_id = len(label_index)
                label_index[name] = label_id
            for fname in sorted(os.listdir(p)):
                if fname.isdigit():
                    fpath = os.path.join(p, fname)
                    with open(fpath, mode='r', encoding='latin-1') as f:
                        t = f.read()
                        t = re.sub(dig_pattern, dig_token, t)
                        texts.append(t)
                    if label_dic == None:
                        labels.append(label_id)
                    else:
                        labels.append(label_dic[name])


    # clean texts to Docs[ sentences] ]
    for text in texts:
        # split to paragraph
        t = text.split('\n\n')

        sentences = []

        for p in t:

            # replace newline to space
            p = p.replace('\n', ' ')

            # convert paragraph to sentences
            sents = p.split('.  ')

            # remove all whitespace inside the sentences
            for s in sents:
                s = s.strip()
                s = ' '.join(s.split())
                if len(s) != 0:
                    # add sentences to paragraph
                    sentences.append(s)
        docs.append(sentences)

    return docs, labels, label_index

