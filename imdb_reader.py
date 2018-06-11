import os
import pickle
import numpy as np
from bs4 import BeautifulSoup

train_path = "data/imdb/train/"
test_path = "data/imdb/train/"

save_path = "data/imdb/imdb.pcl"

label_dic = {"neg":0, "pos":1}

def get_imdb():

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    if os.path.exists(save_path):
        print("loading imdb data from pickle...")
        with open(save_path, mode='rb') as f:
            data = pickle.load(f)

        train_data, train_labels, test_data, test_labels = data
        return train_data, train_labels, test_data, test_labels
    # read train files
    if os.path.exists(train_path):
        for key, val in label_dic.items():
            path = os.path.join(train_path, key)
            if os.path.exists(path):
                data = read_reviews(path)
                l = len(data)
                labels = np.full((l,), val)
                train_data.extend(data)
                train_labels.extend(labels)
            else:
                raise FileNotFoundError
    else:
        raise FileNotFoundError

    # read test files
    if os.path.exists(test_path):
        for key, val in label_dic.items():
            path = os.path.join(test_path, key)
            if os.path.exists(path):
                data = read_reviews(path)
                l = len(data)
                labels = np.full((l,), val)
                test_data.extend(data)
                test_labels.extend(labels)
            else:
                raise FileNotFoundError
    else:
        raise FileNotFoundError

    assert len(train_data)==len(train_labels) == len(test_data) == len(test_labels)

    with open(save_path, mode='wb') as f:
        pickle.dump([train_data, train_labels, test_data, test_labels], f)
    return train_data, train_labels, test_data, test_labels

def read_reviews(path):
    review_files = [os.path.join(path, f)
                           for f in os.listdir(path)
                           if f.endswith(".txt")]

    reviews = []
    for file in review_files:
        review = open(file, mode='r').read()
        bs = BeautifulSoup(review, "html.parser")
        lower_case = bs.get_text().lower()
        sentences = lower_case.split('. ')
        reviews.append(sentences)
    return reviews
