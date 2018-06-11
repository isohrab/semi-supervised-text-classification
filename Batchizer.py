import os
import wikiloader as wl
import Tokenizer as tk
import numpy as np
import threading
import time

np.random.seed(1)

class Batchizer(object):
    """
    This class will convert a raw stream of text (generator) and produce X, Y based `num_batch`

    """
    def __init__(self, batch_size, tokenizer_max_words=10000, pad_len=None, validation_ratio=0.2, data_loc='data/', retokenize=False):
        """

        :param batch_size: Batch size
        :param num_backet: the buckets will fill up with row data, and then we choose randomly from each bucket
        :param bucket_size: number of sample in each bucket, default is batch_size
        :param data_loc: the location of row data, we use it to load wikipedia files
        :param val_set: this persent of list files will hold for validation
        :param pad_len: None=Dynamic padding, else will be passed to tokenizer to pad sentences to equal length
        """
        self.batch_size = batch_size
        self.pad_len = pad_len

        # because of computation limit, we have to fix 5 files.
#         # load all wikipedia files
#         self.wiki_files = [os.path.join(data_loc, f)
#                            for f in os.listdir(data_loc)
#                            if f.endswith(".txt")]
        # self.wiki_files = sorted(self.wiki_files)
        # temporary solution to shortened the data
#         self.wiki_files = self.wiki_files[0:5]

        # assign wiki_files to train_files and then remove val_files from train_files
        self.train_files = ["data/wiki_sentences2.txt",
                           "data/wiki_sentences20.txt",
                           "data/wiki_sentences32.txt",
                           "data/wiki_sentences140.txt"]

        # choose validation files randomly
#         val_size = round(validation_ratio * len(self.wiki_files))
        self.val_files = ["data/wiki_sentences98.txt"]
#         for i in range(val_size):
#             k = np.random.randint(0, len(self.wiki_files))
#             # remove k-th filename from train_files list and append it to validation files list
#             self.val_files.append(self.train_files.pop(k))

        self.train_data = wl.Wikiloader(files=self.train_files, title="train")
        self.train_len = self.train_data.__len__()

        self.val_data = wl.Wikiloader(files=self.val_files, title="test")
        self.val_len = self.val_data.__len__()

        # initial Tokenizer
        self.tokenizer = tk.Tokenizer(max_words=tokenizer_max_words, retokenize=retokenize)
        # fit tokenizer on train files
        if len(self.tokenizer.word2index) == 0:
            self.tokenizer.fit_on_texts(self.train_data)

    def next_train_batch(self):
        """
        in every call, it will read `batch_size` of data from different buckets
        :return: return `batch_size` of processed data which are ready for Tensorflow
        """
        texts = []
        for line in self.train_data:
            if len(texts) == self.batch_size:
                enc_in, enc_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_eos= True)
                dec_in, dec_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_sos= True)
                yield [enc_in, enc_len, dec_in, dec_len]
                texts = []
            else:
                texts.append(line)
        
        if len(texts)!=0:
            enc_in, enc_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_eos= True)
            dec_in, dec_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_sos= True)
            yield [enc_in, enc_len, dec_in, dec_len]
        return

    def next_val_batch(self):
        """
        in every call, it will read `batch_size` of data from different buckets
        :return: return `batch_size` of processed data which are ready for Tensorflow
        """
        texts = []
        for line in self.val_data:
            if len(texts) == self.batch_size:
                enc_in, enc_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_eos= True)
                dec_in, dec_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_sos= True)
                yield [enc_in, enc_len, dec_in, dec_len]
                texts = []
            else:
                texts.append(line)

        enc_in, enc_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_eos= True)
        dec_in, dec_len = self.tokenizer.text2idx(texts, pad_length=self.pad_len, add_sos= True)
        yield [enc_in, enc_len, dec_in, dec_len]
        return





