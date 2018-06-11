import os
import re
import numpy as np
from collections import OrderedDict

UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'

root_name = 'tokenizer/'
file_name = 'word2index.txt'  # the row number inidcate the index id


class Tokenizer(object):
    """
    get a list of text and extract unique words.
    this class inspired form Keras tokenizer. a lot of code snipped from there.
    """

    def __init__(self, max_words=20000, clean=True, digit_token=None, retokenize=False, unk=None, sos=None, eos=None,
                 lower=True, ):
        """TODO: need update
        Process corpus and get all most occurence `max_word` unique words. by default save word2index
        to `root_name` + `file_name` location.

        Choosing a good value for max_words is very important, because at the end of decoder, we should
        calculate `logit` over this value
        :param max_words: length of dictionary, based first most occurred words
        :param clean: if true, all ~!@#$%^&*()?><:"{}| will be removed from words,
        :param clean_digit: if provided, it will replace digits with token
        :param load_from_file: if true, and if the file is exict, it will be load
                            from file. otherwise process data
        :param unk_token: if a word not found in dict, put this token instead
        :param convert to lower case
        """
        self.max_words = max_words
        self.clean = clean
        self.digit_token = digit_token

        self.unk = unk if unk else UNK
        self.sos = sos if sos else SOS
        self.eos = eos if eos else EOS

        self.lower = lower

        self.word2index = {}
        self.index2word = {}
        self.word_counts = OrderedDict()

        if not retokenize:  # check out if we already processed the data

            save_loc = os.path.join(root_name, file_name)

            if os.path.exists(save_loc):
                # it means we already processed the data. load it
                print("A processed word2index file has been found in %s" % save_loc)
                with open(save_loc, mode='r') as f:
                    liner = 1
                    for l in f:
                        l = l.strip()  # remove \n
                        self.word2index[l] = liner
                        self.index2word[liner] = l
                        liner += 1

                # get back indexes for unk, sos, eos tokens
                self.unk_index = self.word2index[self.unk]
                self.start_index = self.word2index[self.sos]
                self.end_index = self.word2index[self.eos]
                self.max_words = len(self.word2index)
                print("%i tokens have been loaded sucessfully." % self.max_words)

    def fit_on_texts(self, texts):
        """
        Like Keras, use this method to create word2index file.
        for first time on a corpus, we should use it
        :param texts: a list of text to process
        :return: None
        """
        for t in texts:
            # first clean text
            if self.clean:
                t = self._clean_str(t)

            # to lower case
            if self.lower:
                t = t.lower()

            # split to words
            words = t.split()

            # add words to word_count list
            for w in words:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]

        # truncate sorted_voc to max_word other words will treated as unknown
        if len(sorted_voc) > self.max_words:
            sorted_voc = sorted_voc[0:self.max_words]

        # this is a magic from keras! :)
        self.word2index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        # add unk, sos, eos tokens to vocab
        self.unk_index = len(self.word2index) + 1
        self.word2index[self.unk] = self.unk_index
        self.start_index = len(self.word2index) + 1
        self.word2index[self.sos] = self.start_index
        self.end_index = len(self.word2index) + 1
        self.word2index[self.eos] = self.end_index

        # fill index2word dictionary
        self.index2word = {v: k for k, v in self.word2index.items()}

        # save to file for later use!
        # check if root directory is not exist
        if not os.path.exists(root_name):
            os.mkdir(root_name)

        save_loc = os.path.join(root_name, file_name)
        try:
            with open(save_loc, mode='w') as f:
                for i in range(len(self.index2word)):
                    s = self.index2word[i + 1] + "\n"
                    f.write(s)  # note: index begin from 1
            print("%i tokens has been saved to: \n %s \n" % (len(self.index2word), save_loc))
        except IOError as e:
            print("Error: Could not save tokens to:\n%s\n Error details: %s" % (save_loc, e))

    def text2idx(self, texts, pad_length=None, pad_value=0., clean=True, add_sos=None, add_eos=None):
        '''
        get a text, clean it if clean be true, then convert it a list of integer based
        word2index.
        :param text: input text
        :param pad_length: add pad_value (usually 0) to the input string if it is less than pad_length, otherwise
                        text will be truncated
        :param pad_value: the value of pad (default and unused)
        :param add_sos: if true, it will add a `sos` token at the end of sentence. it is good for decoder input
        :param add_eos: if it is true, it will add a `eos` toekn at the begining of sentence. It is good if text will
                        be used for encoder input and decoder output.
        :param clean: remove signs and whitespaces from text
        :return: a vector of length `pad_length` which each word replaced by its index value based vocabulary word2index
        '''
        if clean:
            texts = [self._clean_str(text) for text in texts]
        # TODO: improve hard code
        max_pad = 40
        if not pad_length:
            l = [len(t.split()) for t in texts]
            if len(l)==0:
                pad_length=max_pad
            else:
                pad_length = max(l)
            # stop large senteces
            if pad_length > max_pad:
                pad_length = max_pad
                
        if add_eos:
            pad_length += 1
        if add_sos:
            pad_length += 1

        vec_list = []
        len_list = []
        
        # call unknown idx just once
        unk_idx = self.word2index.get(self.unk)
        
        for text in texts:
            words = text.lower().split()
            if len(words) < 3:
                continue
            
            if add_sos:
                words.insert(0, self.sos)

            if add_eos:
                if len(words) >= pad_length:
                    words.insert(pad_length-1,self.eos)
                else:
                    words.append(self.eos)
                    
            len_list.append(min(len(words),pad_length))
                
            vec = np.zeros(pad_length, dtype=np.int32)
            
            for i, w in enumerate(words):
                if i == pad_length:
                    break
                k = self.word2index.get(w.lower())
                if k is None:
                    vec[i] = unk_idx
                else:
                    vec[i] = k
            vec_list.append(vec)
        return vec_list, len_list
                

    def idx2one_hot(self, idxs, pad_length):
        """
        get a list of integers and convert it to one_hot vector
        :param idxs: list of integers which returned from text2idx
        :param pad_length: is equal to timestep
        :return: a on_hot matrix
        """
        oh = np.zeros((pad_length, len(self.word2index)+1))
        oh[np.arange(pad_length), idxs] = 1

        return oh

    def idx2text(self, idxs):
        """
        get a list of indices and lookup it in index2word and convert it to text
        :param idxs: list of indices
        :return: text
        """
        text = []
        for i in idxs:
            if i == 0:
                continue
            text.append(self.index2word[i])

        return ' '.join(text)

    def _clean_str(self, string):
        """
            Tokenization/string cleaning for all text entry.
           Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
         """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        if self.digit_token is not None:
            string = re.sub(r"[0-9]", "", string)

        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()


