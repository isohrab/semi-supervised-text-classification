import numpy as np
import os

def load_embedding(word2idx, dim, file_path):
    
    v_len = len(word2idx)
    # word2idx is included with UNK, EOS, SOS
    weights = np.random.uniform(low=-0.05, high=0.05,size=(v_len, dim))
    # index 0 is for PAD
    weights= np.concatenate((np.zeros((1,dim)), weights))
    token_in_we = 0
    if not os.path.exists(file_path):
        raise FileNotFound
    with open(file_path, mode='r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            idx = word2idx.get(word)
            if idx is not None:
                coef = np.asarray(values[1:], dtype='float32')
                weights[idx] = coef
                token_in_we +=1
    
    no_we_words = v_len - token_in_we
    print(no_we_words, " tokens does not exist in the word embedding")
    
    return weights


def data2idx(data, HP, tokenizer):
    
    data_idxs = []
    sent_lengths = []
    docs_lengths = []
    for doc in data:
        doc_idx, sent_lens = tokenizer.text2idx(doc, pad_length=HP.MAX_WORD, add_eos=True)
        data_idxs.append(doc_idx)
        sent_lengths.append(sent_lens)
        docs_lengths.append(len(doc))
    
    return data_idxs, sent_lengths, docs_lengths

def batchizer(docs_idx, sent_lengths, docs_lengths, labels, batch_size, max_sent):

#     # shuffle data
    m = len(labels)

    per = np.arange(m)
    np.random.shuffle(per)
    docs_idx_shuffled = [docs_idx[index] for index in per]
    sent_lengths_shuffled = [sent_lengths[index] for index in per]
    docs_lengths_shuffled = [docs_lengths[index] for index in per]
    labels_shuffled = [labels[index] for index in per]

    n_batch = int(m / batch_size)

    for i in range(n_batch):
        # slice the list
        batch_docs_idx = docs_idx_shuffled[i*batch_size:(i+1)*batch_size]
        batch_sent_length = sent_lengths_shuffled[i*batch_size:(i+1)*batch_size]
        # pad sentences to max_sent length
        batch_sent_length = [k[:max_sent] + [0]*np.max(max_sent - len(k),0) for k in batch_sent_length]
        
        batch_docs_length = docs_lengths_shuffled[i*batch_size:(i+1)*batch_size]
        # trim docs_length greater than 20
        batch_docs_length = [min(k, max_sent) for k in batch_docs_length]
        batch_labels = labels_shuffled[i*batch_size:(i+1)*batch_size]
        batch_max_words = np.max(np.max(batch_sent_length))
        batch_x = np.zeros((batch_size, max_sent, batch_max_words), dtype=np.int32)
        for d, doc in enumerate(batch_docs_idx):
            for s, sent in enumerate(doc):
                if s == max_sent:
                    break
                batch_x[d,s, 0:len(sent)] = sent[:]
        yield batch_x, batch_sent_length, batch_docs_length, batch_labels