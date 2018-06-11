class HyperParameters(object):
    START_TOKEN = 0
    END_TOKEN = 0
    VOCAB_LEN = 0
    MAX_WORD = 40
    TOKENIZER_MAX_WORDS = 100000
    MAX_GRADIANT_NORM = 5.0
    EMB_DIM = 300
    LEARNING_RATE = 0.001
    SEED = 47
    AE_LSTM_UNITS = 256
    KEEP_PROB = 0.95
    DECAY_RATE = 0.96
    GLOVE_PATH = 'data/word_embeddings/glove.42B.300d.txt'
    FASTTEXT_PATH = 'data/word_embeddings/wiki-news-300d-1M.vec'
    LEXVEC_PATH = 'data/word_embeddings/lexvec.enwiki+newscrawl.300d.W.pos.vectors'

class AEncoderHP(HyperParameters):
    BATCH_SIZE = 32
    MODEL_DIR = "model/AE/256/"
    SUMMARY_DIR = "model/AE/summary/256/"
    MODEL_NAME = "wiki.ckpt"
    N_EPOCHS = 10
    DISPLAY_EVERY = 200
    SAVE_EVERY= 2000
    VALIDATE_EVERY = 2000
    DECAY_STEP = 10000
    
class ImdbHP(HyperParameters):
    NAME = 'imdb'
    LSTM_UNITS = 64
    BATCH_SIZE = 128
    MAX_SENT = 10
    MODEL_DIR = "model/IMDB/"
    SUMMARY_DIR = "model/IMDB/summary/"
    MODEL_NAME = "imdb.ckpt"
    N_EPOCHS = 20
    DECAY_STEP = 500
    DENSE_REDUCER = 16
    DENSE_1 = 256
    DENSE_2 = 64
    N_LABELS = 2
    
    
class NewsGroupsHP(HyperParameters):
    NAME = '20newsgroups'
    LSTM_UNITS = 256
    BATCH_SIZE = 128
    MAX_SENT = 30
    MODEL_DIR = "model/20NG/"
    SUMMARY_DIR = "model/20NG/summary/"
    MODEL_NAME = "20NG.ckpt"
    N_EPOCHS = 50
    DECAY_STEP = 500
    DENSE_REDUCER = 32
    DENSE_1 = 512
    DENSE_2 = 128
    N_LABELS = 20
    
