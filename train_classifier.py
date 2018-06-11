import os
import random
import numpy as np
import tensorflow as tf
import Tokenizer as tk
import imdb_reader as imdb
import ngreader as ng
from hyper_parameters import *
from Classifier import classifier
from Seq2SeqModel import Seq2SeqModel
from utils import *
import argparse

def get_encoder(tokenizer, word_embedding):
    HP = AEncoderHP()
    HP.START_TOKEN = tokenizer.start_index
    HP.END_TOKEN = tokenizer.end_index
    HP.VOCAB_LEN = len(tokenizer.word2index)+1
    HP.MODEL_DIR = HP.MODEL_DIR + word_embedding + "/"
    with tf.Session() as sess:
        model = Seq2SeqModel(HP, 'train')
        ckpt = tf.train.get_checkpoint_state(HP.MODEL_DIR)
        model.restore(sess, ckpt.model_checkpoint_path)

        # assign encoder weights
        encoder_kernel_tensor = sess.graph.get_tensor_by_name("encoder/rnn/basic_lstm_cell/kernel:0")
        encoder_bias_tensor = sess.graph.get_tensor_by_name("encoder/rnn/basic_lstm_cell/bias:0")
        print("following weights loaded successfully from encoder model...")
        print(encoder_kernel_tensor)
        print(encoder_bias_tensor)

        # get numpy array
        encoder_kernel = sess.run(encoder_kernel_tensor)
        encoder_bias = sess.run(encoder_bias_tensor)
    del model
    tf.reset_default_graph()
    return encoder_kernel, encoder_bias

        
def main(dataset, word_embedding, use_encoder, train_encoder):
    all_dataset = ["imdb", "20newsgroups"]
    dataset = dataset.lower()
    if not dataset in all_dataset:
        raise InvalidArgument
    
    all_word_embeddings = ["glove", "fasttext", "lexvec"]
    word_embedding = word_embedding.lower()
    if not word_embedding in all_word_embeddings:
        raise InvalidArgument
        
    # initial tokenizer
    tokenizer = tk.Tokenizer()
    
    if use_encoder:
        # get trained encoder weigths
        kernel, bias = get_encoder(tokenizer, word_embedding)
    
    # load data set
    if dataset == "imdb":
        HP = ImdbHP()
        train_data, train_labels, test_data, test_labels = imdb.get_imdb()
    else:
        HP = NewsGroupsHP()
        train_data, train_labels, labels_index = ng.get_data('data/20newsgroups/train')
        test_data, test_labels, _ = ng.get_data('data/20newsgroups/test')
    
    HP.MODEL_DIR = HP.MODEL_DIR + word_embedding + "/"
    # convert word to index
    train_idxs, train_sent_lens, train_docs_lens = data2idx(train_data, HP, tokenizer)
    test_idxs, test_sent_lens, test_docs_lens = data2idx(test_data, HP, tokenizer)
    
    # Important: we need to add 1 for zero index (pad value)
    HP.VOCAB_LEN = len(tokenizer.word2index)+1
    HP.START_TOKEN = tokenizer.start_index
    HP.END_TOKEN = tokenizer.end_index
    
    # get word embedding path
    if word_embedding == all_word_embeddings[0]:
        word_embedding_path = HP.GLOVE_PATH
    elif word_embedding == all_word_embeddings[1]:
        word_embedding_path = HP.FASTTEXT_PATH
    else:
        word_embedding_path = HP.LEXVEC_PATH
        
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            epoch_loss = 0

            # Create a new model or reload existing checkpoint
            model = classifier(HP)
            sess.run(tf.global_variables_initializer())
            
            if use_encoder:
                model.assign_encoder(sess, kernel, bias, train_encoder)
                
            # load glove and feed it to embedding layer
            weights = load_embedding(tokenizer.word2index, HP.EMB_DIM, word_embedding_path)
            model.assign_embedding(sess, weights)
            
            # Create a log writer object
            log_writer = tf.summary.FileWriter(HP.MODEL_DIR, graph=sess.graph)

            accuracies = []
            valid_loss = 0
            best_accuracy = 0
            
            saver = tf.train.Saver(max_to_keep=3)
            
            for e in range(HP.N_EPOCHS):
                # initial batchizer
                train_batchizer = batchizer(train_idxs,
                                            train_sent_lens,
                                            train_docs_lens,
                                            train_labels,
                                            HP.BATCH_SIZE,
                                            HP.MAX_SENT)
                test_batchizer = batchizer(test_idxs, 
                                           test_sent_lens,
                                           test_docs_lens, 
                                           test_labels, 
                                           HP.BATCH_SIZE,
                                           HP.MAX_SENT)

                if model.global_epoch_step.eval() >= HP.N_EPOCHS:
                    print('Training is already complete.')
                    break

                for batch_x, sent_len, docs_len, batch_y in train_batchizer:
                    if batch_x is None:
                        continue

                    batch_loss, summary = model.train(sess, batch_x, sent_len, docs_len, batch_y, HP.KEEP_PROB)
                    epoch_loss += batch_loss
                    log_writer.add_summary(summary, model.global_step.eval())

                for batch_x, sent_len, docs_len, batch_y in test_batchizer:

                    batch_loss, _, accuracy = model.eval(sess, batch_x, sent_len, docs_len, batch_y, 1.0)
                    valid_loss += batch_loss
                    accuracies.append(accuracy)

                acc_mean = np.mean(accuracies)
#                 if acc_mean > best_accuracy:
#                     best_accuracy = acc_mean
#                     with tf.device('/cpu:0'):
#                         checkpoint_path = os.path.join(HP.MODEL_DIR, HP.MODEL_NAME)
#                         save_path = saver.save(sess, checkpoint_path, global_step=model.global_step)
#                         print('model saved at %s' % save_path)
                print("epoch:{0:2}: train loss:{1:8.2f},".format(e, epoch_loss),
                      "validation loss:{0:8.2f} accuracy:{1:6.2f}%".format(valid_loss,
                                                                           acc_mean*100))
                accuracies = []
                epoch_loss = 0
                valid_loss = 0

                # Increase the epoch index of the model
                model.global_epoch_step_op.eval()
                
#             checkpoint_path = os.path.join(HP.MODEL_DIR, HP.MODEL_NAME)
#             save_path = saver.save(sess, checkpoint_path, global_step=model.global_step)
#             print('model saved at %s' % save_path)
        print('Training Terminated')        
        
        
        
if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)
    parser.add_argument('-d',
                        help='imdb or 20newsgroups',
                        default='imdb',
                        dest='dataset')
    parser.add_argument('-w',
                        help='word embedding: glove or fasttext or lexvec',
                        default='glove',
                        dest='word_embedding')
    parser.add_argument('--use_encoder',
                        help="use pretrained encoder",
                        default='yes',
                        dest='use_encoder')
    parser.add_argument('--train_encoder',
                        help="allow pretrained encoder to train during the classification",
                        default='yes',
                        dest='train_encoder')

    args = parser.parse_args()
    
    use_encoder = True
    if args.use_encoder.lower() == 'no':
        use_encoder = False
    
    train_encoder = True
    if args.train_encoder.lower() == 'no':
        train_encoder = False

    main(args.dataset, args.word_embedding, use_encoder, train_encoder)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        