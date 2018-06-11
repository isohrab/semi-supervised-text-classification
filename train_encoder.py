import os
import json
import math

import numpy as np
import tensorflow as tf

import wikiloader as wl
import Batchizer
from hyper_parameters import AEncoderHP
from Seq2SeqModel import Seq2SeqModel
from utils import *
import argparse

def create_model(session, hp):
    model = Seq2SeqModel(hp, 'train')

    ckpt = tf.train.get_checkpoint_state(hp.MODEL_DIR)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        if not os.path.exists(hp.MODEL_DIR):
            os.makedirs(hp.MODEL_DIR)
        print('Created new model parameters..')
        session.run(tf.global_variables_initializer())

    return model

    
    
def main(HP, word_embedding):
    HP.START_TOKEN = batchizer.tokenizer.start_index
    HP.END_TOKEN = batchizer.tokenizer.end_index
    HP.VOCAB_LEN = len(batchizer.tokenizer.word2index)+1 # +1 because of PAD=zero index in weight
    
    all_word_embeddings = ["glove", "fasttext", "lexvec"]
    word_embedding = word_embedding.lower()
    if not word_embedding in all_word_embeddings:
        raise InvalidArgument
    
    # get word embedding path
    if word_embedding == all_word_embeddings[0]:
        word_embedding_path = HP.GLOVE_PATH
    elif word_embedding == all_word_embeddings[1]:
        word_embedding_path = HP.FASTTEXT_PATH
    else:
        word_embedding_path = HP.LEXVEC_PATH
       
    HP.MODEL_DIR = HP.MODEL_DIR +word_embedding+"/"
        
    with tf.Session() as sess:
        epoch_loss = 0
        display_freq = HP.DISPLAY_EVERY
        save_every= HP.SAVE_EVERY
        validate_every = HP.VALIDATE_EVERY
        validating = 5 # Batches
        
        
        # Create a new model or reload existing checkpoint
        model = create_model(sess, HP)
        
        # Create a log writer object
        log_writer = tf.summary.FileWriter(HP.MODEL_DIR, graph=sess.graph)
        
        # load glove and feed it to embedding layer
        weights = load_embedding(batchizer.tokenizer.word2index, HP.EMB_DIM, word_embedding_path)
        model.assign_embedding(sess, weights)
        
        saver = tf.train.Saver(max_to_keep=3)
        
        for e in range(HP.N_EPOCHS):
            if model.global_epoch_step.eval() >= HP.N_EPOCHS:
                print('Training is already complete.')
                break

            for enc_in, enc_len, dec_in, dec_len in batchizer.next_train_batch():
                if enc_in is None:
                    continue

                batch_loss, summary, logits = model.train(sess, enc_in, enc_len, dec_in, dec_len)
                
                epoch_loss += float(batch_loss) / display_freq

                if model.global_step.eval() % display_freq == 0:
                    avg_perplexity = math.exp(float(epoch_loss)) if epoch_loss < 300 else float("inf")
                    print('Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(),
                              'Perplexity {0:.3f}'.format(avg_perplexity))
                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())
                    epoch_loss = 0

                if model.global_step.eval() % validate_every == 0:
                    print('Validation step')
                    valid_loss = 0
                    validated = 0
                    valid_sents_seen = 0
                    for enc_in_val, enc_len_val, dec_in_val, dec_len_val in batchizer.next_val_batch():
                        batch_loss, _, sample_ids = model.eval(sess, enc_in_val, enc_len_val, dec_in_val, dec_len_val)

                        valid_loss += batch_loss * HP.BATCH_SIZE
                        valid_sents_seen += HP.BATCH_SIZE
                        validated += 1
                        rand_idx = np.random.randint(0, len(sample_ids))
                        enc_told = batchizer.tokenizer.idx2text(enc_in_val[rand_idx])
                        dec_thought = batchizer.tokenizer.idx2text(sample_ids[rand_idx])
                        print("encoder told:", enc_told)
                        print("decoder thought:", dec_thought)
                        if validated > validating:
                            break
                     
                    valid_loss = valid_loss / valid_sents_seen
                    print('Valid perplexity: {0:.3f}'.format(math.exp(valid_loss)))

                # Save the model checkpoint
                if model.global_step.eval() % save_every == 0:
                    print('Saving the model..')
                    with tf.device('/cpu:0'):
                        checkpoint_path = os.path.join(HP.MODEL_DIR, HP.MODEL_NAME)
                        save_path = saver.save(sess, save_path=checkpoint_path, global_step=model.global_step)
                        print('model saved at %s' % save_path)

            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))

        print('Saving the last model..')
        checkpoint_path = os.path.join(HP.MODEL_DIR, HP.MODEL_NAME)
        save_path = saver.save(sess, save_path=checkpoint_path, global_step=model.global_step)
        print('model saved at %s' % save_path)

    print('Training Terminated')



if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)
    parser.add_argument('word_embedding',
                        help='word embedding: glove or fasttext or lexvec')
    parser.add_argument('--vocab_size',
                        help="number of token in tokenizer",
                        default=100000,
                        type=int)
    parser.add_argument('--retokenize',
                        help="retokenize wikipedia file, if vocab size is changed",
                        default=False,
                        type=bool)
    parser.add_argument('--lstm_units',
                        help="size of Auto encoder LSTM",
                        default=128,
                        type=int)
    parser.add_argument('--n_epochs',
                        help="number of epcohs",
                        default=10,
                        type=int)
    

    args = parser.parse_args()
    HP = AEncoderHP()
    HP.TOKENIZER_MAX_WORDS = args.vocab_size
    HP.N_EPOCHS = args.n_epochs
    HP.AE_LSTM_UNITS = args.lstm_units

    batchizer = Batchizer.Batchizer(HP.BATCH_SIZE,
                                    tokenizer_max_words=HP.TOKENIZER_MAX_WORDS,
                                    pad_len=None,
                                    retokenize=args.retokenize)
    main(HP, args.word_embedding)