import os
import json
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

# borrowed from: https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py
class Seq2SeqModel(object):

    def __init__(self, hParams, mode):

        self.HP = hParams
        self.mode = mode
        self.max_gradient_norm = hParams.MAX_GRADIANT_NORM
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        
        self.build_model()

    def build_model(self):
        print("building the model...")

        self.init_placeholders()
        self.init_embedding()
        self.build_encoder()
        self.build_decoder()

        self.summary_op = tf.summary.merge_all()


    def init_placeholders(self):
        # encoder inputs are include </s> tokens. e.g: "hello world </s>". So we can use them as decoder_output too.
        # shape: [Batch_size, max_timestep]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name="encoder_inputs")

        # shape: [Batch_size]
        self.encoder_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name="encoder_inputs_length")

        # shape: [Batch_size, max_timestep]
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name="decoder_inputs")

        # shape: [Batch_size]
        self.decoder_inputs_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name="decoder_inputs_length")

         # get dynamic batch size
        self.batch_size = tf.shape(self.encoder_inputs)[0]


    def init_embedding(self):
        
        self.emb_weights = tf.Variable(tf.constant(0.0,
                            shape=[self.HP.VOCAB_LEN, self.HP.EMB_DIM],dtype=tf.float32),
                            trainable=False,
                            name="embeddingWeights")
        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    [self.HP.VOCAB_LEN, self.HP.EMB_DIM],
                                                    name="weights_placeholder")
        self.embeddings = self.emb_weights.assign(self.embedding_placeholder)
        
    
    # assign embeding
    def assign_embedding(self, sess, weights):
        sess.run(self.embeddings, feed_dict={self.embedding_placeholder:weights})
        
        
    def build_encoder(self):
        print("build encoder layer...")
        with tf.variable_scope('encoder'):
            self.encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.HP.AE_LSTM_UNITS)
            # Embedded_inputs: [batch_size, time_step, embedding_size]
            with tf.device('/cpu:0'):
                self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.emb_weights,
                                                                      ids=self.encoder_inputs)

            # Encode input sequences into context vectors:
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                                                              inputs=self.encoder_inputs_embedded,
                                                                              sequence_length=self.encoder_lengths,
                                                                              dtype=tf.float32,
                                                                              time_major=False)


    def build_decoder(self):
        print("building decoder layer...")
        with tf.variable_scope('decoder'):
            self.decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.HP.AE_LSTM_UNITS)


            # Output projection layer to convert cell_outputs to logits
            output_layer = tf.layers.Dense(self.HP.VOCAB_LEN,
                                           kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=1),
                                           dtype=tf.float32,
                                           name='output_projection')

            if self.mode == 'train':
                with tf.device('/cpu:0'):
                    # decoder_inputs_embedded: [batch_size, max_time_step + <sos>, embedding_size]
                    self.decoder_inputs_embedded = tf.nn.embedding_lookup(params=self.emb_weights,
                                                                          ids=self.decoder_inputs)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                         sequence_length=self.decoder_inputs_lengths,
                                                         time_major=False,
                                                         name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=self.encoder_last_state,
                                                   output_layer=output_layer)

                # Maximum encoder time_steps in current batch
                max_encoder_length = tf.reduce_max(self.encoder_lengths)
                # Maximum encoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_lengths)

                # decoder_outputs_train: BasicDecoderOutput
                #                        namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_train.rnn_output: [batch_size, max_time_step + 1, num_decoder_symbols] if output_time_major=False
                #                                   [max_time_step + 1, batch_size, num_decoder_symbols] if output_time_major=True
                # decoder_outputs_train.sample_id: [batch_size], tf.int32
                (self.decoder_outputs_train,
                 self.decoder_last_state_train,
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                                                            decoder=training_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_decoder_length))

                # logits_train: [batch_size, max_time_step + 1, decoder_vocab_size]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)
                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train,
                                                    axis=-1,
                                                    name='decoder_pred_train')

                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.encoder_lengths,
                                         maxlen=max_encoder_length,
                                         dtype=tf.float32,
                                         name='masks')

                # Computes per word average cross-entropy over a batch
                # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                  targets=self.encoder_inputs,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True,)
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)

                # Construct graphs for minimizing loss
                self.init_optimizer()


    def init_optimizer(self):
            print("setting optimizer..")
            trainable_params = tf.trainable_variables()
            
            learning_rate = tf.train.exponential_decay(self.HP.LEARNING_RATE, self.global_step,
                                               self.HP.DECAY_STEP, self.HP.DECAY_RATE, staircase=True)
            
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.loss, trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.HP.MAX_GRADIANT_NORM)

            # Update the model
            self.updates = self.opt.apply_gradients(zip(clip_gradients, trainable_params),
                                                    global_step=self.global_step)

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)
        
    
    def get_encoder(self):
        return self.encoder_cell

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.
        Args:
        session: tensorflow session to use.
        encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
            to feed as encoder inputs
        encoder_inputs_length: a numpy int vector of [batch_size]
            to feed as sequence lengths for each element in the given batch
        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
        average perplexity, and the outputs.
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_lengths.name] = encoder_inputs_length
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.decoder_inputs_lengths.name] = decoder_inputs_length

        output_feed = [self.updates,     # Update Op that does optimization
                       self.loss,        # Loss for current batch
                       self.summary_op,
                       self.decoder_logits_train]  # Training summary]

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2], outputs[3]

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        """Run a evaluation step of the model feeding the given inputs.
        Args:
        session: tensorflow session to use.
        encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
        to feed as encoder inputs
        encoder_inputs_length: a numpy int vector of [batch_size]
        to feed as sequence lengths for each element in the given batch
        Returns:
        A triple consisting of gradient norm (or None if we did not do backward),
        average perplexity, and the outputs.
        """
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_lengths.name] = encoder_inputs_length
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.decoder_inputs_lengths.name] = decoder_inputs_length

        output_feed = [self.loss,       # Loss for current batch
                       self.summary_op,
                      self.decoder_pred_train] # Training summary

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]   # loss

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        # Input feeds for dropout
        # input_feed[self.keep_prob_placeholder.name] = 1.0
    
        output_feed = [self.decoder_pred_decode]
        outputs = sess.run(output_feed, input_feed)
    
        # GreedyDecoder: [batch_size, max_time_step]
        return outputs[0]