import tensorflow as tf
import numpy as np

class classifier(object):
    def __init__(self, hParams):
        self.HP = hParams
        self.max_gradient_norm = hParams.MAX_GRADIANT_NORM
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        
        self.init_placeholders()
        self.init_embedding()
        self.build_graph()
        # Construct graphs for minimizing loss
        self.init_optimizer()
        self.init_prediction()
        self.summary_op = tf.summary.merge_all()
    
    def init_placeholders(self):
        # shape= [batch, max_sentences, words + <eos>]
        self.inputs = tf.placeholder(dtype=tf.int32, 
                                     shape=(self.HP.BATCH_SIZE, self.HP.MAX_SENT, self.HP.MAX_WORD+1), 
                                     name="inputs")
        # shape = [batch]
        self.labels = tf.placeholder(dtype=tf.int32, shape=(self.HP.BATCH_SIZE,), name="labels")
        # shape = [batch, sentences]
        self.sent_len = tf.placeholder(dtype=tf.int32, shape=(self.HP.BATCH_SIZE, self.HP.MAX_SENT), name="sentences_length")
        # shape = [batch]
        self.docs_len = tf.placeholder(dtype=tf.int32, shape=(self.HP.BATCH_SIZE), name="docs_length")
        
        # dropout 
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="keep_prob")
        
    
    def init_embedding(self):
        self.emb_weights = tf.Variable(tf.constant(0.0,
                            shape=[self.HP.VOCAB_LEN, self.HP.EMB_DIM],dtype=tf.float32),
                            trainable=False,
                            name="embeddingWeights")
        self.embedding_placeholder = tf.placeholder(dtype=tf.float32,
                                                    shape=[self.HP.VOCAB_LEN, self.HP.EMB_DIM],
                                                    name="weights_placeholder")
        self.embeddings = self.emb_weights.assign(self.embedding_placeholder)
     
    
    # assign embeding
    def assign_embedding(self, sess, weights):
        sess.run(self.embeddings, feed_dict={self.embedding_placeholder:weights})
        
        
    def assign_encoder(self, sess, kernel, bias, trainable):
        # TODO: correct the dimensions
        trained_kernel = tf.placeholder(dtype=tf.float32, shape=(556, 1024), name="trained_kernel")
        trained_bias = tf.placeholder(dtype=tf.float32, shape=(1024,), name="trained_bias")

        
        xavi_kernel = self.get_var(name="rnn/basic_lstm_cell/kernel:0")
        if xavi_kernel is None:
            raise ValueError("there is no kernel")
        
        xavi_bias = self.get_var(name="rnn/basic_lstm_cell/bias:0")
        if xavi_bias is None:
            raise ValueError("there is no Bias")
        
        
        xavi_kernel.trainable = trainable
        xavi_bias.trainable = trainable
        
        kernel_assign_op = xavi_kernel.assign(trained_kernel)
        bias_assign_op = xavi_bias.assign(trained_bias)
        
        sess.run([kernel_assign_op, bias_assign_op],
                 feed_dict={trained_kernel:kernel,
                            trained_bias:bias})
        
    def build_graph(self):
        xavi = tf.contrib.layers.xavier_initializer()
        l2_reg = tf.contrib.layers.l2_regularizer(0.1)
        # reshape to = [batch_size * max_sent, words] TODO: max_sent is hard coded
        inputs = tf.reshape(self.inputs, shape=(self.HP.BATCH_SIZE*self.HP.MAX_SENT, -1))
        
        # reshape inputs len too
        inputs_len = tf.reshape(self.sent_len, shape=(-1,))

        with tf.device('/cpu:0'):
            # shape = [batch_size * max_sent, words, emb_dim]
            self.inputs_embedded = tf.nn.embedding_lookup(params=self.emb_weights, ids=inputs)
        
        with tf.name_scope("encoder"):
            self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.HP.AE_LSTM_UNITS)


            # encoder_outputs: [batch_size * max_sent, words, cell_output_size]
            inputs, _ = tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                          inputs=self.inputs_embedded,
                                          sequence_length=inputs_len,
                                          dtype=tf.float32,
                                          time_major=False)
        with tf.name_scope("reshaping"):
            # reshape to [batch_size *  max_sent * words, cell_output_size]
            inputs = tf.reshape(inputs, shape=(-1, self.HP.AE_LSTM_UNITS))

            inputs = tf.layers.dense(inputs, self.HP.DENSE_REDUCER,
                                     activation=tf.nn.relu,
                                     kernel_initializer=xavi,
                                     kernel_regularizer=l2_reg)

            # reshape back to [batch_size, max_sen, -1]
            inputs = tf.reshape(inputs, shape=(self.HP.BATCH_SIZE, self.HP.MAX_SENT, -1))
        
        
        with tf.variable_scope("classifier"):
        # initial classifier LSTM cell
            classifier_cell = tf.nn.rnn_cell.BasicLSTMCell(self.HP.LSTM_UNITS)

            # wrap classifier cell inside a dropout
            classifier_cell = tf.contrib.rnn.DropoutWrapper(classifier_cell,
                                                            output_keep_prob=self.keep_prob,
                                                            seed = self.HP.SEED)

            # classifier_outputs = [batch_size, max_sent, cell_output_size]
            classifier_outputs, _ = tf.nn.dynamic_rnn(cell=classifier_cell,
                                                      inputs=inputs,
                                                      sequence_length=self.docs_len,
                                                      dtype=tf.float32,
                                                      time_major=False)
        with tf.name_scope("densing"):
            # flatten outputs
            inputs = tf.contrib.layers.flatten(classifier_outputs)

            # drop out some values
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
            # reduce dimension
            inputs = tf.layers.dense(inputs, self.HP.DENSE_1,
                                     activation=tf.nn.relu,
                                     kernel_initializer=xavi,
                                     kernel_regularizer=l2_reg)

            # drop out some values
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
            # reduce dimension
            inputs = tf.layers.dense(inputs, self.HP.DENSE_2,
                                     activation=tf.nn.relu,
                                     kernel_initializer=xavi,
                                     kernel_regularizer=l2_reg)
        
        with tf.name_scope("logits"):
            # calculate logits
            self.logits = tf.layers.dense(inputs, self.HP.N_LABELS)

            oh = tf.one_hot(self.labels, self.HP.N_LABELS, dtype=tf.float32)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=oh, logits=self.logits))
        tf.summary.scalar('loss', self.loss)
    
    def init_prediction(self):
        # calculate prediction for validation set
        with tf.name_scope("prediction"):
            pred = tf.nn.softmax(self.logits)
            pred = tf.argmax(pred, axis=1, output_type=tf.int32)
            correct_prediction = tf.equal(pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    def init_optimizer(self):
        with tf.name_scope("optimizer"):
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
    
    def get_var(self, name):
        for v in tf.trainable_variables():
            if v.name==name:
                return v
        return None
    
    def train(self, sess, inputs, sent_len, docs_len, labels, keep_prob):

        sent_len = np.asarray(sent_len)
        docs_len = np.asarray(docs_len)
        labels = np.asarray(labels)
        
        input_feed = {}
        input_feed[self.inputs.name] = inputs
        input_feed[self.sent_len.name] = sent_len
        input_feed[self.docs_len.name] = docs_len
        input_feed[self.labels.name] = labels
        input_feed[self.keep_prob.name] = keep_prob

        output_feed = [self.updates,     # Update Op that does optimization
                       self.loss,        # Loss for current batch
                       self.summary_op]  

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]

    def eval(self, sess, inputs, sent_len, docs_len, labels, keep_prob):
        
        sent_len = np.asarray(sent_len)
        docs_len = np.asarray(docs_len)
        labels = np.asarray(labels)
        
        input_feed = {}
        input_feed[self.inputs.name] = inputs
        input_feed[self.sent_len.name] = sent_len
        input_feed[self.docs_len.name] = docs_len
        input_feed[self.labels.name] = labels
        input_feed[self.keep_prob.name] = keep_prob

        output_feed = [self.loss,
                       self.summary_op,
                       self.accuracy] 

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]

#     def predict(self, sess, encoder_inputs, encoder_inputs_length):
#         # Input feeds for dropout
#         # input_feed[self.keep_prob_placeholder.name] = 1.0
#         sent_len = np.asarray(sent_len)
#         docs_len = np.asarray(docs_len)
#         labels = np.asarray(labels)
        
#         output_feed = [self.decoder_pred_decode]
#         outputs = sess.run(output_feed, input_feed)
    
#         # GreedyDecoder: [batch_size, max_time_step]
#         return outputs[0]