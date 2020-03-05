import sys
import os
import nn
import time
import math
import numpy as np
import tensorflow as tf
import read_data
import random
from config import Config
from random import shuffle
from data_utils import pad_sequences
from general_utils import Progbar
from cnn import masked_conv1d_and_max
from attention import add_timing_signal
from __future__ import print_function


def load_glove(glove_path):
    with open(glove_path, "r", encoding="utf-8") as glove_f:
        all_vectors = []
        for line in glove_f:
            try:
                vectors = [float(word) for word in line.strip().split()[1:]]
                all_vectors.append(vectors)
                assert len(vectors) == 300
            except Exception as e:
                print("Warning : incomplete glove vector!")
                print(line.strip().split())
        return np.asarray(all_vectors, dtype=np.float32)


class Classifer(object):

    def get_hidden_states_before(self, hidden_states, step, shape, hidden_size):
        #padding :/zeros
        padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        #remove last steps
        displaced_hidden_states=hidden_states[:,:-step,:]
        #concat padding
        return tf.concat([padding, displaced_hidden_states], axis=1)

    def get_hidden_states_after(self, hidden_states, step, shape, hidden_size):
        #padding zeros
        padding=tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        #remove last steps
        displaced_hidden_states=hidden_states[:,step:,:]
        #concat padding
        return tf.concat([displaced_hidden_states, padding], axis=1)

    def sum_together(self, l):
        combined_state=None
        for tensor in l:
            if combined_state==None:
                combined_state=tensor
            else:
                combined_state=combined_state+tensor
        return combined_state
    
    def slstm_cell(self, name_scope_name, hidden_size, lengths, initial_hidden_states, initial_cell_states, config):
        with tf.name_scope(name_scope_name):
            #forget gate for left 
            with tf.name_scope("f1_gate"):
                #current
                Wxf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                #left right
                Whf1 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                #initial state
                Wif1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                #dummy node
                Wdf1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for right 
            with tf.name_scope("f2_gate"):
                Wxf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf2 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for inital states     
            with tf.name_scope("f3_gate"):
                Wxf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf3 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #forget gate for dummy states     
            with tf.name_scope("f4_gate"):
                Wxf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                Whf4 = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
                Wif4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wif")
                Wdf4 = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdf")
            #input gate for current state     
            with tf.name_scope("i_gate"):
                Wxi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxi")
                Whi = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whi")
                Wii = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wii")
                Wdi = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdi")
            #input gate for output gate
            with tf.name_scope("o_gate"):
                Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                Who = tf.Variable(tf.random_normal([2*hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
                Wio = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wio")
                Wdo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wdo")
            #bias for the gates    
            with tf.name_scope("biases"):
                bi = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
                bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
                bf1 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf1")
                bf2 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf2")
                bf3 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf3")
                bf4 = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bf4")

            #dummy node gated attention parameters
            #input gate for dummy state
            with tf.name_scope("gated_d_gate"):
                gated_Wxd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxf")
                gated_Whd = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Whf")
            #output gate
            with tf.name_scope("gated_o_gate"):
                gated_Wxo = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                gated_Who = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
            #forget gate for states of word
            with tf.name_scope("gated_f_gate"):
                gated_Wxf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Wxo")
                gated_Whf = tf.Variable(tf.random_normal([hidden_size, hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="Who")
            #biases
            with tf.name_scope("gated_biases"):
                gated_bd = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bi")
                gated_bo = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")
                gated_bf = tf.Variable(tf.random_normal([hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="bo")

        #filters for attention        
        mask_softmax_score = tf.cast(tf.sequence_mask(lengths), tf.float32)*1e25-1e25
        mask_softmax_score_expanded = tf.expand_dims(mask_softmax_score, dim=2)               
        #filter invalid steps
        sequence_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32),axis=2)
        #filter embedding states
        initial_hidden_states = initial_hidden_states * sequence_mask
        initial_cell_states = initial_cell_states * sequence_mask
        #record shape of the batch
        shape = tf.shape(initial_hidden_states)
        batch_size, seq_len = shape[0], shape[1]
        keep_word_emb = tf.identity(initial_hidden_states)

        with tf.variable_scope("layer_cal", initializer=tf.contrib.layers.xavier_initializer()):
            self.layer_w = tf.get_variable("layer_w", [hidden_size, config.layer_emb_size])
            self.layer_b = tf.get_variable("layer_b", [config.layer_emb_size])
            self.restore_w = tf.get_variable("restore_w", [config.layer_emb_size, hidden_size])


        with tf.variable_scope("first-bi-lstm",initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('fw'):
                fw_lstm0 = tf.contrib.rnn.BasicLSTMCell(config.hidden_size_lstm, forget_bias=0.0)
                fw_lstm0 = tf.contrib.rnn.DropoutWrapper(fw_lstm0, output_keep_prob=config.rnn_keep_prob)

            with tf.variable_scope('bw'):
                bw_lstm0 = tf.contrib.rnn.BasicLSTMCell(config.hidden_size_lstm, forget_bias=0.0)
                bw_lstm0 = tf.contrib.rnn.DropoutWrapper(bw_lstm0, output_keep_prob=config.rnn_keep_prob)

            (fw0, bw0), _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm0, bw_lstm0, inputs=initial_hidden_states, sequence_length=lengths, time_major=False, dtype=tf.float32)
            lstm_output = tf.concat([fw0, bw0], axis=-1)

        initial_hidden_states = tf.nn.dropout(lstm_output, config.rnn_keep_prob)
        layer_rep = tf.reshape(initial_hidden_states, [-1, config.hidden_size])   # [b*s, hidden_size]
        layer_rep = tf.matmul(layer_rep, self.layer_w) + self.layer_b
        layer_rep = tf.nn.tanh(layer_rep)
        # add  a small droupout
        layer_rep = tf.nn.dropout(layer_rep, 0.9)
        # restore to hidden size
        # predict depths
        input_hidden_states = tf.reshape(tf.matmul(layer_rep, self.restore_w), [batch_size, seq_len, hidden_size]) * sequence_mask
        input_cell_states = input_hidden_states
        layer_logits = tf.matmul(layer_rep, tf.transpose(self.layer_embedding, [1,0]))  # [b*s, layer_dim]  [layer_dim, layer_num]
        layer_logits = tf.reshape(layer_logits, [batch_size, seq_len, config.layer])
        layer_logits = tf.nn.softmax(layer_logits, axis=-1) # [b, s, layer]
        index = tf.range(0, config.layer, dtype=tf.float32)  # [0, 1, 2, ...]
        pred_layer_ids = tf.cast(tf.reduce_sum(tf.reshape(index, [1, 1, -1]) * layer_logits, axis=-1), tf.int32) # [b, s]
        layer_emb = tf.nn.embedding_lookup(self.layer_embedding, pred_layer_ids)   # [b, s, layer_emb]   
        pred_layer_ids = tf.reshape(pred_layer_ids, [-1])

        #initial embedding states
        input_emb = tf.concat([keep_word_emb, layer_emb], -1)
        embedding_hidden_state = tf.reshape(input_emb, [-1, hidden_size])
        embedding_cell_state =  tf.reshape(input_emb, [-1, hidden_size])

        #inital dummy node states
        dummynode_hidden_states = tf.reduce_mean(input_emb, axis=1)
        dummynode_cell_states = tf.reduce_mean(input_emb, axis=1)

        for i in range(config.layer):
            #update dummy node states
            #average states
            combined_word_hidden_state = tf.reduce_mean(input_hidden_states, axis=1)  # [b, hidden]
            reshaped_hidden_output = tf.reshape(input_hidden_states, [-1, hidden_size]) # [b,*s, hidden + layer_emb]
            #copy dummy states for computing forget gate
            # [b*s, h+layer]
            transformed_dummynode_hidden_states = tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
            #input gate
            gated_d_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxd) + tf.matmul(combined_word_hidden_state, gated_Whd) + gated_bd
            )
            #output gate
            gated_o_t = tf.nn.sigmoid(
                tf.matmul(dummynode_hidden_states, gated_Wxo) + tf.matmul(combined_word_hidden_state, gated_Who) + gated_bo
            )
            #forget gate for hidden states
            gated_f_t = tf.nn.sigmoid(
                tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) + tf.matmul(reshaped_hidden_output, gated_Whf) + gated_bf
            )

            #softmax on each hidden dimension 
            reshaped_gated_f_t = tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size])+ mask_softmax_score_expanded
            gated_softmax_scores = tf.nn.softmax(tf.concat([reshaped_gated_f_t, tf.expand_dims(gated_d_t, dim=1)], axis=1), axis=1)
            #split the softmax scores
            new_reshaped_gated_f_t = gated_softmax_scores[:,:shape[1],:]
            new_gated_d_t = gated_softmax_scores[:,shape[1]:,:]
            #new dummy states
            dummy_c_t = tf.reduce_sum(new_reshaped_gated_f_t * input_cell_states, axis=1) + tf.squeeze(new_gated_d_t, axis=1)*dummynode_cell_states
            dummy_h_t = gated_o_t * tf.nn.tanh(dummy_c_t) # [b, h+layer]
            #update word node states
            #get states before
            input_hidden_states_before = [tf.reshape(self.get_hidden_states_before(input_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            input_hidden_states_before = self.sum_together(input_hidden_states_before)
            input_hidden_states_after = [tf.reshape(self.get_hidden_states_after(input_hidden_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            input_hidden_states_after = self.sum_together(input_hidden_states_after)
            #get states after
            input_cell_states_before = [tf.reshape(self.get_hidden_states_before(input_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            input_cell_states_before = self.sum_together(input_cell_states_before)
            input_cell_states_after = [tf.reshape(self.get_hidden_states_after(input_cell_states, step+1, shape, hidden_size), [-1, hidden_size]) for step in range(self.config.step)]
            input_cell_states_after = self.sum_together(input_cell_states_after)
            
            #reshape for matmul
            input_hidden_states = tf.reshape(input_hidden_states, [-1, hidden_size]) # [b*s, h+layer]
            input_cell_states = tf.reshape(input_cell_states, [-1, hidden_size]) # [b*s, h+layer]

            #concat before and after hidden states
            concat_before_after = tf.concat([input_hidden_states_before, input_hidden_states_after], axis=-1)

            #copy dummy node states 
            transformed_dummynode_hidden_states = tf.reshape(tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1],1]), [-1, hidden_size])
            transformed_dummynode_cell_states = tf.reshape(tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1],1]), [-1, hidden_size])

            f1_t = tf.nn.sigmoid(
                tf.matmul(input_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) + 
                tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1)+ bf1
            )

            f2_t = tf.nn.sigmoid(
                tf.matmul(input_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) + 
                tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2)+ bf2
            )

            f3_t = tf.nn.sigmoid(
                tf.matmul(input_hidden_states, Wxf3) + tf.matmul(concat_before_after, Whf3) + 
                tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) + bf3
            )

            f4_t = tf.nn.sigmoid(
                tf.matmul(input_hidden_states, Wxf4) + tf.matmul(concat_before_after, Whf4) + 
                tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) + bf4
            )
            
            i_t = tf.nn.sigmoid(
                tf.matmul(input_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) + 
                tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi)+ bi
            )
            
            o_t = tf.nn.sigmoid(
                tf.matmul(input_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) + 
                tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) + bo
            )
            
            f1_t, f2_t, f3_t, f4_t, i_t=tf.expand_dims(f1_t, axis=1), tf.expand_dims(f2_t, axis=1),tf.expand_dims(f3_t, axis=1), tf.expand_dims(f4_t, axis=1), tf.expand_dims(i_t, axis=1)


            five_gates=tf.concat([f1_t, f2_t, f3_t, f4_t,i_t], axis=1)
            five_gates=tf.nn.softmax(five_gates, dim=1)
            f1_t,f2_t,f3_t, f4_t,i_t= tf.split(five_gates, num_or_size_splits=5, axis=1)
            
            f1_t, f2_t, f3_t, f4_t, i_t=tf.squeeze(f1_t, axis=1), tf.squeeze(f2_t, axis=1),tf.squeeze(f3_t, axis=1), tf.squeeze(f4_t, axis=1),tf.squeeze(i_t, axis=1)

            c_t = (f1_t * input_cell_states_before) + (f2_t * input_cell_states_after)+(f3_t * embedding_cell_state)+ (f4_t * transformed_dummynode_cell_states)+ (i_t * input_cell_states)
            
            h_t = o_t * tf.nn.tanh(c_t)
            #update states
            output_hidden_states = tf.reshape(h_t, [shape[0], shape[1], hidden_size])
            output_cell_states = tf.reshape(c_t, [shape[0], shape[1], hidden_size])
            output_hidden_states = output_hidden_states * sequence_mask
            output_cell_states = output_cell_states * sequence_mask

            dummynode_hidden_states = dummy_h_t
            dummynode_cell_states = dummy_c_t

            cur_layer_ids = tf.fill([shape[0]*shape[1]], i)
            output_hidden_states = tf.reshape(output_hidden_states, [shape[0]*shape[1], hidden_size])
            output_cell_states = tf.reshape(output_cell_states, [shape[0]*shape[1], hidden_size])
            
            input_hidden_states = tf.where(cur_layer_ids < pred_layer_ids, output_hidden_states, input_hidden_states)
            input_hidden_states = tf.reshape(input_hidden_states, [shape[0], shape[1], hidden_size])

            input_cell_states = tf.where(cur_layer_ids < pred_layer_ids, output_cell_states, input_cell_states)
            input_cell_states = tf.reshape(input_cell_states, [shape[0], shape[1], hidden_size])
            
        input_hidden_states = tf.nn.dropout(input_hidden_states, config.rnn_keep_prob)
        input_cell_states = tf.nn.dropout(input_cell_states, config.rnn_keep_prob)

        return input_hidden_states, input_cell_states, dummynode_hidden_states



    def __init__(self, config, session):
        #inputs: features, mask, keep_prob, labels
        self.input_data = tf.placeholder(tf.int32, [None, None], name="inputs")
        self.labels = tf.placeholder(tf.int64, [None,], name="labels")
        self.mask = tf.placeholder(tf.int32, [None,], name="mask")
        self.dropout = self.keep_prob = keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.chars = tf.placeholder(tf.int32, [None, None, None], name="input_char")
        self.char_mask = tf.placeholder(tf.int32, [None, None], name="char_mask")
        self.max_word_len = tf.placeholder(tf.int32, [], name="max_length_word")

        self.config = config
        shape = tf.shape(self.input_data)
        with tf.variable_scope("word_embedding"):
            self.embedding = tf.Variable(load_glove(self.config.glove_path),
                                         dtype=tf.float32,
                                         name="embedding", 
                                         trainable=config.embedding_trainable
            )
            input_words = tf.nn.embedding_lookup(self.embedding, self.input_data)
        with tf.variable_scope("layer_embedding", reuse=tf.AUTO_REUSE):
            self.layer_embedding = tf.get_variable("layer_embedding", [self.config.layer, self.config.layer_emb_size], initializer=tf.contrib.layers.xavier_initializer())  
     
        with tf.variable_scope("char_embedding"):
            char_emb = tf.get_variable("char_embedding",
                                     [self.config.char_size, self.config.char_emb_size],
                                      initializer=tf.contrib.layers.xavier_initializer()
            )
            char_inputs = tf.nn.embedding_lookup(char_emb, self.chars)
            mask_weights = tf.sequence_mask(self.char_mask, maxlen=self.max_word_len)
            char_cnn_emb = masked_conv1d_and_max(char_inputs, mask_weights, self.config.char_emb_size, 3, self.config.char_emb_size)

        concated_inputs = tf.concat([input_words, char_cnn_emb], -1)
        final_inputs = tf.reshape(concated_inputs, [shape[0], -1, self.config.char_emb_size + 300])
        initial_hidden_states = final_inputs
        initial_cell_states = tf.identity(initial_hidden_states)
        initial_hidden_states = tf.nn.dropout(initial_hidden_states,keep_prob)
        initial_cell_states = tf.nn.dropout(initial_cell_states, keep_prob)

        #create layers 
        new_hidden_states, new_cell_state, dummynode_hidden_states=self.slstm_cell("slstm", config.hidden_size, self.mask, initial_hidden_states, initial_cell_states, config)
        
        softmax_w = tf.Variable(tf.random_normal([3*(config.hidden_size ), config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w")
        softmax_b = tf.Variable(tf.random_normal([config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b")
        batch_size = tf.shape(new_hidden_states)[0]
        sent_size = tf.shape(new_hidden_states)[1]  

        with tf.name_scope("mean"):
            mask_weight = tf.expand_dims(self.mask, -1)
            mask_weight = tf.to_float(mask_weight)
            mean_emb = tf.reduce_sum(new_hidden_states, axis=-2) / mask_weight

        with tf.name_scope("max"):
            float_mask = tf.to_float(self.mask)
            reshaped_mask = tf.reshape(float_mask, shape=[batch_size, 1, 1])
            tile_mask = tf.tile(reshaped_mask, [1, sent_size, (config.hidden_size) * 1])
            concat_hidden = tf.concat([new_hidden_states, tile_mask], axis=-2)
            max_emb = tf.map_fn(lambda x: tf.reduce_max(x[: tf.cast(x[-1][-1], tf.int32)], axis=-2), concat_hidden)

        sent_emb = tf.concat([mean_emb, max_emb, dummynode_hidden_states], axis=-1)
        representation = tf.nn.relu(sent_emb)

        
        self.logits = tf.matmul(representation, softmax_w) + softmax_b
        #operators for prediction
        self.prediction = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(self.prediction, self.labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        
        #cross entropy loss
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.cost = tf.reduce_mean(self.loss) + config.l2_beta * tf.nn.l2_loss(self.embedding)

        #designate training variables
        tvars = tf.trainable_variables()
        self.lr = tf.Variable(0.0, trainable=False)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

        grads = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)
        self.grads = grads
        optimizer = tf.train.AdamOptimizer(config.learning_rate)        
        self.train_op = optimizer.apply_gradients(zip(self.grads, tvars), global_step=self.global_step)
    
    #assign value to learning rate
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

def get_minibatches_idx(n_samples, batch_size, shuffle=True):
    idx_list = np.arange(n_samples, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n_samples // batch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + batch_size])
        minibatch_start += batch_size
    if (minibatch_start != n_samples):
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

def run_epoch(session, config, model, data, eval_op, keep_prob, is_training, saver=None):
    n_samples = len(data[0])
    minibatches = get_minibatches_idx(n_samples, config.batch_size, shuffle=config.shuffle)
    correct = 0.
    total = 0
    total_cost=0
    prog = Progbar(target=len(minibatches))
    for i, idx in enumerate(minibatches):
        x = data[0][idx]
        chars = data[3][idx]
        x, mask = pad_sequences(x, 0, max_seq_len=config.max_seq_len)
        if [] in chars:
            continue
        chars, char_mask,  max_word_len = pad_sequences(chars, pad_tok=0, nlevels=2, max_seq_len=config.max_seq_len, max_word_len=config.max_word_len)
        
        y = data[1][idx]
        max_word_len = max_word_len if max_word_len < config.max_word_len else config.max_word_len

        global_step, count, _, cost = session.run([model.global_step, model.accuracy, eval_op, model.cost],\
            {model.input_data: x, model.labels: y, model.mask: mask, model.keep_prob: keep_prob, model.chars: chars, model.char_mask: char_mask, model.max_word_len :  max_word_len})
        correct += count 
        total += len(idx)
        total_cost += cost
        if is_training:
            prog.update(i + 1, [("train loss", cost), ("step", global_step)])
        if global_step % config.save_step == 0 and saver:
           saver.save(session, config.save_path, global_step=global_step)
        del cost
    accuracy = correct / total
    return accuracy, total_cost, 

def train_test_model(config, epoch_id, session, model, train_dataset, test_dataset, saver=None):
    lr_decay = config.lr_decay ** max(epoch_id - config.max_epoch, 0.0)
    model.assign_lr(session, config.learning_rate * lr_decay)
    print("Epoch: %d Learning rate: %.5f" % (epoch_id + 1, session.run(model.lr)))
    start_time = time.time()
    train_acc, total_loss = run_epoch(session, config, model, train_dataset, model.train_op, config.keep_prob, True, saver)
    print("Train Accuracy = %.2f  loss: %.2f  time = %.2f seconds\n" % (100*train_acc, total_loss, time.time() - start_time))
    if epoch_id % config.eval_step == 0:
        start_time = time.time()
        test_acc, _ = run_epoch(session, config, model, test_dataset, tf.no_op(), 1, False)
        print("Eval Accuracy = %.2f  time: %.3f\n" % (100*test_acc, time.time()-start_time))   


def word_to_vec(matrix, session,config, *args):
    for model in args:
        session.run(tf.assign(model.embedding, matrix))

if __name__ == "__main__":
    
    config = Config()
    train_dataset, test_dataset = read_data.load_data(path=config.data_path + "parsed_data/")   
    # conver datas into matrix
    train_dataset = read_data.prepare_data(train_dataset[0], train_dataset[1], train_dataset[2])
    test_dataset = read_data.prepare_data(test_dataset[0], test_dataset[1], test_dataset[2])

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto()) as session:
        classifier = Classifer(config=config, session=session)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        for epoch_id in range(config.max_max_epoch):
            train_test_model(config, epoch_id, session, classifier, train_dataset, test_dataset, saver)

