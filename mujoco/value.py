import tensorflow as tf
import numpy as np
import utils as U

class Value:
    def __init__(self, config, sess, suffix):
        self.config = config
        self.sess = sess
        
        with tf.variable_scope('Value_{}'.format(suffix)):
            self.state = tf.placeholder(tf.float32, [None, None, config.state_dim])
            self.batch_size = tf.shape(self.state)[0]
            self.traj_len = tf.shape(self.state)[1]
            self.code = tf.placeholder(tf.int32, [None])
            self.value_spvs = tf.placeholder(tf.float32, [None, None])
            self.value_old = tf.placeholder(tf.float32, [None, None])
            self.update_op = None

            self.code_oh = tf.one_hot(self.code, depth = config.code_dim)
            
            self.state_encoding = self._encode_state(self.state)
            
            self.code_encoding = self._encode_code(self.code_oh)

            self.state_code_encoding = self._encode_state_code(self.state_encoding, self.code_encoding)

            self.value = self._get_value(self.state_code_encoding)
            
            self.value_old_abs = tf.abs(self.value_old)
            self.value_clipped = self.value_old + tf.clip_by_value(
                self.value - self.value_old, 
                - config.value_clip_range * self.value_old_abs, 
                config.value_clip_range * self.value_old_abs
            )
            self.value_loss_1 = (self.value - self.value_spvs) ** 2
            self.value_loss_2 = (self.value_clipped - self.value_spvs) ** 2
            self.clipped_freq = tf.reduce_mean(tf.cast(self.value_loss_1 < self.value_loss_2, tf.float32))

            self.value_loss = 0.5 * tf.reduce_mean(tf.maximum(self.value_loss_1, self.value_loss_2))
            self.train_op = tf.train.AdamOptimizer(config.value_lr).minimize(self.value_loss)


    def get_value(self, state, code):
        value = self.sess.run(
            self.value,
            feed_dict = {
                self.state: state,
                self.code: code
            }
        )

        return value
    

    def train(self, state, code, value_spvs, value_old):
        loss, value, clipped_freq, _ = self.sess.run(
            [self.value_loss, self.value, self.clipped_freq, self.train_op],
            feed_dict = {
                self.state: state, 
                self.code: code,
                self.value_spvs: value_spvs,
                self.value_old: value_old
            }
        )

        return loss, value, clipped_freq


    def _encode_state(self, state):
        state_layer = state
        for fc_dim in self.config.state_fc_dims:
            state_layer = tf.layers.dense(state_layer, fc_dim, activation = self.config.activation)
        
        cell = tf.keras.layers.GRU(self.config.hidden_dim, return_sequences = True, name = 'gru')
        state_encode = cell(state_layer)
        return state_encode


    def _encode_code(self, code):
        code_layer = code
        for fc_dim in self.config.code_fc_dims:
            code_layer = tf.layers.dense(code_layer, fc_dim, activation = self.config.activation)
        
        return code_layer
    

    def _encode_state_code(self, state_encode, code_encode):
        code_encode = tf.tile(code_encode[:, tf.newaxis, :], [1, self.traj_len, 1])
        if self.config.state_code_arch == 'add':
            state_code_encode = state_encode + code_encode
        elif self.config.state_code_arch == 'concat':
            state_code_encode = tf.concat([state_encode, code_encode], axis = 2)
        else:
            raise NotImplementedError
        
        return state_code_encode

    
    def _get_value(self, state_code):
        value_layer = state_code
        for dim in self.config.value_fc_dims:
            value_layer = tf.layers.dense(value_layer, dim, activation = self.config.activation)
        
        value = tf.layers.dense(value_layer, 1, activation = None)[:, :, 0]
        
        return value


    def set_update_op(self, value_vars, old_value_vars, soft = False):
        if self.update_op is None:
            if soft:
                self.update_op = tf.group([v_t.assign(v_t * (1 - 0.2) + v * 0.2) for v_t, v in zip(old_value_vars, value_vars)])
            else:
                self.update_op = tf.group([v_t.assign(v) for v_t, v in zip(old_value_vars, value_vars)])
        else:
            raise Exception('duplicate update_op in value_net')
    

    def run_update_op(self):
        # assert(self.update_op is not None)
        self.sess.run(self.update_op)
