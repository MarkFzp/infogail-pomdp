import tensorflow as tf
import numpy as np

class Posterior:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

        with tf.name_scope('Posterior'):
            self.stu_state = tf.placeholder(tf.float32, [None, None, config.state_dim])
            self.stu_action = tf.placeholder(tf.float32, [None, None, config.action_dim])
            self.code = tf.placeholder(tf.int32, [None])

            self.batch_size = tf.shape(self.stu_state)[0]
            self.traj_len = tf.shape(self.stu_state)[1]

            self.code_posterior_dist = self._get_posterior_dist(self.stu_state, self.stu_action)
            self.code_posterior_idx = tf.reshape(tf.stack([
                tf.tile(tf.range(self.batch_size)[:, tf.newaxis], [1, self.traj_len]),
                tf.tile(tf.range(self.traj_len)[tf.newaxis, :], [self.batch_size, 1]),
                tf.tile(self.code[:, tf.newaxis], [1, self.traj_len])
            ], axis = 2), [-1, 3])
            self.log_code_posterior = tf.reshape(
                tf.log(tf.gather_nd(self.code_posterior_dist, self.code_posterior_idx)) / tf.log(2.0), 
                [self.batch_size, self.traj_len]
            )
            
            self.mean_log_code_posterior = tf.reduce_mean(self.log_code_posterior)
            self.loss = -1 * config.post_coef * self.mean_log_code_posterior
            self.train_opt = tf.train.AdamOptimizer(config.post_lr).minimize(self.loss)
    

    def get_log_posterior(self, stu_state, stu_action, code):
        log_post = self.sess.run(
            self.log_code_posterior, 
            feed_dict = {
                self.stu_state: stu_state,
                self.stu_action: stu_action,
                self.code: code
            }
        )

        return log_post
    

    def train(self, stu_state, stu_action, code):
        loss, log_post, _ = self.sess.run(
            [self.loss, self.mean_log_code_posterior, self.train_opt], 
            feed_dict = {
                self.stu_state: stu_state,
                self.stu_action: stu_action,
                self.code: code
            }
        )

        return loss, log_post
    
    
    def _get_posterior_dist(self, state, action):
        state_layer = state
        for dim in self.config.state_fc_dims:
            dense = tf.layers.Dense(dim, activation = self.config.activation)
            state_layer = dense(state_layer)
        
        cell = tf.keras.layers.GRU(self.config.hidden_dim, return_sequences = True, name = 'gru')
        state_layer = cell(state_layer)

        action_layer = action
        for dim in self.config.action_fc_dims:
            dense = tf.layers.Dense(dim, activation = self.config.activation)
            action_layer = dense(action_layer)
        
        state_action_layer = tf.concat([state_layer, action_layer], axis = 2)
        
        for dim in self.config.post_fc_dims:
            state_action_layer = tf.layers.dense(state_action_layer, dim, activation = self.config.activation)
        
        logit = tf.layers.dense(state_action_layer, self.config.num_code, activation = None)
        prob_dist = tf.nn.softmax(logit)
        
        return prob_dist
