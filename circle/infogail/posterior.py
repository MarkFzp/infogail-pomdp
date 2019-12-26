import tensorflow as tf
import numpy as np

class Posterior:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

        with tf.name_scope('Posterior'):
            self.stu_state = tf.placeholder(tf.float32, [None, config.num_past_obs, 2])
            self.stu_action = tf.placeholder(tf.float32, [None])
            self.code = tf.placeholder(tf.int32, [None])

            self.stu_state_ = tf.reshape(self.stu_state, [-1, config.num_past_obs * 2])
            self.stu_action_ = tf.reshape(self.stu_action, [-1, 1])

            self.code_posterior_dist = self._get_posterior_dist(self.stu_state_, self.stu_action_)
            self.code_posterior_idx = tf.stack([tf.range(tf.shape(self.code)[0], dtype = tf.int32), self.code], axis = 1)
            self.log_code_posterior = tf.log(tf.gather_nd(self.code_posterior_dist, self.code_posterior_idx)) / tf.log(2.0)
            
            self.loss = -1 * config.post_coef * tf.reduce_mean(self.log_code_posterior)
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
        loss, _ = self.sess.run(
            [self.loss, self.train_opt], 
            feed_dict = {
                self.stu_state: stu_state,
                self.stu_action: stu_action,
                self.code: code
            }
        )

        return loss
    
    def _get_posterior_dist(self, state, action):
        state_action = tf.concat([state, action], axis = 1)
        state_action_layer = state_action
        for dim in self.config.post_fc_dims:
            state_action_layer = tf.layers.dense(state_action_layer, dim, activation = self.config.activation)
        
        logit = tf.layers.dense(state_action_layer, self.config.num_code, activation = None)
        prob_dist = tf.nn.softmax(logit)
        
        return prob_dist
