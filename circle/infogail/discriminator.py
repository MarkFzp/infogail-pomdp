import tensorflow as tf
import numpy as np

class Discriminator:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

        with tf.variable_scope('Discriminator'):
            self.expert_state = tf.placeholder(tf.float32, [None, config.num_past_obs, 2])
            self.expert_action = tf.placeholder(tf.float32, [None])
            self.stu_state = tf.placeholder(tf.float32, [None, config.num_past_obs, 2])
            self.stu_action = tf.placeholder(tf.float32, [None])

            self.expert_state_ = tf.reshape(self.expert_state, [-1, config.num_past_obs * 2])
            self.expert_action_ = tf.reshape(self.expert_action, [-1, 1])
            self.stu_state_ = tf.reshape(self.stu_state, [-1, config.num_past_obs * 2])
            self.stu_action_ = tf.reshape(self.stu_action, [-1, 1])

            self.expert_score, self.stu_score = self._get_score(self.expert_state_, self.expert_action_, self.stu_state_, self.stu_action_)

            self.mean_expert_score = tf.reduce_mean(self.expert_score)
            self.mean_stu_score = tf.reduce_mean(self.stu_score)
            self.loss = self.mean_stu_score - self.mean_expert_score
            self.train_op = tf.train.AdamOptimizer(config.dis_lr).minimize(self.loss)
            
            self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator/')
            with tf.get_default_graph().control_dependencies([self.train_op]):
                self.clip_op = tf.group(*[w.assign(tf.clip_by_value(w, - config.dis_weight_clip, config.dis_weight_clip)) for w in self.param])
            
            self.train_op = tf.group(self.train_op, self.clip_op)


    def get_stu_score(self, stu_state, stu_action):
        stu_score = self.sess.run(
            self.stu_score,
            feed_dict={
                self.stu_state: stu_state,
                self.stu_action: stu_action
            }
        )

        return stu_score
    

    def train(self, expert_state, expert_action, stu_state, stu_action):
        loss, expert_score, stu_score, _ = self.sess.run(
            [self.loss, self.mean_expert_score, self.mean_stu_score, self.train_op], 
            feed_dict={
                self.expert_state: expert_state,
                self.expert_action: expert_action,
                self.stu_state: stu_state, 
                self.stu_action: stu_action
            }
        )

        return loss, expert_score, stu_score
    
    
    def _get_score(self, expert_state, expert_action, stu_state, stu_action):
        init = tf.initializers.random_normal(0.0, self.config.dis_weight_clip)
        expert_layer = tf.concat([expert_state, expert_action], axis = 1)
        stu_layer = tf.concat([stu_state, stu_action], axis = 1)

        for dim in self.config.dis_fc_dims:
            dense = tf.layers.Dense(dim, activation = self.config.activation, kernel_initializer = init)
            expert_layer = dense(expert_layer)
            stu_layer = dense(stu_layer)
        
        score_layer = tf.layers.Dense(1, activation = None, kernel_initializer = init)
        expert_score = score_layer(expert_layer)[:, 0]
        stu_score = score_layer(stu_layer)[:, 0]

        return expert_score, stu_score
