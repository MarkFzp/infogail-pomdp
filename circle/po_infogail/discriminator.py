import tensorflow as tf
import numpy as np

class Discriminator:
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess

        with tf.variable_scope('Discriminator'):
            self.expert_state = tf.placeholder(tf.float32, [None, None, 2])
            self.expert_action = tf.placeholder(tf.float32, [None, None])
            self.stu_state = tf.placeholder(tf.float32, [None, None, 2])
            self.stu_action = tf.placeholder(tf.float32, [None, None])

            self.expert_score, self.stu_score = self._get_score(self.expert_state, self.expert_action, self.stu_state, self.stu_action)

            self.mean_expert_score = tf.reduce_mean(self.expert_score)
            self.mean_stu_score = tf.reduce_mean(self.stu_score)
            self.loss = self.mean_stu_score - self.mean_expert_score
            self.train_op = tf.train.AdamOptimizer(config.dis_lr).minimize(self.loss)
            
            self.param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator/')
            with tf.get_default_graph().control_dependencies([self.train_op]):
                self.clip_op = tf.group([w.assign(tf.clip_by_value(w, - config.dis_weight_clip, config.dis_weight_clip)) for w in self.param])
            
            self.train_op = tf.group(self.train_op, self.clip_op)


    def get_stu_score(self, stu_state, stu_action):
        stu_score = self.sess.run(
            self.stu_score,
            feed_dict = {
                self.stu_state: stu_state,
                self.stu_action: stu_action
            }
        )

        return stu_score
    

    def train(self, expert_state, expert_action, stu_state, stu_action):
        loss, expert_score, stu_score, _ = self.sess.run(
            [self.loss, self.mean_expert_score, self.mean_stu_score, self.train_op], 
            feed_dict = {
                self.expert_state: expert_state,
                self.expert_action: expert_action,
                self.stu_state: stu_state, 
                self.stu_action: stu_action
            }
        )

        return loss, expert_score, stu_score
    
    
    def _get_score(self, expert_state, expert_action, stu_state, stu_action):
        init = tf.initializers.random_normal(0.0, self.config.dis_weight_clip)
        
        expert_state_layer = expert_state
        stu_state_layer = stu_state
        for dim in self.config.state_fc_dims:
            dense = tf.layers.Dense(dim, activation = self.config.activation, kernel_initializer = init)
            expert_state_layer = dense(expert_state_layer)
            stu_state_layer = dense(stu_state_layer)
        
        cell = tf.keras.layers.GRU(self.config.hidden_dim, kernel_initializer = init, return_sequences = True, name = 'gru')
        expert_state_layer = cell(expert_state_layer)
        stu_state_layer = cell(stu_state_layer)

        expert_action_layer = expert_action[:, :, tf.newaxis]
        stu_action_layer = stu_action[:, :, tf.newaxis]
        for dim in self.config.action_fc_dims:
            dense = tf.layers.Dense(dim, activation = self.config.activation, kernel_initializer = init)
            expert_action_layer = dense(expert_action_layer)
            stu_action_layer = dense(stu_action_layer)

        expert_layer = tf.concat([expert_state_layer, expert_action_layer], axis = 2)
        stu_layer = tf.concat([stu_state_layer, stu_action_layer], axis = 2)

        for dim in self.config.dis_fc_dims:
            dense = tf.layers.Dense(dim, activation = self.config.activation, kernel_initializer = init)
            expert_layer = dense(expert_layer)
            stu_layer = dense(stu_layer)
        
        score_layer = tf.layers.Dense(1, activation = None, kernel_initializer = init)
        expert_score = score_layer(expert_layer)[:, :, 0]
        stu_score = score_layer(stu_layer)[:, :, 0]

        return expert_score, stu_score
