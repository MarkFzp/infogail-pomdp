import tensorflow as tf
import numpy as np
import utils as U

class Policy:
    def __init__(self, config, sess, suffix):
        self.config = config
        self.sess = sess

        with tf.variable_scope('Policy_{}'.format(suffix)):
            self.state = tf.placeholder(tf.float32, [None, None, config.state_dim])
            self.batch_size = tf.shape(self.state)[0]
            self.traj_len = tf.shape(self.state)[1]
            self.code = tf.placeholder(tf.int32, [None])
            self.in_h_state = tf.placeholder_with_default(tf.zeros([self.batch_size, config.hidden_dim]), [None, config.hidden_dim])

            self.action_spvs = tf.placeholder(tf.float32, [None, None, config.action_dim])
            self.adv = tf.placeholder(tf.float32, [None, None])
            self.action_log_prob_old = tf.placeholder(tf.float32, [None, None])
            self.update_op = None
            self.global_step = tf.Variable(0, trainable = False)

            self.code_oh = tf.one_hot(self.code, depth = config.code_dim)
            if config.normalize_adv:
                self.adv = U.normalize(self.adv)
            
            self.state_encoding, self.out_h_state = self._encode_state(self.state, self.in_h_state)
            
            self.code_encoding = self._encode_code(self.code_oh)

            self.state_code_encoding = self._encode_state_code(self.state_encoding, self.code_encoding)
            
            self.action_mean, self.action_std, self.action_log_std = self._get_action_dist_param(self.state_code_encoding)
            self.action_sample = tf.clip_by_value(self.action_mean + self.action_std * tf.random.normal(tf.shape(self.action_mean)), 
                config.action_low, config.action_high)
            self.action_greedy = self.action_mean
            self.action_sample_1d = self.action_sample[:, -1]
            self.action_greedy_1d = self.action_greedy[:, -1]
            self.action_op = self.action_greedy_1d if config.greedy else self.action_sample_1d

            self.entropy_scaled = tf.reduce_sum(self.action_log_std) # 0.5 + tf.log((2 * np.pi) ** 0.5 * self.action_std) # in nats not in bits

            if config.greedy:
                self.bc_loss = tf.reduce_mean(0.5 * (self.action_spvs - self.action_greedy) ** 2) - config.entropy_loss_coef * self.entropy_scaled
            else:
                self.bc_loss = tf.reduce_mean(0.5 * (self.action_spvs - self.action_sample) ** 2) - config.entropy_loss_coef * self.entropy_scaled
            self.bc_train_op = tf.train.AdamOptimizer(config.policy_lr).minimize(self.bc_loss)

            self.action_log_prob = U.log_gaussian_prob_density(self.action_mean, self.action_log_std, self.action_spvs)
            self.ratio = tf.exp( # self.action_log_prob - self.action_log_prob_old)
                tf.clip_by_value(self.action_log_prob - self.action_log_prob_old, -np.inf, 10)
            )
            self.ratio_clipped = tf.clip_by_value(self.ratio, 1.0 - config.policy_clip_range, 1.0 + config.policy_clip_range)
            self.policy_reward_1 = self.ratio * self.adv
            self.policy_reward_2 = self.ratio_clipped * self.adv
            self.clipped_freq = tf.reduce_mean(tf.cast(self.policy_reward_1 > self.policy_reward_2, tf.float32))

            self.policy_reward = tf.reduce_mean(tf.minimum(
                self.policy_reward_1,
                self.policy_reward_2
            ))

            self.loss = -1 * (self.policy_reward + config.entropy_loss_coef * self.entropy_scaled)
            self.train_op = tf.train.AdamOptimizer(config.policy_lr).minimize(self.loss, global_step = self.global_step)


    def bc_train(self, state, code, action_spvs):
        bc_loss, _ = self.sess.run(
            [self.bc_loss, self.bc_train_op],
            feed_dict = {
                self.state: state,
                self.code: code,
                self.action_spvs: action_spvs
            }
        )

        return bc_loss


    def sample_action(self, state_2d, code, in_h_state):
        fd = {
            self.state: state_2d[:, np.newaxis, :], 
            self.code: code,
            self.in_h_state: in_h_state
        } if in_h_state is not None else {
            self.state: state_2d[:, np.newaxis, :], 
            self.code: code
        }

        action, out_hidden_state = self.sess.run(
            [self.action_op, self.out_h_state], 
            feed_dict = fd
        )

        return action, out_hidden_state
    
    
    def get_action_log_prob(self, state, code, action_spvs):
        action_log_prob = self.sess.run(
            self.action_log_prob, 
            feed_dict = {
                self.state: state,
                self.code: code,
                self.action_spvs: action_spvs
            }
        )

        return action_log_prob


    def train(self, state, code, action_spvs, adv, action_log_prob_old):
        loss, policy_reward, entropy, clipped_freq, _ = self.sess.run(
            [self.loss, self.policy_reward, self.entropy_scaled, self.clipped_freq, self.train_op], 
            feed_dict = {
                self.state: state,
                self.code: code, 
                self.action_spvs: action_spvs,
                self.adv: adv,
                self.action_log_prob_old: action_log_prob_old
            }
        )

        return loss, policy_reward, entropy, clipped_freq


    def _encode_state(self, state, in_h_state):
        state_layer = state
        for fc_dim in self.config.state_fc_dims:
            state_layer = tf.layers.dense(state_layer, fc_dim, activation = self.config.activation)
        
        cell = tf.keras.layers.GRU(self.config.hidden_dim, return_sequences = True, return_state = True, name = 'gru')
        state_encode, out_h_state = cell(state_layer, initial_state = in_h_state)
        return state_encode, out_h_state


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
    

    def _get_action_dist_param(self, encoding):
        policy_layer = encoding
        for dim in self.config.policy_fc_dims:
            policy_layer = tf.layers.dense(policy_layer, dim, activation = self.config.activation)
        
        if self.config.scale_action:
            action_mean = tf.layers.dense(policy_layer, self.config.action_dim, activation = tf.nn.tanh) * self.config.action_range
        else:
            action_mean = tf.layers.dense(policy_layer, self.config.action_dim, activation = None)

        if self.config.policy_log_std_mode == 'constant':
            action_log_std = tf.constant(self.config.start_log_std, name = 'log_std')
        elif self.config.policy_log_std_mode == 'variable':
            action_log_std = tf.Variable(self.config.start_log_std, trainable = True, name = 'log_std')
        elif self.config.policy_log_std_mode == 'anneal':
            action_log_std = tf.train.polynomial_decay(
                self.config.start_log_std,
                self.global_step,
                self.config.anneal_step, 
                self.config.end_log_std,
                power = 1.0
            )
        
        action_std = tf.exp(action_log_std)
        
        return action_mean, action_std, action_log_std
    

    def set_update_op(self, policy_vars, old_policy_vars, soft = False):
        if self.update_op is None:
            if soft:
                self.update_op = tf.group([v_t.assign(v_t * (1 - 0.2) + v * 0.2) for v_t, v in zip(old_policy_vars, policy_vars)])
            else:
                self.update_op = tf.group([v_t.assign(v) for v_t, v in zip(old_policy_vars, policy_vars)])
        else:
            raise Exception('duplicate update_op in policy_net')
    

    def run_update_op(self):
        # assert(self.update_op is not None)
        self.sess.run(self.update_op)
