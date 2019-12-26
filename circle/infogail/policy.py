import tensorflow as tf
import numpy as np
import utils as U

class Policy:
    def __init__(self, config, sess, suffix):
        self.config = config
        self.sess = sess

        with tf.variable_scope('Policy_{}'.format(suffix)):
            self.state = tf.placeholder(tf.float32, [None, config.num_past_obs, 2])
            self.code = tf.placeholder(tf.int32, [None]) # tf.placeholder_with_default(tf.fill([tf.shape(self.state)[0]], -1), [None])
            self.action_spvs = tf.placeholder(tf.float32, [None])
            self.adv = tf.placeholder(tf.float32, [None])
            self.action_log_prob_old = tf.placeholder(tf.float32, [None])
            self.update_op = None

            self.state_ = tf.reshape(self.state, [-1, config.num_past_obs * 2])
            self.code_oh = tf.one_hot(self.code, depth = config.code_dim)
            if config.normalize_adv:
                self.adv = U.normalize(self.adv)
            
            self.state_encoding = self._encode_state(self.state_)
            
            self.code_encoding = self._encode_code(self.code_oh)

            self.state_code_encoding = self.state_encoding + self.code_encoding
            
            self.action_mean, self.action_std, self.action_log_std = self._get_action_dist_param(self.state_code_encoding)
            self.action_sample = self.action_mean + self.action_std * tf.random.normal(tf.shape(self.action_mean))
            self.action_greedy = self.action_mean
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
            self.train_op = tf.train.AdamOptimizer(config.policy_lr).minimize(self.loss)


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


    def sample_action(self, state, code):
        if self.config.greedy:
            action = self.sess.run(
                self.action_greedy, 
                feed_dict = {
                    self.state: state, 
                    self.code: code
                }
            )
        else:
            action = self.sess.run(
                self.action_sample, 
                feed_dict = {
                    self.state: state, 
                    self.code: code
                }
            ) 

        return action
    

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


    def _encode_state(self, state):
        state_layers = [state]
        for fc_dim in self.config.state_fc_dims:
            state_layers.append(tf.layers.dense(state_layers[-1], fc_dim, activation = self.config.activation))
        return state_layers[-1]


    def _encode_code(self, code):
        code_layers = [code]
        for fc_dim in self.config.code_fc_dims:
            code_layers.append(tf.layers.dense(code_layers[-1], fc_dim, activation = self.config.activation))
        return code_layers[-1]
    

    def _get_action_dist_param(self, state_code):
        policy_layer = state_code
        for dim in self.config.policy_fc_dims:
            policy_layer = tf.layers.dense(policy_layer, dim, activation = self.config.activation)
        
        if self.config.scale_action:
            action_mean = 3 * np.pi * tf.layers.dense(policy_layer, 1, activation = tf.nn.tanh)[:, 0] - np.pi / 2
        else:
            action_mean = tf.layers.dense(policy_layer, 1, activation = None)[:, 0]

        if self.config.policy_log_std_const:
            action_log_std = tf.constant(self.config.policy_log_std_init, name = 'log_std')
        else:
            action_log_std = tf.Variable(self.config.policy_log_std_init, trainable = True, name = 'log_std')
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
