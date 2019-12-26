from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
import os

dir_name = os.path.dirname(__file__)

config = edict()

config.greedy = False
config.mode = 'draw' # 'bc', 'draw', 'gail'
config.gpu = False
config.sess_nan_test = False

config.activation = tf.nn.elu
config.normalize_adv = True
config.scale_action = False

config.itr = 10000
config.warm_up_itr = 100
config.inner_itr_1 = 10
config.inner_itr_2 = 10
config.update_period = 1
config.print_itr = 10
config.save_itr = 10
config.batch_size_traj = 32
config.batch_size_exp = 128

config.save_path = os.path.join(dir_name, './traj_7_obs1_test2/traj_7_obs1_test2.ckpt-200')
config.load_path = os.path.join(dir_name, './traj_7_obs1_test2/traj_7_obs1_test2.ckpt-200')
config.draw_traj_dir = os.path.join(dir_name, './draw/')
config.draw_extension = 'png'

config.expert_traj_path = os.path.join(dir_name, '../env/traj_7.npy')
config.max_traj_len = 33
config.x_range = 10
config.y_range = 20

config.empty_code = False
config.code_dim = config.num_code = 2
config.num_past_obs = 1

config.gamma = 0.99
config.lam = 0.95

config.state_fc_dims = [128, 128]
config.code_fc_dims = [128]

config.policy_lr = 2e-4
config.policy_clip_range = 0.15
config.policy_log_std_init = -1.0
config.policy_log_std_const = True
config.entropy_loss_coef = 0.0
config.policy_fc_dims = [128]
# assert(config.state_fc_dims[-1] == config.code_fc_dims[-1])

config.value_lr = 5e-4
config.value_clip_range = 0.3
config.value_fc_dims = [128]

config.dis_lr = 2e-4
config.dis_weight_clip = 0.01
config.dis_fc_dims = [128, 128]

config.post_lr = 2e-4
config.post_coef = 10.0
config.post_fc_dims = [128, 128]

assert(config.batch_size_traj * config.max_traj_len >= config.batch_size_exp)
