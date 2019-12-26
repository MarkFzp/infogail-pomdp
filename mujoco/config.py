from easydict import EasyDict as edict
import tensorflow as tf
import numpy as np
import os

dir_name = os.path.dirname(__file__)

config = edict()

config.greedy = False
config.gpu = False
config.sess_nan_test = False
config.mode = 'train'

config.activation = tf.nn.elu
config.normalize_adv = True
config.scale_action = True

config.itr = 500
config.test_itr = 32
config.max_traj_len = 128
config.update_period = 5
config.inner_itr_1 = 2
config.inner_itr_2 = 5
config.print_itr = 5
config.save_itr = 10
config.batch_size_traj = config.n_cpu = 8
if config.mode == 'render':
    config.batch_size_traj = config.n_cpu = 1

config.save_path = os.path.join(dir_name, './ckpt/stand.ckpt')
config.load_path = os.path.join(dir_name, './ckpt/stand.ckpt')

config.expert_traj_prefix = os.path.join(dir_name, 'expert', 'Hopper-v3')

config.state_dim = 5
config.action_dim = 3
config.code_dim = config.num_code = 3
config.action_range = 1.0
config.action_high = 1.0
config.action_low = -1.0

config.gamma = 0.99
config.lam = 0.95

config.hidden_dim = 128
config.state_code_arch = 'add'
config.state_fc_dims = [128, 128]
config.code_fc_dims = [128]
config.action_fc_dims = [32]

config.policy_lr = 2e-4
config.policy_clip_range = 0.2
config.policy_log_std_init = -1.0
config.entropy_loss_coef = 0.0
config.policy_fc_dims = [128]

config.policy_log_std_mode = 'variable'
config.start_log_std = -2.0
config.end_log_std = -2.0
config.anneal_step = 10000

config.value_lr = 5e-4
config.value_clip_range = 1.0
config.value_fc_dims = [128, 128] # [128]

config.dis_lr = 2e-4
config.dis_coef = 1.0
config.dis_weight_clip = 0.01
config.dis_fc_dims = [128]

config.post_lr = 2e-4
config.post_coef = 0.0
config.post_fc_dims = [128]
