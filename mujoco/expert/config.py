from easydict import EasyDict as edict 
config = edict()

config.task = 'stand'
config.env_name = 'Hopper-v3'

if config.task == 'forward':
    config.forward_reward_weight = 1.0
    config.ctrl_cost_weight = 1e-3
elif config.task == 'stand':
    config.forward_reward_weight = 0.0
    config.ctrl_cost_weight = 1.0
elif config.task == 'backward':
    config.forward_reward_weight = -1.0
    config.ctrl_cost_weight = 1e-3
else:
    raise NotImplementedError

config.terminate_when_unhealthy = False
config.test = True
config.load = True
config.ckpt_path = config.task + '_ckpt'

config.itr = 5000000
config.sample_size = 2000

config.verbose = 1
config.n_cpu = 1
config.n_step = 128 # 1000, 128, 200
config.n_minibatch = 8
config.n_optepoch = 4
