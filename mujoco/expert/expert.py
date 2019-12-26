import gym
import numpy as np
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2
import os
from tqdm import tqdm
import pdb

from config import config


def main():
    if config.n_cpu == 1:
        env = DummyVecEnv([lambda: gym.make(config.env_name, forward_reward_weight = config.forward_reward_weight, \
            ctrl_cost_weight = config.ctrl_cost_weight, terminate_when_unhealthy = config.terminate_when_unhealthy)])
    else:
        env = SubprocVecEnv([lambda: gym.make(config.env_name, forward_reward_weight = config.forward_reward_weight, \
            ctrl_cost_weight = config.ctrl_cost_weight, terminate_when_unhealthy = config.terminate_when_unhealthy) for i in range(config.n_cpu)])

    model = PPO2(MlpPolicy, env, verbose = config.verbose)
    
    if config.load and os.path.exists(config.ckpt_path + '.zip'):
        model = PPO2.load(config.ckpt_path)
        print("ckpt loaded !!!!")

    if not config.test:
        model.learn(total_timesteps = config.itr)
        model.save(config.ckpt_path)
        print('ckpt saved')

    else:
        trajs_state = []
        trajs_action = []
        trajs_reward = []

        # aaa = np.load('Hopper-v3_backward_action.npy')

        for _ in tqdm(range(config.sample_size)):
            prestop = False
            traj_states = []
            traj_actions = []
            traj_rewards = []
            
            obs = env.reset()

            for t in range(config.n_step):
                action, _states = model.predict(obs)
                # action = aaa[_][t][np.newaxis, :]
                
                traj_states.append(obs[0])
                traj_actions.append(action[0])

                obs, rewards, dones, info = env.step(action)
                # print(obs, rewards, dones, info)
                # env.render()
                traj_rewards.append(rewards[0])
                # traj_rewards.append(info[0]['backward_reward'])

                if True in dones:
                    prestop = True
                    break
            
            if not prestop:
                trajs_state.append(traj_states)
                trajs_action.append(traj_actions)
                trajs_reward.append(np.sum(traj_rewards))
        
        print(np.mean(trajs_reward))


        traj_state_np = np.array(trajs_state)
        traj_action_np = np.array(trajs_action)

        np.save('{}_{}_state'.format(config.env_name, config.task), traj_state_np)
        np.save('{}_{}_action'.format(config.env_name, config.task), traj_action_np)
        print(traj_state_np.shape)
        print(traj_action_np.shape)



if __name__ == "__main__":
    main()
