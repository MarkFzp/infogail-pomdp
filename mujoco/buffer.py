import numpy as np
import sys
import os

class Buffer:
    def __init__(self, config, expert_traj, expert_theta, policy, env):
        self.config = config
        self.policy = policy
        self.env = env

        self.expert_traj = expert_traj
        self.expert_theta = expert_theta
        self.num_expert_traj = len(self.expert_traj)
        print('num expert traj: ', self.num_expert_traj)

        # self.empty_code = np.array([-1] * config.batch_size_traj)


    def sample_expert_traj(self):
        expert_traj_idx = np.random.choice(range(2000, 4000), size = self.config.batch_size_traj, replace = False)
        expert_traj_state_sampled = self.expert_traj[expert_traj_idx]
        expert_traj_action_sampled = self.expert_theta[expert_traj_idx]

        return expert_traj_state_sampled, expert_traj_action_sampled
    

    # def sample_code(self):
    #     return np.random.randint(self.config.num_code, size = self.config.batch_size_traj) if not self.config.empty_code else self.empty_code


    def sample_stu_traj(self):
        stu_traj_states = []
        stu_traj_actions = []
        stu_traj_code_np = np.random.randint(self.config.num_code, size = self.config.batch_size_traj)

        init_h_state = None
        init_state = self.env.reset()
        curr_h_state = init_h_state
        curr_state = init_state[:, :self.config.state_dim]

        forward_rewards = []
        stand_rewards = []
        backward_rewards = []
        
        # rollout trajectory
        for _ in range(self.config.max_traj_len):
            action_sampled, curr_h_state = self.policy.sample_action(curr_state, stu_traj_code_np, curr_h_state)
            stu_traj_states.append(curr_state)
            stu_traj_actions.append(action_sampled)

            next_state, reward, done, info = self.env.step(action_sampled)
            curr_state = next_state[:, :self.config.state_dim]

            forward_rewards.append(reward)
            stand_rewards.append([inf['stand_reward'] for inf in info])
            backward_rewards.append([inf['backward_reward'] for inf in info])
        
        stu_traj_state_np = np.stack(stu_traj_states, axis = 1)
        stu_traj_action_np = np.stack(stu_traj_actions, axis = 1)

        forward_reward_np = np.sum(np.array(forward_rewards), axis = 0)
        stand_reward_np = np.sum(np.array(stand_rewards), axis = 0)
        backward_reward_np = np.sum(np.array(backward_rewards), axis = 0)

        return stu_traj_state_np, stu_traj_action_np, stu_traj_code_np, forward_reward_np, stand_reward_np, backward_reward_np
