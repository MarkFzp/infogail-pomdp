import numpy as np
import sys
import os

dir_name = os.path.dirname(__file__)

sys.path.append(os.path.join(dir_name, '../env/'))
import circle_theta as env


class Buffer:
    def __init__(self, config, expert_traj, expert_theta, policy):
        self.config = config
        self.policy = policy
        self.num_r, self.num_traj_each_r = expert_traj.shape[:2]
        assert((self.num_r, self.num_traj_each_r) == expert_theta.shape[:2])
        print('max traj len: ', config.max_traj_len)

        if config.num_past_obs > 1:
            traj_padding = np.zeros([config.num_past_obs - 1, 2])

        expert_exp_states = []
        expert_exp_actions = []
        expert_traj_lens = []
        for i in range(self.num_r):
            for j in range(self.num_traj_each_r):
                if config.num_past_obs > 1:
                    tj = np.concatenate([traj_padding, expert_traj[i, j]], axis = 0)
                else:
                    tj = expert_traj[i, j]
                th = expert_theta[i, j]
                l = len(th)
                for t in range(l):
                    state = tj[t: t + config.num_past_obs]
                    action = th[t]
                    expert_exp_states.append(state)
                    expert_exp_actions.append(action)
                expert_traj_lens.append(l)
        
        print('expert traj lens: ', expert_traj_lens)
        self.expert_exp_state_np = np.array(expert_exp_states)
        self.expert_exp_action_np = np.array(expert_exp_actions)

        self.num_expert_exp = len(self.expert_exp_state_np)
        assert(self.num_expert_exp == len(self.expert_exp_action_np))
        print('num of expert exp in buffer: ', self.num_expert_exp)

        self.empty_code = np.array([-1] * config.batch_size_exp)


    def sample_code(self):
        return np.random.randint(self.config.num_code, size = self.config.batch_size_exp) if not self.config.empty_code else self.empty_code


    def sample_expert_exp(self):
        expert_exp_idx = np.random.choice(self.num_expert_exp, size = self.config.batch_size_exp, replace = False)
        expert_exp_state_sampled = self.expert_exp_state_np[expert_exp_idx]
        expert_exp_action_sampled = self.expert_exp_action_np[expert_exp_idx]

        return expert_exp_state_sampled, expert_exp_action_sampled


    def sample_stu_traj(self):
        stu_traj_states = []
        stu_traj_actions = []
        # stu_traj_xys = []
        stu_traj_code_np = np.random.randint(0, self.config.num_code, size = self.config.batch_size_traj)

        init_state = np.zeros([self.config.batch_size_traj, self.config.num_past_obs, 2])
        init_xy = np.zeros([self.config.batch_size_traj, 2])
        curr_state = init_state
        curr_xy = init_xy
        
        # rollout trajectory
        for _ in range(self.config.max_traj_len):
            action_sampled = self.policy.sample_action(curr_state, stu_traj_code_np)
            stu_traj_states.append(curr_state)
            stu_traj_actions.append(action_sampled)
            # stu_traj_xys.append(curr_xy)

            next_xy = env.next_xy(curr_xy, action_sampled)
            next_xy_3d = next_xy[:, np.newaxis, :]
            if self.config.num_past_obs == 1:
                curr_state = next_xy_3d
            else:
                curr_state = np.concatenate([curr_state[:, 1:, :], next_xy_3d], axis = 1)
            curr_xy = next_xy
        
        stu_traj_state_np = np.stack(stu_traj_states, axis = 1)
        stu_traj_action_np = np.stack(stu_traj_actions, axis = 1)

        return stu_traj_state_np, stu_traj_action_np, stu_traj_code_np


    def sample_stu_exp(self, state_traj, action_traj, code_traj):
        exp_idx = (
            np.random.randint(self.config.batch_size_traj, size = self.config.batch_size_exp), 
            np.random.randint(self.config.max_traj_len, size = self.config.batch_size_exp)
        )

        state_exp = state_traj[exp_idx]
        action_exp = action_traj[exp_idx]
        code_exp = code_traj[exp_idx[0]]

        return state_exp, action_exp, code_exp, exp_idx
    

    def traj_to_exp(self, state_traj, action_traj, code_traj):
        portion = 1

        state_exp = state_traj[:, :self.config.max_traj_len // portion, ...].reshape([-1, self.config.num_past_obs, 2])
        action_exp = action_traj[:, :self.config.max_traj_len // portion, ...].reshape([-1, 2])
        code_exp = np.repeat(code_traj, self.config.max_traj_len // portion)
        exp_idx = (np.repeat(np.arange(self.config.batch_size_traj), self.config.max_traj_len // portion), 
            np.repeat([np.arange(self.config.max_traj_len // portion)], self.config.batch_size_traj, axis = 0).ravel())

        return state_exp, action_exp, code_exp, exp_idx
