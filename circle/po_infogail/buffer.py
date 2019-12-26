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
        self.num_r, self.num_traj_each_r, self.traj_len = expert_traj.shape[:3]
        assert((self.num_r, self.num_traj_each_r, self.traj_len) == expert_theta.shape[:3])
        print('max traj len: ', config.max_traj_len)
        print('len of traj in npy: ', self.traj_len)

        self.expert_traj = expert_traj.reshape([-1, *expert_traj.shape[2:]])
        self.expert_theta = expert_theta.reshape([-1, *expert_theta.shape[2:]])
        self.num_expert_traj = len(self.expert_traj)
        print('num expert traj: ', self.num_expert_traj)

        self.empty_code = np.array([-1] * config.batch_size_traj)


    def sample_expert_traj(self):
        expert_traj_idx = np.random.choice(self.num_expert_traj, size = self.config.batch_size_traj, replace = False)
        expert_traj_state_sampled = self.expert_traj[expert_traj_idx]
        expert_traj_action_sampled = self.expert_theta[expert_traj_idx]

        return expert_traj_state_sampled, expert_traj_action_sampled
    

    def sample_code(self):
        return np.random.randint(self.config.num_code, size = self.config.batch_size_traj) if not self.config.empty_code else self.empty_code


    def sample_stu_traj(self):
        stu_traj_states = []
        stu_traj_actions = []
        stu_traj_code_np = self.sample_code()

        init_h_state = None
        init_state = np.zeros([self.config.batch_size_traj, 2])
        curr_h_state = init_h_state
        curr_state = init_state
        
        # rollout trajectory
        for _ in range(self.config.max_traj_len):
            action_sampled, curr_h_state = self.policy.sample_action(curr_state, stu_traj_code_np, curr_h_state)
            stu_traj_states.append(curr_state)
            stu_traj_actions.append(action_sampled)

            next_state = env.next_xy(curr_state, action_sampled)
            curr_state = next_state
        
        stu_traj_state_np = np.stack(stu_traj_states, axis = 1)
        stu_traj_action_np = np.stack(stu_traj_actions, axis = 1)

        return stu_traj_state_np, stu_traj_action_np, stu_traj_code_np
    

    def traj_to_exp(self, state_traj, action_traj, code_traj):
        portion = 1

        state_exp = state_traj[:, :self.config.max_traj_len // portion, ...].reshape([-1, self.config.num_past_obs, 2])
        action_exp = action_traj[:, :self.config.max_traj_len // portion, ...].reshape([-1, 2])
        code_exp = np.repeat(code_traj, self.config.max_traj_len // portion)
        exp_idx = (np.repeat(np.arange(self.config.batch_size_traj), self.config.max_traj_len // portion), 
            np.repeat([np.arange(self.config.max_traj_len // portion)], self.config.batch_size_traj, axis = 0).ravel())

        return state_exp, action_exp, code_exp, exp_idx
