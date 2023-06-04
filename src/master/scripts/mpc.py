#! /usr/bin/env python3

import torch
import numpy as np

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'


class MPController():
    def __init__(self, model, traject_num=100000, horizon=10, hold=False):
        self.u_dimension = 3

        self.model = model
        self.N = traject_num
        self.T = horizon
        self.hold = hold

        self.internal = 0

    def get_action(self, x_ref_seq, y_ref_seq, z_ref_seq, state):
        """
        evaluate N trajectories for T time horizon
        """

        action_limit = 150
        action_sample_step = 5

        action_group1 = np.arange(start=-action_limit, stop=action_limit, step=action_sample_step, dtype=float)
        action_group2 = np.arange(start=-action_limit-100, stop=action_limit+100, step=action_sample_step, dtype=float)

        with torch.no_grad():

            state0 = torch.tensor(state, dtype=torch.float32).repeat(self.N, 1).to(device)                  # N * 1 * x_dimension
            states = torch.zeros(self.N, self.T + 1, state.shape[-1], dtype=torch.float32).to(device)       # N * T+1 * x_dimension
            states[:, 0, :] = state0

            rewards = torch.zeros(self.N, self.T, dtype=torch.float32).to(device)       # N * T

            actions = torch.zeros(self.N, self.u_dimension).to(device)                           # N * u_dimension

            # generate N candidate action sequences of T steps
            if self.hold:  # action is constant for all time.
                action_candidates = torch.tensor(np.random.choice(action_group1, size=(self.N,1, self.u_dimension), replace=True), dtype=float).repeat(1,self.T, 1).to(device)
            else:
                action_candidates = torch.tensor(np.random.choice(action_group1, size=(self.N, self.T, self.u_dimension), replace=True), dtype=float).to(device)              # N * T * u_dimension
                action_candidates[:, :, 2] = torch.tensor(np.random.choice(action_group2, size=(self.N, self.T), replace=True), dtype=float).to(device)

            for t in range(self.T):
                # take actions of step t from candidate action sequences
                action_batch = action_candidates[:, t, :]       # N * u_dimension
                # print('action_batch',action_batch)

                # take states of step t from candidate action sequences
                state_batch = states[:, t, :]           # N * x_dimension
                # genarate states of step t+1 with states and actions of step t
                states[:, t + 1, :] = (state_batch + self.model.predict_cuda(state_batch, action_batch))
                # states[:, t + 1, 0] = state_batch[:, 0] + 0.001 * action_batch[:, 0] - 0.001 * action_batch[:, 1]
                # states[:, t + 1, 1] = state_batch[:, 1] + 0.001 * action_batch[:, 2] - 0.001 * action_batch[:, 3]

                # reward
                rewards[:, t].add_(
                    -(
                            ((states[:,t+1,0] - x_ref_seq[t]) ** 2 + (states[:,t+1,1] - y_ref_seq[t]) ** 2 + (states[:,t+1,2] - z_ref_seq[t]) ** 2) * 1 +
                            (
                                    (states[:,t+1,0] - states[:,t,0])**2 +
                                    (states[:,t+1,1] - states[:,t,1])**2 +
                                    (states[:,t+1,2] - states[:,t,2])**2
                            )
                    ))

            # TODO compute rewards
            cumulative_reward = torch.sum(rewards, dim=1)
            # print('cumulative_reward',cumulative_reward)

            # Find the index of best action
            best = torch.argmax(cumulative_reward)

            actions_seq = action_candidates[best, :, :]
            if actions_seq.shape[1] > 4:
                best_action = actions_seq[0][0:4]
            else:
                best_action = actions_seq[0]

            self.last_action = best_action
            self.internal += 1
            # print('best', best_action)
            return best_action.cpu().numpy()


if __name__=="__main__":

    model = 0
    mpc = MPController(model)
