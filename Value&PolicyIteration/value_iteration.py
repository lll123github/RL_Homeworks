import sys
sys.path.append('..')
from environment import *
import numpy as np
import collections
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')



def value_iteration(env_wrapper:DiscreteWrapper, gamma=0.5, max_iter=1000, theta=1e-5):
    env = env_wrapper.env
    n_bins = env_wrapper.ob_bins
    n_dims = 6  # theta_lr, theta_1, theta_2, d_theta_lr, d_theta_1, d_theta_2
    n_states = n_bins ** n_dims  # 例如 10^6 = 1,000,000
    n_actions = env_wrapper.a_bins
    state_shape = (n_bins,) * n_dims
    
    V = np.zeros(state_shape)
    policy = np.zeros(state_shape, dtype=int)
    for iter_idx in range(max_iter):
        max_reward=0.0
        min_reward=0.0
        V_last=V.copy()
        for s in np.ndindex(*state_shape):
            v = V[s]
            q = np.zeros(n_actions)
            for a in range(n_actions):
                env.state = env_wrapper.get_continuous_state(s)
                s_next, reward, done, _ = env_wrapper.step(a)
                value = reward + gamma * V[s_next]
                # with open('log/value_iteration.log', 'a') as log_file:
                #     log_file.write(f'State: {s}\n')
                #     log_file.write(f'Action: {a}\n')
                #     log_file.write(f'Next State: {s_next}\n')
                #     log_file.write(f'Reward: {reward}\n')
                #     log_file.write('-----------------------------------\n')
                q[a] = value
                max_reward = max(max_reward, reward)
                min_reward = min(min_reward, reward)
            #policy update
            policy[s]= np.argmax(q)
            #value update
            V[s] = np.max(q)
        max_value_change = np.abs(np.max(V) - np.max(V_last))
        min_value_change= np.abs(np.min(V) - np.min(V_last))
        print('Max value change:', max_value_change)
        print('Max reward:', max_reward)
        print('Min reward:', min_reward)
        value_change = np.max(np.abs(V - V_last))
        print('Value change:', value_change)
        if value_change < theta :
            print(f'Converged after {iter_idx} iterations.')

            break
    return V, policy
