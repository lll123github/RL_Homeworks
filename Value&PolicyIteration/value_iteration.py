import sys
sys.path.append('..')
from environment import *
import numpy as np
import collections
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')



def value_iteration(env_wrapper:DiscreteWrapper, gamma=0.1, max_iter=1000, theta=1e-4):
    env = env_wrapper.env
    n_bins = env_wrapper.ob_bins
    n_dims = 6  # theta_lr, theta_1, theta_2, d_theta_lr, d_theta_1, d_theta_2
    n_states = n_bins ** n_dims  # 例如 10^6 = 1,000,000
    n_actions = env_wrapper.a_bins
    state_shape = (n_bins,) * n_dims
    
    V = np.zeros(state_shape)
    
    for _ in range(max_iter):
        delta = 0
        for s in np.ndindex(*state_shape):
            v = V[s]
            max_value = -np.inf
            
            for a in range(n_actions):
                env.state = env_wrapper.get_continuous_state(s)
                s_next, reward, done, _ = env_wrapper.step(a)
                value = reward + gamma * (0 if done else V[s_next])
                if value > max_value:
                    max_value = value
            
            V[s] = max_value.item()
            delta = max(delta, abs(v - V[s]))
            print('delta:', delta)
        if delta < theta:
            
            break
    
    # 提取最优策略
    policy = np.zeros(state_shape, dtype=int)
    for s in np.ndindex(*state_shape):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            env.state = env_wrapper.get_continuous_state(s)
            s_next, reward, done, _ = env_wrapper.step(a)
            action_values[a] = reward + gamma * (0 if done else V[s_next].item())
        policy[s] = np.argmax(action_values)
    
    return V, policy
