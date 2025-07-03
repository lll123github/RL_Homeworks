# 需要将vscode的执行目录设置为当前文件所在目录，否则无法导入环境模块
import sys
sys.path.append('..')
from environment import *
import numpy as np
import collections
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

def policy_iteration(env_wrapper:DiscreteWrapper, gamma=0.99, max_iter=1000, theta=1e-4):
    env = env_wrapper.env
    n_bins = env_wrapper.ob_bins
    n_dims = 6  # theta_lr, theta_1, theta_2, d_theta_lr, d_theta_1, d_theta_2
    n_states = n_bins ** n_dims  # 例如 10^6 = 1,000,000
    n_actions = env_wrapper.a_bins
    state_shape = (n_bins,) * n_dims
    
    # 初始化值函数和策略
    V = np.zeros(state_shape)
    policy = np.random.randint(0, n_actions, size=state_shape)
    
    for iter_idx in range(max_iter):
        print('Iteration:', iter_idx)
        # 策略评估
        while True:
            delta = 0
            for s in np.ndindex(*state_shape):
                v = V[s]
                a = policy[s]
                # action = env_wrapper.get_action_from_idx(a)
                
                # 执行动作，得到下一个状态和奖励
                env.state = env_wrapper.get_continuous_state(s)
                s_next, reward, done, iter_idx = env_wrapper.step(a)
                
                # 更新值函数
                V[s] = reward + gamma * (0 if done else V[s_next].item())
                delta = max(delta, abs(v - V[s]))

            
                
            if delta < theta:
                print('delta:', delta)
                break
        
        # 策略改进
        policy_stable = True
        for s in np.ndindex(*state_shape):
            old_action = policy[s]
            action_values = np.zeros(n_actions)
            
            for a in range(n_actions):
                
                env.state = env_wrapper.get_continuous_state(s)
                s_next, reward, done, _ = env_wrapper.step(a)
                action_values[a] = reward + gamma * (0 if done else V[s_next].item())
            
            # 选择最优动作
            policy[s] = np.argmax(action_values)
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break

    
    return V, policy