import sys
sys.path.append('..')
from environment import *
from policy_iteration import *
import numpy as np
import collections
import matplotlib.pyplot as plt
import warnings
import time
from figure import show_res
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    env = Environment()
    env_wrapper = DiscreteWrapper(env)  # 离散化为 5 bins

    # P, R = compute_model_matrices(env_wrapper, n_samples=1000)
    begin_time = time.time()
    print("Running Policy Iteration...")
    V_pi, policy_pi = policy_iteration(env_wrapper)
    print("Policy Iteration Completed!")
    np.save( 'policy_pi.npy',policy_pi,)
    
    
    
    # print("\nRunning Value Iteration...")
    # V_vi, policy_vi = value_iteration(env_wrapper)
    # print("Value Iteration Completed!")



    # print("Running Q Learning...")
    # # Q, policy_q = q_learning(env_wrapper)

    # print("Q Learning Completed!")

    cost = time.time()-begin_time
    print('cost time:', cost)
    
    # # 比较两种算法的策略是否一致
    # print("\nPolicy Difference:", np.sum(policy_pi != policy_vi))

    # V_pi = np.load('V_pi.npy')
    # policy_pi = np.load('policy_pi.npy')
    
    # np.save('policy_q.npy', policy_q)
    policy_q = np.load('policy_q.npy')

    state = env.reset()
    print(env.state)
    l = 1000
    history = np.zeros((l, 5))  # [theta_LR, theta_1, theta_2, action, reward]
    for t in range(l):
        s = env_wrapper.observation(state)
        action_idx =policy_q[s]
        # print(action_idx)
        action = env_wrapper.action(action_idx)
        # print(action)
        next_state, reward, terminated, _ = env_wrapper.step(action_idx)
        # print(env.state)
        # 确保状态值是数值类型
        theta_LR = float(next_state['theta_lr'][0][0])
        theta_1 = float(next_state['theta_1'][0][0])
        theta_2 = float(next_state['theta_2'][0][0])
        action = float(action['u_lr'][0][0])
        reward = float(reward)
        history[t] = np.array([theta_LR, theta_1, theta_2, action, reward])
        env.render()
        if terminated:
            state = env.reset()
    env.close()
    show_res(history)