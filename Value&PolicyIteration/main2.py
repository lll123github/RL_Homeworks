import sys
sys.path.append('..')
from environment import *
from policy_iteration import *
from value_iteration import *
import numpy as np
import collections
import matplotlib.pyplot as plt
import warnings
import time
from figure import show_res
warnings.filterwarnings('ignore')


import platform

def play_beep(frequency=200, duration=100):
    if platform.system() == 'Windows':
        import winsound
        # 播放系统提示音
        winsound.Beep(frequency, duration)
    elif platform.system() == 'Linux':
        import os
        # 使用系统命令播放提示音
        os.system('echo -e "\a"')  # \a 是响铃字符

def train(index,env_wrapper:DiscreteWrapper):
    if index ==0:
        print("Running Policy Iteration...")
        V, policy = policy_iteration(env_wrapper)
        print("Policy Iteration Completed!")
        np.save( 'policy_pi.npy',policy,)
        np.save('V_pi.npy', V)
    elif index == 1:
        print("\nRunning Value Iteration...")
        V, policy = value_iteration(env_wrapper)
        print("Value Iteration Completed!")
        np.save('policy_vi.npy', policy)
        np.save('V_vi.npy', V)
    elif index == 2:
        # print("\nRunning Q Learning...")
        # Q, policy_q = q_learning(env_wrapper)
        # print("Q Learning Completed!")
        # np.save('policy_q.npy', policy_q)
        # np.save('Q.npy', Q)
        pass
    else:
        raise ValueError("Invalid index. Use 0 for Policy Iteration, 1 for Value Iteration, or 2 for Q Learning.")
    return
    
if __name__ == "__main__":
    env = Environment()
    env_wrapper = DiscreteWrapper(env)  # 离散化为 5 bins

    # P, R = compute_model_matrices(env_wrapper, n_samples=1000)
    begin_time = time.time()
    index=1  # 0: Policy Iteration, 1: Value Iteration, 2: Q Learning
    mapping_method={0: 'Policy Iteration', 1: 'Value Iteration', 2: 'Q Learning'}
    mapping_file={0: 'policy_pi.npy', 1: 'policy_vi.npy', 2: 'policy_q.npy'}




    
    train(index,env_wrapper)  # 0: Policy Iteration, 1: Value Iteration, 2: Q Learning






    
    play_beep(300,1000)
    cost = time.time()-begin_time
    print('cost time:', cost)
    
    
    # # 比较两种算法的策略是否一致
    # print("\nPolicy Difference:", np.sum(policy_pi != policy_vi))

    # V_pi = np.load('V_pi.npy')
    # policy_pi = np.load('policy_pi.npy')
    
    # np.save('policy_q.npy', policy_q)
    policy=np.load(mapping_file[index])

    state = env.reset()
    print(env.state)
    l = 1000
    history = np.zeros((l, 5))  # [theta_LR, theta_1, theta_2, action, reward]
    for t in range(l):
        s = env_wrapper.observation(state)#离散化
        action_idx =policy[s]
        # print(action_idx)
        action = env_wrapper.action(action_idx)#离散化
        print(action)
        next_state, reward, terminated, _ = env.step(action) #注意这里使用的是env的step方法，是因为这里仿真需要从近乎连续的状态空间中采样动作，而不是离散化的动作空间
        # print(env.state)
        # 确保状态值是数值类型
        theta_LR = float(next_state['theta_lr'][0][0])
        theta_1 = float(next_state['theta_1'][0][0])
        theta_2 = float(next_state['theta_2'][0][0])
        action = float(action['u_lr'][0])
        reward = float(reward)
        history[t] = np.array([theta_LR, theta_1, theta_2, action, reward])
        env.render()
        if terminated:
            state = env.reset()
            play_beep()

    env.close()
    show_res(history)