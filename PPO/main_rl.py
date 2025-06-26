import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import scipy.signal
import scipy.linalg

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1):
        super().__init__(observation_space, features_dim)
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 8, bias=False)
        self.fc3 = nn.Linear(8, 1, bias=False)
        # self.end_fc = nn.Linear(3, 1, bias=False)
    
    def forward(self, observations):
        s1 = torch.relu(self.fc1(observations))
        s2 = torch.relu(s1)
        s3 = torch.relu(s2)
        # s4 = torch.cat(s3, dim=1)
        # return self.end_fc(s4)
        return self.fc3(s3)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_extractor = CustomNetwork(self.observation_space)

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 参数定义
        self.m_1 = 0.9
        self.m_2 = 0.1
        self.r = 0.0335
        self.L_1 = 0.126
        self.L_2 = 0.390
        self.l_1 = self.L_1 / 2
        self.l_2 = self.L_2 / 2
        self.g = 9.8
        self.I_1 = (1 / 12) * self.m_1 * self.L_1 ** 2
        self.I_2 = (1 / 12) * self.m_2 * self.L_2 ** 2

        # 状态、动作空间定义
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32)
        # 状态初始化

        self.steps = 0
        self.mx_step = 30
        self.writer = None
        self.step_count = 0

        self.Ts = 0.01
        self._build_dynamics()
        self.reset()

    def _build_dynamics(self):
        # 构建 p 和 q 矩阵
        p = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [(self.r / 2) * (self.m_1 * self.l_1 + self.m_2 * self.L_1), (self.r / 2) * (self.m_1 * self.l_1 + self.m_2 * self.L_1), self.m_1 * self.l_1 ** 2 + self.m_2 * self.L_1 ** 2 + self.I_1, self.m_2 * self.L_1 * self.l_2],
            [(self.r / 2) * self.m_2 * self.l_2, (self.r / 2) * self.m_2 * self.l_2, self.m_2 * self.L_1 * self.l_2, self.m_2 * self.l_2 ** 2 + self.I_2]
        ])

        q = np.zeros((4, 10))
        q[0, 8] = 1
        q[1, 9] = 1
        q[2, 2] = (self.m_1 * self.l_1 + self.m_2 * self.L_1) * self.g
        q[3, 3] = self.m_2 * self.g * self.l_2

        temp = np.linalg.inv(p) @ q
        A_top = np.hstack((np.zeros((4, 4)), np.eye(4)))
        A_bottom = temp[:, 0:8]
        A = np.vstack((A_top, A_bottom))

        B_top = np.zeros((4, 2))
        B_bottom = temp[:, 8:10]
        B = np.vstack((B_top, B_bottom))

        # 离散化系统
        C = np.eye(4, 8)
        D = np.zeros((4, 2))
        sys_d = scipy.signal.cont2discrete((A, B, C, D), self.Ts)
        self.G, self.H = sys_d[0].astype(np.float32), sys_d[1].astype(np.float32)

        # 控制器设计
        Q = np.diag([51.2938] * 2 + [32.8281, 131.3123] + [51.2938] * 2 + [131.3123] * 2)
        R = 0.0005 * np.eye(2)
        X = scipy.linalg.solve_discrete_are(self.G, self.H, Q, R)
        self.K = np.linalg.inv(self.H.T @ X @ self.H + R) @ (self.H.T @ X @ self.G)

    def reset(self, seed=None, options=None):

        theta_1 = np.random.uniform(-0.1, 0.1)
        theta_2 = np.random.uniform(-0.1, 0.1)
        dtheta_1 = np.random.uniform(-0.05, 0.05)
        dtheta_2 = np.random.uniform(-0.05, 0.05)

        self.state = np.array([
            0.0, 0.0, theta_1, theta_2,
            0.0, 0.0, dtheta_1, dtheta_2
        ], dtype=np.float32)
        # self.state = np.array([0, 0, -0.1745, 0.1745, 0, 0, 0, 0], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        # 控制器输出
        self.steps += 1
        self.step_count += 1

        # 将单个动作值转换为两个相同的控制输入
        action = np.array([action, action])

        next_state  = self.G @ self.state + (self.H @ action).reshape(-1) 
        # u += action.clip(-1, 1) * 0.01
        # next_state = self.G @ self.state + self.H @ u


        theta_1, theta_2 = next_state[2], next_state[3]
        dtheta_1, dtheta_2 = next_state[6], next_state[7]
        u_L,u_R = action

        reward = - (
         1 * theta_1**2 +
         1 * theta_2**2 
        )

        # 限制最大 reward 范围
        reward = float(np.clip(reward, -10.0, 0.0))
        # print(self.steps , reward)

        # reward = - float(np.abs(next_state[2 ])) - float(np.abs(next_state[3])) 
        # # 添加扰动：受控动作可加上 agent 的输入

        # 停止条件（这里设为定长）
        done = False
        if self.steps >= self.mx_step :
            done = True
        self.state = next_state
        info = {}

        if hasattr(self, "writer") and self.writer is not None:
            state_names = [
                "theta_L", "theta_R", "theta_1", "theta_2",
                "dtheta_L", "dtheta_R", "dtheta_1", "dtheta_2"
            ]
            for i, name in enumerate(state_names):
                self.writer.add_scalar(f"state/{name}", self.state[i], self.step_count)
            
            # 同时记录控制输入 u_L, u_R
            self.writer.add_scalar("action/u_L", u_L, self.step_count)
            self.writer.add_scalar("action/u_R", u_R, self.step_count)

        return next_state, reward, done, False, info

    def render(self):
        print(f"Current state: {self.state}")

    def close(self):
        pass


env = CustomEnv()
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
check_env(env)
from stable_baselines3.common.monitor import Monitor

env = DummyVecEnv([lambda: Monitor(CustomEnv())])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
# 定义自定义策略网络架构
policy_kwargs = dict(
    features_extractor_class=CustomNetwork,
    # features_extractor_kwargs=dict(features_dim=1),
    net_arch=[dict(pi=[], vf=[])]
)
model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.01, tensorboard_log="./ppo_tensorboard_log", policy_kwargs=policy_kwargs)

# 训练模型
model.learn(total_timesteps=1000000,
            tb_log_name="PPO_CustomEnv")

# 保存模型
model.save("ppo_custom_env_model")
print("模型已保存到 ppo_custom_env_model.zip")



