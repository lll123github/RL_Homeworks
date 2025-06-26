import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from PPO.train_ppo import show_res  # 导入你的自定义环境
import gymnasium as gym
from gymnasium import spaces
import scipy.signal
import scipy.linalg

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)
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

def load_and_test_model(model_path):
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 保存策略网络权重矩阵
    weights_dir = "policy_weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # 遍历策略网络的所有参数
    for name, param in model.policy.named_parameters():
        if param.requires_grad:  # 只保存需要梯度的参数
            # 将参数转换为numpy数组
            weight_matrix = param.detach().cpu().numpy()
            # 保存为txt文件
            weight_file = os.path.join(weights_dir, f"{name.replace('.', '_')}.txt")
            np.savetxt(weight_file, weight_matrix, fmt='%.6f')
            print(f"保存权重矩阵 {name} 到 {weight_file}")
            print(f"矩阵形状: {weight_matrix.shape}")
    
    # 打印模型参数
    print("\n=== 模型参数 ===")
    print(f"策略网络架构: {model.policy}")
    print(f"学习率: {model.learning_rate}")
    print(f"折扣因子: {model.gamma}")
    print(f"GAE-Lambda: {model.gae_lambda}")
    print(f"熵系数: {model.ent_coef}")
    print(f"值函数系数: {model.vf_coef}")
    print(f"最大梯度范数: {model.max_grad_norm}")
    print(f"目标KL散度: {model.target_kl}")
    
    # 保存模型参数到txt文件
    params_file = "model_parameters.txt"
    with open(params_file, "w", encoding="utf-8") as f:
        f.write("=== 模型参数 ===\n")
        f.write(f"策略网络架构: {model.policy}\n")
        f.write(f"学习率: {model.learning_rate}\n")
        f.write(f"折扣因子: {model.gamma}\n")
        f.write(f"GAE-Lambda: {model.gae_lambda}\n")
        f.write(f"熵系数: {model.ent_coef}\n")
        f.write(f"值函数系数: {model.vf_coef}\n")
        f.write(f"最大梯度范数: {model.max_grad_norm}\n")
        f.write(f"目标KL散度: {model.target_kl}\n")
    print(f"\n模型参数已保存到 {params_file}")
    
    # 创建测试环境
    env = CustomEnv()  # 直接实例化你的自定义环境
    env = DummyVecEnv([lambda: env])
    
    # 评估模型
    print("\n=== 模型评估 ===")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # 交互式测试
    print("\n=== 交互式测试 ===")
    for episode in range(1000):
        obs = env.reset()
        history = np.zeros((1000, 7))
        for i in range(1000):  # 运行1000步
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            o, r, a = obs.squeeze(), reward.squeeze(), action.squeeze()
            history[i] = np.array([o[0], o[1], o[2], o[3], a, a, r])
            # print(f"步骤 {i}: 动作={action}, 奖励={reward}")
            # if done:
            #     break
            
        show_res(history)    
    env.close()

if __name__ == "__main__":
    model_path = "ppo_custom_env_model_1.zip"
    load_and_test_model(model_path)
