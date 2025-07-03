from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import gymnasium as gym  # 使用gymnasium作为强化学习环境接口标准

# 定义PPO参数
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
PPO_EPOCHS = 10
BATCH_SIZE = 64
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
ENTROPY_BETA = 0.01  # 熵正则化系数
MAX_EPISODES = 1000  # 最大训练回合数
SAVE_INTERVAL = 20  # 每N个回合保存模型

# 作者：孙波
# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        # 动作标准差，这里我们假设动作是连续的，并且动作空间是独立的
        # 我们可以让标准差学习，或者固定一个小的常数
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 可学习的log_std

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x))  # 假设动作是-1到1之间归一化的
        std = torch.exp(self.log_std)
        return mean, std


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value_layer = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_layer(x)
        return value


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 用于存储经验 (state, action, log_prob, reward, done, value)
        # 注意：这里的 state, action, log_prob, value 都应该存储为 PyTorch Tensor
        self.memory = []

    def remember(self, state, action, log_prob, reward, done, value):
        self.memory.append((state, action, log_prob, reward, done, value))

    def choose_action(self, state_np):  # 输入依然接受 numpy 数组
        state = torch.FloatTensor(state_np).unsqueeze(0)  # 内部转换为 Tensor
        mean, std = self.actor(state)
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        # 返回动作的 numpy 数组，以及 log_prob 和 value 的 detached tensor
        return action.squeeze(0).detach().numpy(), log_prob.squeeze(0).detach(), value.squeeze(0).detach()

    def calculate_advantages(self, rewards, values, dones):
        advantages = []
        returns = []
        gae = 0
        # values 列表的最后一个元素是 next_state_value，用于计算最后一个 delta
        # 在这里，我们假设 values 列表的长度是 len(rewards) + 1
        # 最后一个 values[i+1] 对应的是下一个状态的价值，如果 done=True，则为 0
        for i in reversed(range(len(rewards))):
            # 如果是回合的最后一个状态且 done 为 True，则下一个状态的价值为 0
            # 否则，使用 values[i+1]
            next_value = values[i + 1] if not dones[i] else 0.0  # 修正这里对 done 的处理
            delta = rewards[i] + GAMMA * next_value - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])  # V(s_t) + A(s_t) = Q(s_t)

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def learn(self):
        # 从 memory 中解压数据
        # states, actions, old_log_probs, values 此时已经是 Tensor
        states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)

        states = torch.stack(list(states)).squeeze(1)  # 将 list(states) 转换为 Tensor list，再堆叠
        actions = torch.stack(list(actions))
        old_log_probs = torch.stack(list(old_log_probs))
        values = torch.stack(list(values)).squeeze(1)  # 移除unsqueeze(0)带来的维度，得到形状为(N,)的Tensor

        # 由于 dones 列表中的元素可能是 True/False (布尔值)，需要转换为 int/float
        # 并且 calculate_advantages 需要下一个状态的价值，所以 values 需要多一个元素
        # 我们需要从 memory 中取出所有状态对应的 value，并在末尾添加一个 0 （表示终结状态的价值）
        # 或者在收集数据时，就确保 value 包含了下一个状态的 value。
        # 最简单的方法是在 learn() 内部构造一个完整的 value 列表
        all_values_for_gae = list(values.detach().numpy())  # 将 tensor 转换为 numpy 数组，用于 GAE 计算
        # GAE 需要下一个状态的价值，对于 done=True 的情况，下一个状态的价值视为 0
        # 这里假设 values 中已经包含了所有 s_t 的价值，而不是 s_{t+1} 的价值
        # 因此，我们需要一个额外的 0 来表示终结状态的价值
        all_values_for_gae.append(0.0)  # 附加一个终结状态的价值，用于最后一个时间步的 GAE 计算

        # 将 dones 列表转换为 NumPy 数组，以方便进行元素级操作
        dones_np = np.array(dones, dtype=np.float32)

        advantages, returns = self.calculate_advantages(list(rewards), all_values_for_gae, dones_np)

        # 进行PPO_EPOCHS次训练
        for _ in range(PPO_EPOCHS):
            # 随机打乱数据并按批次处理
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Actor Loss
                mean, std = self.actor(batch_states)
                dist = MultivariateNormal(mean, torch.diag_embed(std))
                new_log_probs = dist.log_prob(batch_actions)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages

                # 策略梯度损失 + 熵正则化
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_BETA * dist.entropy().mean()

                # Critic Loss
                current_values = self.critic(batch_states).squeeze(-1)
                critic_loss = torch.mean((current_values - batch_returns) ** 2)

                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear()  # 清空经验回放缓冲区

    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))

# from balance_car_communicator import BalanceCarCommunicator
# 你的环境接口（你需要实现这些）
class TwoWheeledBalanceCarEnv(gym.Env):
    def __init__(self):
        super(TwoWheeledBalanceCarEnv, self).__init__()
        # 定义状态空间 (8维向量)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # 定义动作空间 (2个连续值，例如左右轮的扭矩或速度)
        # 假设动作范围在 -1.0 到 1.0 之间
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.current_state = None  # 初始化状态
        self.max_episode_steps = 200  # 设置最大步数，用于 truncated
        self.current_step = 0
        self.communicator = None


        self.Ad = [
            [1, 0, 0, 0, 0.0100000000000000, 0, 0, 0],
            [0, 1, 0, 0, 0, 0.0100000000000000, 0, 0],
            [0, 0, 1.00658889597248, -0.000898767868656603, 0, 0, 0.0100219527186279, -2.99411655300117e-06],
            [0, 0, -0.00319408273322577, 1.00232090086286, 0, 0, -1.06406295960503e-05, 0.0100077345036121],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1.31931980503437, -0.180020101303433, 0, 0, 1.00658889597248, -0.000898767868656603],
            [0, 0, -0.639763744632199, 0.464455118844735, 0, 0, -0.00319408273322577, 1.00232090086286]
        ]
        self.Ad = np.array(self.Ad)

        self.Bd = [
            [5.00000000000000e-05, 0],
            [0, 5.00000000000000e-05],
            [-9.72547405500003e-06, -9.72547405500003e-06],
            [1.49242819680389e-06, 1.49242819680389e-06],
            [0.0100000000000000, 0],
            [0, 0.0100000000000000],
            [-0.00194727245280543, -0.00194727245280543],
            [0.000299634641014288, 0.000299634641014288]
        ]
        self.Bd = np.array(self.Bd)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: 在这里实现你的环境重置逻辑
        # 初始化 theta_1, theta_dot_1, ..., theta_R_dot
        # 例如:
        initial_theta_1 = np.random.uniform(-0.1, 0.1)
        initial_theta_dot_1 = np.random.uniform(-0.05, 0.05)
        initial_theta_2 = np.random.uniform(-0.06, 0.06)
        initial_theta_dot_2 = np.random.uniform(-0.05, 0.05)
        initial_theta_L = np.random.uniform(-0.05, 0.05)
        initial_theta_dot_L = np.random.uniform(-0.05, 0.05)
        initial_theta_R = np.random.uniform(-0.05, 0.05)
        initial_theta_dot_R = np.random.uniform(-0.05, 0.05)
        # ...
        self.current_state = np.array([initial_theta_L, initial_theta_R, initial_theta_1, initial_theta_2, initial_theta_dot_L, initial_theta_dot_R, initial_theta_dot_1, initial_theta_dot_2])
        # 这里只是一个示例，你需要根据你的物理模型初始化状态
        # 确保初始状态是平衡的，或者从平衡点附近的小扰动开始
        # print('开始新的一轮，请点击复位键')
        # self.communicator = BalanceCarCommunicator()
        # self.communicator.connect_wifi()
        # sensor_data = self.communicator.receive_sensor_data()



        # self.current_state = sensor_data[:8]  # 示例，实际需要您的环境初始化
        # self.current_step = 0
        info = {}  # 可以返回额外信息
        return self.current_state, info

    def step(self, action):
        # TODO: 在这里实现你的环境步进逻辑
        # action 是一个包含两个值的numpy数组，对应左右轮的输入
        # 例如: left_wheel_torque = action[0], right_wheel_torque = action[1]

        # 1. 根据当前状态和动作更新环境（物理模拟）
        # self.current_state = self._update_physics(self.current_state, action)
        # ！！非常重要：下面的模拟是随机的，你需要用真实的物理模型替换它！！
        # 模拟下一个状态（你需要根据你的物理模型来实际更新）
        # 这里只是一个占位符，模拟随机变化，或者一个简单的线性动力学模型

        # 示例：一个非常简化的、不准确的模拟，仅用于代码运行
        # 假设 action 对状态有一定影响，但主要还是随机扰动
        action_vec = np.array([action[0], action[1]])

        u_L = action[0]*1000
        u_R = action[1]*1000

        # if not self.communicator.send_control_data(u_L, u_R):
        #     print('指令发送失败')

        # sensor_data = self.communicator.receive_sensor_data()

        # self.current_state = sensor_data[:8]

        self.current_state = (np.matmul(self.Ad, self.current_state) + np.matmul(self.Bd, action_vec))

        # 1. 提取状态变量
        theta_L = self.current_state[0]
        theta_R = self.current_state[1]
        theta_1 = self.current_state[2]
        theta_2 = self.current_state[3]
        theta_dot_L = self.current_state[4]
        theta_dot_R = self.current_state[5]
        theta_dot_1 = self.current_state[6]
        theta_dot_2 = self.current_state[7]

        # 动作值
        action_L = action[0]
        action_R = action[1]

        # 2. 计算奖励
        reward = 0

        # **核心平衡奖励：车身倾角惩罚**
        # 目标是theta_1接近0，保持车身直立
        reward -= 5 * ((theta_1-0.1) ** 2)

        # **车身倾角速度惩罚：抑制晃动**
        # 目标是dot_theta_1接近0
        reward -= 2 * (theta_dot_1 ** 2)

        # **摆杆倾角惩罚：保持摆杆直立**
        # 目标是theta_2接近0，如果摆杆也需要直立的话
        reward -= 5 * ((theta_2-0.06) ** 2)  # 系数可根据摆杆的重要性调整

        # **摆杆倾角速度惩罚：抑制摆杆晃动**
        # 目标是dot_theta_2接近0
        reward -= 0.05 * (theta_dot_2 ** 2)  # 系数可根据摆杆的重要性调整

        # **保持原地不动惩罚：轮子绝对位置和速度惩罚**
        # 惩罚左右轮的绝对位移，鼓励小车在原地不动
        # wheel_position_penalty = (theta_L ** 2 + theta_R ** 2) / 2.0
        # reward -= 0.1 * wheel_position_penalty

        # 惩罚左右轮的绝对角速度，避免持续转动
        wheel_velocity_penalty = (theta_dot_L ** 2 + theta_dot_R ** 2) / 2.0
        reward -= 0.5 * wheel_velocity_penalty

        # **抑制左右轮速度差：保持直线**
        # 惩罚左右轮角速度差，鼓励直线运动，避免原地打转
        reward -= 0.05 * ((theta_dot_L - theta_dot_R) ** 2)

        # **动作平滑性/能量效率惩罚：抑制过大的控制输出**
        # 惩罚动作的平方，鼓励使用较小的力矩来维持平衡
        reward -= 0.01 * (action_L ** 2 + action_R ** 2)

        # 3. 判断是否结束
        done = False

        if theta_1 > 1.7 or theta_1 < -1.7:
            u_L = 0
            u_R = 0
            # 发送0指令
            # if not self.communicator.send_control_data(u_L, u_R):
            #     print("发送停止指令失败。")  # 错误处理

            # print('角度过大停止运动，请复位')
            # self.close()  # 关闭当前连接
            done = True
            reward -= 100

        self.current_step += 1
        truncated = False
        if self.current_step >= self.max_episode_steps:
            truncated = True  # 达到最大步数，回合被截断

        info = {}

        return self.current_state, reward, done, truncated, info

def show_res(history):
    # Plot test results
    plt.figure(figsize=(10, 8))
    
    plt.subplot(7, 1, 1)
    plt.plot(history[:, 0])
    plt.title('Cart Position L')
    plt.ylabel('x (m)')
    
    plt.subplot(7, 1, 2)
    plt.plot(history[:, 1])
    plt.title('Cart Position R')
    plt.ylabel('theta (rad)')

    plt.subplot(7, 1, 3)
    plt.plot(history[:, 2])
    plt.title('first Pole Angle')
    plt.ylabel('theta (rad)')
    
    plt.subplot(7, 1, 4)
    plt.plot(history[:, 3])
    plt.title('Second Pole Angle')
    plt.ylabel('F (N)')
    plt.xlabel('Time step')

    plt.subplot(7, 1, 5)
    plt.plot(history[:, 4])
    plt.title('U L')
    plt.ylabel('v')
    plt.xlabel('Time step')

    plt.subplot(7, 1, 6)
    plt.plot(history[:, 4])
    plt.title('U R')
    plt.ylabel('v')
    plt.xlabel('Time step')

    plt.subplot(7, 1, 7)
    plt.plot(history[:, 4])
    plt.title('Reward')
    plt.ylabel('v')
    plt.xlabel('Time step')
    
    plt.tight_layout()
    plt.show()


def test_policy(env, agent, num_episodes=10, max_steps=1000):
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        history = np.zeros((max_steps, 7))
        
        for step in range(max_steps):
            state_array = np.array([
                state[0],
                state[1],
                state[2],
                state[3],
                state[4],
                state[5],
                state[6],
                state[7]
            ])

            action_np, log_prob, value = agent.choose_action(state_array)
            action_np_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)
            next_state_np, reward, done, truncated, info = env.step(action_np_clipped)

            history[step] = np.array([state[0], state[1], state[2], state[3], action_np_clipped[0], action_np_clipped[1], reward])
            
            episode_reward += reward
            state = next_state_np
            
            # env.render()
            
            if done:
                # state, info = env.reset()
                break
        
        print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")
        show_res(history)


# 主训练循环
if __name__ == "__main__":
    env = TwoWheeledBalanceCarEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(state_dim, action_dim)

    # 可以选择加载预训练模型
    agent.load_models("ppo_balance_car_model_episode_1000")

    for episode in range(MAX_EPISODES):
        # 从环境获取的初始状态是 NumPy 数组
        state_np, info = env.reset()
        # 立即将 NumPy 数组转换为 PyTorch Tensor，用于存储在 memory 中
        state_tensor = torch.FloatTensor(state_np)

        episode_reward = 0
        done = False
        truncated = False
        step_count = 0

        while not done and not truncated:
            # agent.choose_action 内部会处理 state_np 到 Tensor 的转换
            action_np, log_prob, value = agent.choose_action(state_np)

            # 将动作裁剪到环境的动作空间范围内
            action_np_clipped = np.clip(action_np, env.action_space.low, env.action_space.high)

            # 环境步进，接收下一个状态 (NumPy 数组)
            next_state_np, reward, done, truncated, info = env.step(action_np_clipped)
            # 将下一个状态 NumPy 数组转换为 PyTorch Tensor
            next_state_tensor = torch.FloatTensor(next_state_np)

            # 存储经验时，state, action, log_prob, value 都应该是 Tensor
            # action_np_clipped 需要转换为 Tensor
            agent.remember(state_tensor, torch.FloatTensor(action_np_clipped), log_prob, reward, done, value)

            # 更新当前状态，以便下一轮循环使用
            state_np = next_state_np
            state_tensor = next_state_tensor

            episode_reward += reward
            step_count += 1

            # 每隔一定步数（例如一个回合结束），进行学习
            # 或者当内存缓冲区达到一定大小后进行学习
            if len(agent.memory) >= BATCH_SIZE * PPO_EPOCHS:  # 确保有足够数据进行训练
                agent.learn()

        # 在一个回合结束后，如果 memory 中还有数据，也进行一次学习
        if len(agent.memory) > 0:
            agent.learn()

        print(f"回合: {episode + 1}, 总奖励: {episode_reward:.2f}, 步数: {step_count}")

        if (episode + 1) % SAVE_INTERVAL == 0:
            agent.save_models(f"ppo_balance_car_model_episode_{episode + 1}")
            print(f"模型在回合 {episode + 1} 时保存")

    print("训练完成。")

    test_policy(env, agent)
    env.close()