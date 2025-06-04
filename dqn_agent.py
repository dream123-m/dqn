import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# 经验回放缓冲区
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'action_mask'])


class ReplayBuffer:
    """支持动作掩码的经验回放缓冲区"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, action_mask):
        """添加经验"""
        self.buffer.append(Experience(state, action, reward, next_state, done, action_mask))

    def sample(self, batch_size):
        """随机采样批次数据"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class MaskedDQNNetwork(nn.Module):
    """支持动作掩码的DQN网络"""

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(MaskedDQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 构建网络层
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        # 输出层：为每个动作维度输出Q值
        layers.append(nn.Linear(input_dim, action_dim * 2))

        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, state, action_mask=None):
        """前向传播，支持动作掩码"""
        q_values = self.network(state)
        q_values = q_values.view(-1, self.action_dim, 2)

        # 应用动作掩码
        if action_mask is not None:
            mask = action_mask.unsqueeze(-1).expand_as(q_values)
            q_values = q_values.masked_fill(~mask, -1e9)  # 无效动作设为极小值

        return q_values


class MaskedDQNAgent:
    """支持动作掩码的DQN智能体"""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=64, target_update=1000,
                 device=None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # 设备选择
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 创建主网络和目标网络
        self.q_network = MaskedDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = MaskedDQNNetwork(state_dim, action_dim).to(self.device)

        # 初始化目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

        # 训练统计
        self.steps = 0
        self.losses = []
        self.rewards_history = []

    def select_action(self, state, action_mask=None, training=True):
        """选择动作，支持动作掩码"""
        if action_mask is None:
            action_mask = np.ones(self.action_dim, dtype=bool)

        # 获取有效动作索引
        valid_actions = np.where(action_mask)[0]

        if training and random.random() < self.epsilon:
            # 探索：在有效动作中随机选择
            action = np.zeros(self.action_dim, dtype=int)
            for valid_idx in valid_actions:
                action[valid_idx] = np.random.randint(0, 2)
            return action
        else:
            # 利用：基于Q值选择最优动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

                q_values = self.q_network(state_tensor, mask_tensor)

                # 为每个有效动作维度选择Q值最大的选择
                action = torch.argmax(q_values, dim=2).squeeze(0).cpu().numpy()

                # 确保无效动作为0
                action = action * action_mask.astype(int)
                return action

    def store_experience(self, state, action, reward, next_state, done, action_mask):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done, action_mask)

    def update(self):
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从经验回放缓冲区采样
        experiences = self.replay_buffer.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        action_masks = torch.BoolTensor([e.action_mask for e in experiences]).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states, action_masks)
        current_q = torch.gather(current_q_values, 2, actions.unsqueeze(2)).squeeze(2)
        current_q = current_q.sum(dim=1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states, action_masks)
            next_q = torch.max(next_q_values, dim=2)[0].sum(dim=1)
            target_q = rewards + (self.gamma * next_q * ~dones)

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 记录损失
        self.losses.append(loss.item())

        # 更新探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'losses': self.losses,
            'rewards_history': self.rewards_history
        }, filepath)
        print(f"模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.losses = checkpoint['losses']
        self.rewards_history = checkpoint['rewards_history']
        print(f"模型已从 {filepath} 加载")

    def plot_training_stats(self):
        """绘制训练统计图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        if self.losses:
            ax1.plot(self.losses)
            ax1.set_title('训练损失')
            ax1.set_xlabel('更新步数')
            ax1.set_ylabel('损失')
            ax1.grid(True)

        if self.rewards_history:
            ax2.plot(self.rewards_history)
            ax2.set_title('Episode奖励')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('总奖励')
            ax2.grid(True)

            if len(self.rewards_history) > 10:
                window_size = min(100, len(self.rewards_history) // 10)
                moving_avg = np.convolve(self.rewards_history,
                                         np.ones(window_size) / window_size,
                                         mode='valid')
                ax2.plot(range(window_size - 1, len(self.rewards_history)),
                         moving_avg, 'r-', label=f'{window_size}-期移动平均')
                ax2.legend()

        plt.tight_layout()
        plt.show()


def test_masked_dqn_agent():
    """测试支持掩码的DQN智能体"""
    state_dim = 50
    action_dim = 20

    agent = MaskedDQNAgent(state_dim, action_dim)

    print("改进的DQN智能体创建成功!")
    print("✅ 支持动作掩码，避免无效动作干扰学习")
    print("✅ 改进的经验回放，包含掩码信息")
    print("✅ 掩码感知的Q值计算")

    # 测试动作选择
    dummy_state = np.random.randn(state_dim)
    action_mask = np.array([True] * 10 + [False] * 10)  # 前10个动作有效
    action = agent.select_action(dummy_state, action_mask)
    print(f"示例动作 (掩码约束): {action}")
    print(f"无效动作是否为0: {np.all(action[10:] == 0)}")


if __name__ == "__main__":
    test_masked_dqn_agent()