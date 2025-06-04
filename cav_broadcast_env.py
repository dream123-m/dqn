import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class CAVBroadcastEnv(gym.Env):
    """改进的CAV协同感知广播优化环境"""

    def __init__(self, data_loader, w1=0.5, w2=0.3, w3=0.2, sharing_threshold=0.95, max_distance=150.0):
        """
        初始化环境
        Args:
            data_loader: MATLABDataLoader实例
            w1: 信息价值权重
            w2: 广播代价权重
            w3: 共享率激励权重
            sharing_threshold: 共享率阈值
            max_distance: 最大传感器距离
        """
        super(CAVBroadcastEnv, self).__init__()

        self.data_loader = data_loader
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.sharing_threshold = sharing_threshold
        self.max_distance = max_distance

        # 获取CAV信息
        self.cav_indices = data_loader.get_cav_indices()
        self.num_cavs = len(self.cav_indices)
        self.cav_los_matrix, _ = data_loader.get_cav_los_matrix()

        # 计算距离矩阵
        self.distance_matrix = data_loader.calculate_distances()

        # 为每个CAV构建其可感知目标列表和距离
        self.cav_targets = {}
        self.cav_distances = {}
        for i, cav_idx in enumerate(self.cav_indices):
            # 获取该CAV可以感知的目标
            los_connections = np.where(self.cav_los_matrix[i] == 1)[0]
            # 过滤掉超出传感器范围的目标
            valid_targets = []
            valid_distances = []
            for target_idx in los_connections:
                dist = self.distance_matrix[cav_idx, target_idx]
                if dist <= self.max_distance and target_idx != cav_idx:  # 排除自己
                    valid_targets.append(target_idx)
                    valid_distances.append(dist)

            self.cav_targets[cav_idx] = valid_targets
            self.cav_distances[cav_idx] = valid_distances

        # 计算最大可能的目标数量（用于定义动作空间）
        self.max_targets_per_cav = max([len(targets) for targets in self.cav_targets.values()])

        # 定义动作空间和状态空间
        self._setup_spaces()

        # 当前状态
        self.current_cav_idx = 0
        self.reset()

    def _setup_spaces(self):
        """设置动作空间和状态空间"""
        # 动作空间：对每个可见目标的广播决策（0或1）
        self.action_space = spaces.MultiBinary(self.max_targets_per_cav)

        # 改进的状态空间：固定长度但包含有效性标记
        # 状态维度：[CAV位置(2) + 每个目标槽位的信息(4*max_targets) + 全局共享率(1)]
        # 每个目标槽位：[相对x, 相对y, 距离, 是否有效(0/1)]
        state_dim = 2 + 4 * self.max_targets_per_cav + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

    def reset(self):
        """重置环境"""
        self.current_cav_idx = 0
        self.total_broadcasts = 0
        self.total_information_value = 0.0
        self.shared_targets = set()  # 已被广播的目标集合
        self.target_broadcast_count = {}  # 每个目标的广播次数
        self.episode_actions = {}  # 记录每个CAV的动作

        return self._get_state()

    def _get_state(self):
        """获取当前状态 - 改进版本"""
        if self.current_cav_idx >= self.num_cavs:
            # 所有CAV都已决策完毕，返回零状态
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        cav_idx = self.cav_indices[self.current_cav_idx]
        cav_pos = self.data_loader.positions[cav_idx]
        targets = self.cav_targets[cav_idx]
        distances = self.cav_distances[cav_idx]

        # 构建状态向量
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # CAV位置
        state[0:2] = cav_pos

        # 目标信息槽位 - 每个槽位4个值
        for i in range(self.max_targets_per_cav):
            slot_start = 2 + i * 4

            if i < len(targets):
                # 有效目标
                target_idx = targets[i]
                target_pos = self.data_loader.positions[target_idx]
                relative_pos = target_pos - cav_pos

                state[slot_start] = relative_pos[0]  # 相对x
                state[slot_start + 1] = relative_pos[1]  # 相对y
                state[slot_start + 2] = distances[i]  # 距离
                state[slot_start + 3] = 1.0  # 有效标记
            else:
                # 无效槽位，保持全零，有效标记为0
                pass

        # 当前全局共享率
        total_targets = len(set().union(*self.cav_targets.values()))
        current_sharing_rate = len(self.shared_targets) / max(total_targets, 1)
        state[-1] = current_sharing_rate

        return state

    def get_action_mask(self):
        """获取当前CAV的动作掩码"""
        if self.current_cav_idx >= self.num_cavs:
            return np.zeros(self.max_targets_per_cav, dtype=bool)

        cav_idx = self.cav_indices[self.current_cav_idx]
        targets = self.cav_targets[cav_idx]

        # 创建掩码：前len(targets)个动作有效
        mask = np.zeros(self.max_targets_per_cav, dtype=bool)
        mask[:len(targets)] = True

        return mask

    def step(self, action):
        """执行动作 - 改进版本"""
        if self.current_cav_idx >= self.num_cavs:
            # 所有CAV已完成决策
            return self._get_state(), 0, True, {}

        cav_idx = self.cav_indices[self.current_cav_idx]
        targets = self.cav_targets[cav_idx]
        distances = self.cav_distances[cav_idx]

        # 应用动作掩码，只考虑有效动作
        action_mask = self.get_action_mask()
        masked_action = np.array(action) * action_mask
        actual_action = masked_action[:len(targets)]

        self.episode_actions[cav_idx] = actual_action

        # 计算奖励
        reward = self._calculate_reward(cav_idx, actual_action, targets, distances)

        # 更新广播状态和计数
        broadcast_count = np.sum(actual_action)
        self.total_broadcasts += broadcast_count

        # 更新已广播目标集合和广播计数
        for i, should_broadcast in enumerate(actual_action):
            if should_broadcast:
                target_idx = targets[i]
                self.shared_targets.add(target_idx)
                self.target_broadcast_count[target_idx] = self.target_broadcast_count.get(target_idx, 0) + 1

        # 移动到下一个CAV
        self.current_cav_idx += 1

        # 检查是否完成
        done = self.current_cav_idx >= self.num_cavs

        info = {
            'cav_idx': cav_idx,
            'broadcast_count': broadcast_count,
            'total_broadcasts': self.total_broadcasts,
            'sharing_rate': len(self.shared_targets) / max(len(set().union(*self.cav_targets.values())), 1),
            'action_mask': action_mask
        }

        return self._get_state(), reward, done, info

    def _calculate_reward(self, cav_idx, action, targets, distances):
        """改进的奖励函数"""
        if len(targets) == 0:
            return 0.0

        # 1. 计算有效信息价值
        effective_value = 0.0
        num_broadcasts = 0

        for i, should_broadcast in enumerate(action):
            if should_broadcast and i < len(distances):
                target_idx = targets[i]

                # 距离价值：(1 - d_i / max_distance)
                distance_value = 1.0 - (distances[i] / self.max_distance)

                # 重复广播惩罚：1 / (1 + 已广播次数)
                broadcast_penalty = 1.0 / (1.0 + self.target_broadcast_count.get(target_idx, 0))

                # 有效价值 = 距离价值 * 重复惩罚
                effective_value += distance_value * broadcast_penalty
                num_broadcasts += 1

        # 2. 广播代价：当前CAV的广播比例
        broadcast_ratio = num_broadcasts / len(targets) if len(targets) > 0 else 0.0

        # 3. 共享率激励/惩罚
        total_targets = len(set().union(*self.cav_targets.values()))
        current_sharing_rate = len(self.shared_targets) / max(total_targets, 1)

        # 如果共享率低于阈值，给予激励；超过太多给予轻微惩罚
        if current_sharing_rate < self.sharing_threshold:
            sharing_incentive = (self.sharing_threshold - current_sharing_rate) * 2.0
        elif current_sharing_rate > self.sharing_threshold + 0.1:
            sharing_incentive = -0.5 * (current_sharing_rate - self.sharing_threshold - 0.1)
        else:
            sharing_incentive = 0.1  # 适中范围内的小奖励

        # 最终奖励函数
        reward = (self.w1 * effective_value -
                  self.w2 * broadcast_ratio +
                  self.w3 * sharing_incentive)

        return reward

    def get_episode_summary(self):
        """获取当前episode的总结"""
        total_targets = len(set().union(*self.cav_targets.values()))
        sharing_rate = len(self.shared_targets) / max(total_targets, 1)

        # 计算总有效信息价值
        total_effective_value = 0.0
        for cav_idx, action in self.episode_actions.items():
            targets = self.cav_targets[cav_idx]
            distances = self.cav_distances[cav_idx]
            for i, should_broadcast in enumerate(action):
                if should_broadcast and i < len(distances):
                    target_idx = targets[i]
                    distance_value = 1.0 - (distances[i] / self.max_distance)
                    broadcast_penalty = 1.0 / (1.0 + self.target_broadcast_count.get(target_idx, 0))
                    total_effective_value += distance_value * broadcast_penalty

        return {
            'total_broadcasts': self.total_broadcasts,
            'sharing_rate': sharing_rate,
            'total_information_value': total_effective_value,
            'meets_sharing_threshold': sharing_rate >= self.sharing_threshold,
            'objective_value': total_effective_value - (
                        self.total_broadcasts / max(sum([len(targets) for targets in self.cav_targets.values()]), 1)),
            'avg_broadcasts_per_target': np.mean(
                list(self.target_broadcast_count.values())) if self.target_broadcast_count else 0.0
        }

    def render(self, mode='human'):
        """可视化环境"""
        if mode == 'human':
            summary = self.get_episode_summary()
            print(f"当前episode状态:")
            print(f"  总广播数: {summary['total_broadcasts']}")
            print(f"  共享率: {summary['sharing_rate']:.3f}")
            print(f"  满足共享率阈值: {'是' if summary['meets_sharing_threshold'] else '否'}")
            print(f"  总有效信息价值: {summary['total_information_value']:.3f}")
            print(f"  目标函数值: {summary['objective_value']:.3f}")
            print(f"  平均重复广播次数: {summary['avg_broadcasts_per_target']:.2f}")


def test_environment():
    """测试改进的环境"""
    print("改进的CAV广播优化环境特性:")
    print("✅ 解决状态稀疏问题：使用有效性标记区分真实目标和填充")
    print("✅ 动作掩码：屏蔽无效动作，避免干扰学习")
    print("✅ 改进奖励函数：考虑距离价值、重复广播惩罚和共享率激励")
    print("✅ 重复广播管理：跟踪每个目标的广播次数")


if __name__ == "__main__":
    test_environment()