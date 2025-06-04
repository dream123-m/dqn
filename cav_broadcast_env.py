import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class CAVBroadcastEnv(gym.Env):
    """改进的CAV协同感知广播优化环境"""

    def __init__(self, data_loader, w1=0.4, w2=0.4, w3=0.4, sharing_threshold=0.95, max_distance=150.0):
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
        """设置动作空间和状态空间
        优化后的状态空间设计：
        1. 移除冗余的目标槽位
        2. 使用极坐标表示替代笛卡尔坐标+距离
        """
        # 动作空间：对每个可见目标的广播决策（0或1）
        self.action_space = spaces.MultiBinary(self.max_targets_per_cav)

        # 优化的状态空间：
        # [当前CAV位置(2) + 
        #  动态目标信息(每个实际目标2个值：角度+距离) + 
        #  全局共享率(1)]
        max_state_dim = 2 + (2 * self.max_targets_per_cav) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_state_dim,), dtype=np.float32
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
        """获取当前状态 - 优化版本"""
        if self.current_cav_idx >= self.num_cavs:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        cav_idx = self.cav_indices[self.current_cav_idx]
        cav_pos = self.data_loader.positions[cav_idx]
        targets = self.cav_targets[cav_idx]
        distances = self.cav_distances[cav_idx]

        # 构建状态向量
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # 1. CAV位置
        state[0:2] = cav_pos

        # 2. 目标信息（使用极坐标：角度+距离）
        for i, target_idx in enumerate(targets):
            if i >= self.max_targets_per_cav:
                break
                
            target_pos = self.data_loader.positions[target_idx]
            relative_pos = target_pos - cav_pos
            
            # 计算角度（弧度）和距离
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            distance = distances[i]
            
            # 存储极坐标信息
            state_idx = 2 + (i * 2)
            state[state_idx] = angle  # 角度
            state[state_idx + 1] = distance  # 距离

        # 3. 当前全局共享率
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
        """优化的奖励函数计算
        1. 信息价值考虑距离和重复性
        2. 广播效率考虑必要性
        3. 共享率采用软阈值设计
        """
        if len(targets) == 0:
            return 0.0

        # 1. 信息价值项（归一化到[0,1]）
        total_value = 0.0
        max_possible_value = len(targets)  # 最大可能价值
        
        for i, should_broadcast in enumerate(action):
            if should_broadcast and i < len(distances):
                # 距离归一化价值
                distance_value = 1.0 - (distances[i] / self.max_distance)
                # 重复广播衰减
                target_idx = targets[i]
                repeat_count = self.target_broadcast_count.get(target_idx, 0)
                # 使用指数衰减惩罚重复广播
                repeat_penalty = np.exp(-0.5 * repeat_count)
                total_value += distance_value * repeat_penalty

        normalized_value = total_value / max_possible_value

        # 2. 广播效率项（考虑必要性）
        broadcast_count = np.sum(action)
        current_sharing_rate = len(self.shared_targets) / max(len(set().union(*self.cav_targets.values())), 1)
        
        # 根据当前共享率动态调整广播效率评估
        if current_sharing_rate >= self.sharing_threshold:
            # 已达到阈值，严格控制广播数量
            broadcast_efficiency = max(0, 0.5 - (broadcast_count / len(targets)))
        else:
            # 未达到阈值，允许适度广播
            needed_sharing = (self.sharing_threshold - current_sharing_rate) / self.sharing_threshold
            optimal_broadcast_ratio = min(0.8, needed_sharing + 0.2)  # 基础广播率0.2，最高0.8
            actual_ratio = broadcast_count / len(targets)
            broadcast_efficiency = 1.0 - abs(optimal_broadcast_ratio - actual_ratio)

        # 3. 共享率奖励（采用分段设计）
        if current_sharing_rate < self.sharing_threshold:
            # 未达到阈值时，提供递增的奖励
            sharing_score = (current_sharing_rate / self.sharing_threshold) * 0.8
        else:
            # 达到阈值后，提供小幅奖励，但避免过度激励
            over_threshold = current_sharing_rate - self.sharing_threshold
            sharing_score = 0.8 + 0.2 * np.exp(-5.0 * over_threshold)

        # 统一的奖励计算（所有项都在[0,1]范围内）
        reward = (self.w1 * normalized_value + 
                 self.w2 * broadcast_efficiency + 
                 self.w3 * sharing_score)

        return reward

    def get_episode_summary(self):
        """获取当前episode的总结，使用与奖励计算一致的逻辑"""
        total_targets = len(set().union(*self.cav_targets.values()))
        sharing_rate = len(self.shared_targets) / max(total_targets, 1)

        # 计算总有效信息价值
        total_effective_value = 0.0
        total_broadcasts = 0
        for cav_idx, action in self.episode_actions.items():
            targets = self.cav_targets[cav_idx]
            distances = self.cav_distances[cav_idx]
            for i, should_broadcast in enumerate(action):
                if should_broadcast and i < len(distances):
                    target_idx = targets[i]
                    distance_value = 1.0 - (distances[i] / self.max_distance)
                    repeat_count = self.target_broadcast_count.get(target_idx, 0)
                    repeat_penalty = np.exp(-0.5 * repeat_count)
                    total_effective_value += distance_value * repeat_penalty
                    total_broadcasts += 1

        # 计算广播效率
        total_possible_broadcasts = sum([len(targets) for targets in self.cav_targets.values()])
        broadcast_efficiency = 1.0 - (total_broadcasts / max(total_possible_broadcasts, 1))

        # 计算最终目标函数值（使用与奖励计算相同的权重）
        if sharing_rate >= self.sharing_threshold:
            sharing_score = 0.8 + 0.2 * np.exp(-5.0 * (sharing_rate - self.sharing_threshold))
        else:
            sharing_score = (sharing_rate / self.sharing_threshold) * 0.8

        objective_value = (self.w1 * total_effective_value / max(total_possible_broadcasts, 1) +
                         self.w2 * broadcast_efficiency +
                         self.w3 * sharing_score)

        return {
            'total_broadcasts': total_broadcasts,
            'sharing_rate': sharing_rate,
            'total_information_value': total_effective_value,
            'broadcast_efficiency': broadcast_efficiency,
            'sharing_score': sharing_score,
            'meets_sharing_threshold': sharing_rate >= self.sharing_threshold,
            'objective_value': objective_value,
            'avg_broadcasts_per_target': np.mean(list(self.target_broadcast_count.values())) if self.target_broadcast_count else 0.0
        }

    def render(self, mode='human'):
        """可视化环境状态"""
        if mode == 'human':
            summary = self.get_episode_summary()
            print(f"\n当前episode状态:")
            print(f"  总广播数: {summary['total_broadcasts']}")
            print(f"  共享率: {summary['sharing_rate']:.3f} ({'高于' if summary['meets_sharing_threshold'] else '低于'}阈值)")
            print(f"  广播效率: {summary['broadcast_efficiency']:.3f}")
            print(f"  信息价值: {summary['total_information_value']:.3f}")
            print(f"  共享得分: {summary['sharing_score']:.3f}")
            print(f"  目标函数值: {summary['objective_value']:.3f}")
            print(f"  平均重复广播: {summary['avg_broadcasts_per_target']:.2f}次/目标")


def test_environment():
    """测试改进的环境"""
    print("改进的CAV广播优化环境特性:")
    print("✅ 解决状态稀疏问题：使用有效性标记区分真实目标和填充")
    print("✅ 动作掩码：屏蔽无效动作，避免干扰学习")
    print("✅ 改进奖励函数：考虑距离价值、重复广播惩罚和共享率激励")
    print("✅ 重复广播管理：跟踪每个目标的广播次数")


if __name__ == "__main__":
    test_environment()