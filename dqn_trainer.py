import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import json
import os
from datetime import datetime


class DQNTrainer:
    """改进的DQN训练器 - 支持动作掩码"""

    def __init__(self, env, agent, save_dir="./models"):
        """
        初始化训练器
        Args:
            env: CAVBroadcastEnv环境
            agent: MaskedDQNAgent智能体
            save_dir: 模型保存目录
        """
        self.env = env
        self.agent = agent
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.sharing_rates = []
        self.broadcast_counts = []
        self.information_values = []
        self.objective_values = []
        self.convergence_window = 100
        self.convergence_threshold = 0.01

        # 最佳模型记录
        self.best_objective_value = -float('inf')
        self.best_episode = 0

    def train(self, max_episodes=5000, eval_frequency=100, save_frequency=500,
              early_stopping=True, patience=300):
        """
        训练DQN智能体
        """
        print("开始改进的DQN训练...")
        print(f"最大episode数: {max_episodes}")
        print(f"环境信息: CAV数={self.env.num_cavs}, 最大目标数/CAV={self.env.max_targets_per_cav}")
        print("-" * 50)

        start_time = time.time()
        no_improvement_count = 0

        for episode in range(max_episodes):
            # 执行一个episode
            total_reward, episode_length, episode_info = self._run_episode(training=True)

            # 记录统计信息
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.sharing_rates.append(episode_info['sharing_rate'])
            self.broadcast_counts.append(episode_info['total_broadcasts'])
            self.information_values.append(episode_info['total_information_value'])
            self.objective_values.append(episode_info['objective_value'])

            # 更新智能体奖励历史
            self.agent.rewards_history.append(total_reward)

            # 检查是否为最佳模型
            if episode_info['objective_value'] > self.best_objective_value:
                self.best_objective_value = episode_info['objective_value']
                self.best_episode = episode
                self._save_best_model(episode)
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # 打印进度
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_sharing_rate = np.mean(self.sharing_rates[-50:])
                avg_broadcasts = np.mean(self.broadcast_counts[-50:])
                avg_objective = np.mean(self.objective_values[-50:])

                print(f"Episode {episode + 1:5d} | "
                      f"奖励: {total_reward:7.3f} | "
                      f"共享率: {episode_info['sharing_rate']:5.3f} | "
                      f"广播数: {episode_info['total_broadcasts']:3d} | "
                      f"目标值: {episode_info['objective_value']:7.3f} | "
                      f"ε: {self.agent.epsilon:.3f}")

                print(f"         最近50期平均 - 奖励: {avg_reward:7.3f} | "
                      f"共享率: {avg_sharing_rate:5.3f} | "
                      f"广播数: {avg_broadcasts:5.1f} | "
                      f"目标值: {avg_objective:7.3f}")

                # 广播效率分析
                total_possible_broadcasts = sum([len(targets) for targets in self.env.cav_targets.values()])
                if total_possible_broadcasts > 0:
                    broadcast_efficiency = avg_broadcasts / total_possible_broadcasts
                    print(f"         广播效率: {broadcast_efficiency:.1%} | "
                          f"平均重复: {episode_info.get('avg_broadcasts_per_target', 0):.2f}")
                print("-" * 80)

            # 评估模型
            if (episode + 1) % eval_frequency == 0:
                self._evaluate_model(num_episodes=10)

            # 保存模型
            if (episode + 1) % save_frequency == 0:
                self._save_checkpoint(episode + 1)

            # 检查收敛性
            if episode >= self.convergence_window and self._check_convergence():
                print(f"\n模型在第 {episode + 1} episode收敛!")
                break

            # 早停检查
            if early_stopping and no_improvement_count >= patience:
                print(f"\n{patience} episodes无改进，触发早停!")
                break

        training_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {training_time:.2f}秒")
        print(f"最佳目标函数值: {self.best_objective_value:.3f} (第{self.best_episode + 1}期)")

        # 保存最终结果
        self._save_training_results()

        return self.episode_rewards, self.objective_values

    def _run_episode(self, training=True):
        """运行一个episode - 支持动作掩码"""
        state = self.env.reset()
        total_reward = 0
        episode_length = 0

        while True:
            # 获取动作掩码
            action_mask = self.env.get_action_mask()

            # 选择动作（考虑掩码）
            action = self.agent.select_action(state, action_mask, training=training)

            # 执行动作
            next_state, reward, done, info = self.env.step(action)

            # 存储经验(仅在训练时)
            if training:
                self.agent.store_experience(state, action, reward, next_state, done, action_mask)
                # 更新网络
                self.agent.update()

            total_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        # 获取episode总结
        episode_info = self.env.get_episode_summary()

        return total_reward, episode_length, episode_info

    def _evaluate_model(self, num_episodes=10):
        """评估模型性能"""
        print(f"\n=== 模型评估 (共{num_episodes}次) ===")

        eval_rewards = []
        eval_sharing_rates = []
        eval_broadcasts = []
        eval_objectives = []
        eval_info_values = []
        meets_threshold_count = 0

        for _ in range(num_episodes):
            reward, _, info = self._run_episode(training=False)
            eval_rewards.append(reward)
            eval_sharing_rates.append(info['sharing_rate'])
            eval_broadcasts.append(info['total_broadcasts'])
            eval_objectives.append(info['objective_value'])
            eval_info_values.append(info['total_information_value'])

            if info['meets_sharing_threshold']:
                meets_threshold_count += 1

        # 计算统计指标
        avg_reward = np.mean(eval_rewards)
        avg_sharing_rate = np.mean(eval_sharing_rates)
        avg_broadcasts = np.mean(eval_broadcasts)
        avg_objective = np.mean(eval_objectives)
        avg_info_value = np.mean(eval_info_values)
        threshold_rate = meets_threshold_count / num_episodes

        print(f"平均奖励: {avg_reward:.3f} ± {np.std(eval_rewards):.3f}")
        print(f"平均共享率: {avg_sharing_rate:.3f} ± {np.std(eval_sharing_rates):.3f}")
        print(f"平均广播数: {avg_broadcasts:.1f} ± {np.std(eval_broadcasts):.1f}")
        print(f"平均信息价值: {avg_info_value:.3f} ± {np.std(eval_info_values):.3f}")
        print(f"平均目标值: {avg_objective:.3f} ± {np.std(eval_objectives):.3f}")
        print(f"满足共享率阈值比例: {threshold_rate:.1%}")

        # 广播效率分析
        total_possible = sum([len(targets) for targets in self.env.cav_targets.values()])
        if total_possible > 0:
            efficiency = avg_broadcasts / total_possible
            print(f"广播效率: {efficiency:.1%} (实际有效目标比例)")
        print("=" * 40)

    def _check_convergence(self):
        """检查是否收敛"""
        if len(self.objective_values) < self.convergence_window:
            return False

        recent_values = self.objective_values[-self.convergence_window:]
        std_dev = np.std(recent_values)

        return std_dev < self.convergence_threshold

    def _save_best_model(self, episode):
        """保存最佳模型"""
        model_path = os.path.join(self.save_dir, "best_model.pth")
        self.agent.save(model_path)

        # 保存最佳模型信息
        best_info = {
            'episode': int(episode + 1),
            'objective_value': float(self.best_objective_value),
            'timestamp': datetime.now().isoformat(),
            'environment_config': {
                'num_cavs': self.env.num_cavs,
                'max_targets_per_cav': self.env.max_targets_per_cav,
                'sharing_threshold': self.env.sharing_threshold,
                'weights': [self.env.w1, self.env.w2, self.env.w3]
            }
        }

        info_path = os.path.join(self.save_dir, "best_model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(best_info, f, indent=2, ensure_ascii=False)

    def _save_checkpoint(self, episode):
        """保存训练检查点"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_ep{episode}.pth")
        self.agent.save(checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

    def _save_training_results(self):
        """保存训练结果"""
        results = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_lengths': [int(x) for x in self.episode_lengths],
            'sharing_rates': [float(x) for x in self.sharing_rates],
            'broadcast_counts': [int(x) for x in self.broadcast_counts],
            'information_values': [float(x) for x in self.information_values],
            'objective_values': [float(x) for x in self.objective_values],
            'best_objective_value': float(self.best_objective_value),
            'best_episode': int(self.best_episode + 1),
            'total_episodes': len(self.episode_rewards),
            'final_epsilon': float(self.agent.epsilon),
            'environment_info': {
                'num_cavs': self.env.num_cavs,
                'max_targets_per_cav': self.env.max_targets_per_cav,
                'sharing_threshold': self.env.sharing_threshold,
                'max_distance': self.env.max_distance
            }
        }

        results_path = os.path.join(self.save_dir, "training_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"训练结果已保存到: {results_path}")

    def plot_training_progress(self, save_plots=True):
        """绘制训练进度图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('改进DQN训练进度', fontsize=16)

        # 1. 奖励曲线
        axes[0, 0].plot(self.episode_rewards, alpha=0.7)
        if len(self.episode_rewards) > 50:
            window_size = min(100, len(self.episode_rewards) // 10)
            moving_avg = np.convolve(self.episode_rewards,
                                     np.ones(window_size) / window_size,
                                     mode='valid')
            axes[0, 0].plot(range(window_size - 1, len(self.episode_rewards)),
                            moving_avg, 'r-', linewidth=2, label='移动平均')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode奖励')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('总奖励')
        axes[0, 0].grid(True)

        # 2. 共享率
        axes[0, 1].plot(self.sharing_rates, 'g-', alpha=0.7)
        axes[0, 1].axhline(y=self.env.sharing_threshold, color='r', linestyle='--',
                           label=f'目标阈值 ({self.env.sharing_threshold})')
        axes[0, 1].set_title('共享率')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('共享率')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 3. 广播数量
        axes[0, 2].plot(self.broadcast_counts, 'orange', alpha=0.7)
        # 计算实际可能的广播数
        total_possible = sum([len(targets) for targets in self.env.cav_targets.values()])
        ideal_broadcasts = total_possible * 0.5  # 假设理想情况是50%的广播率
        axes[0, 2].axhline(y=ideal_broadcasts, color='r', linestyle='--',
                           label=f'参考线 (~{ideal_broadcasts:.0f})')
        axes[0, 2].set_title('广播数量')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('总广播数')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # 4. 信息价值
        axes[1, 0].plot(self.information_values, 'purple', alpha=0.7)
        axes[1, 0].set_title('信息价值')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('总信息价值')
        axes[1, 0].grid(True)

        # 5. 目标函数值
        axes[1, 1].plot(self.objective_values, 'brown', alpha=0.7)
        if len(self.objective_values) > 50:
            window_size = min(100, len(self.objective_values) // 10)
            moving_avg = np.convolve(self.objective_values,
                                     np.ones(window_size) / window_size,
                                     mode='valid')
            axes[1, 1].plot(range(window_size - 1, len(self.objective_values)),
                            moving_avg, 'r-', linewidth=2, label='移动平均')
            axes[1, 1].legend()
        axes[1, 1].set_title('目标函数值')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('目标函数值')
        axes[1, 1].grid(True)

        # 6. 训练损失
        if self.agent.losses:
            axes[1, 2].plot(self.agent.losses, 'red', alpha=0.7)
            axes[1, 2].set_title('训练损失')
            axes[1, 2].set_xlabel('更新步数')
            axes[1, 2].set_ylabel('MSE损失')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True)

        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(self.save_dir, "training_progress.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"训练进度图已保存到: {plot_path}")

        plt.show()

    def test_policy(self, num_episodes=50, verbose=True):
        """测试训练好的策略"""
        print(f"\n=== 策略测试 (共{num_episodes}次) ===")

        test_results = {
            'rewards': [],
            'sharing_rates': [],
            'broadcast_counts': [],
            'information_values': [],
            'objective_values': [],
            'meets_threshold': []
        }

        for episode in range(num_episodes):
            reward, _, info = self._run_episode(training=False)

            test_results['rewards'].append(reward)
            test_results['sharing_rates'].append(info['sharing_rate'])
            test_results['broadcast_counts'].append(info['total_broadcasts'])
            test_results['information_values'].append(info['total_information_value'])
            test_results['objective_values'].append(info['objective_value'])
            test_results['meets_threshold'].append(info['meets_sharing_threshold'])

            if verbose and (episode + 1) % 10 == 0:
                print(f"测试 {episode + 1:2d}/{num_episodes}: 奖励={reward:6.3f}, "
                      f"共享率={info['sharing_rate']:.3f}, "
                      f"广播数={info['total_broadcasts']:2d}, "
                      f"目标值={info['objective_value']:6.3f}")

        # 计算统计结果
        stats = {}
        for key, values in test_results.items():
            if key != 'meets_threshold':
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        threshold_success_rate = np.mean(test_results['meets_threshold'])

        print(f"\n=== 测试结果统计 ===")
        print(f"满足共享率阈值成功率: {threshold_success_rate:.1%}")
        print(f"平均奖励: {stats['rewards']['mean']:.3f} ± {stats['rewards']['std']:.3f}")
        print(f"平均共享率: {stats['sharing_rates']['mean']:.3f} ± {stats['sharing_rates']['std']:.3f}")
        print(f"平均广播数: {stats['broadcast_counts']['mean']:.1f} ± {stats['broadcast_counts']['std']:.1f}")
        print(f"平均目标函数值: {stats['objective_values']['mean']:.3f} ± {stats['objective_values']['std']:.3f}")

        # 广播效率分析
        total_possible = sum([len(targets) for targets in self.env.cav_targets.values()])
        if total_possible > 0:
            broadcast_efficiency = stats['broadcast_counts']['mean'] / total_possible
            print(f"广播效率: {broadcast_efficiency:.1%}")

        return test_results, stats


def create_optimized_config():
    """创建优化的训练配置"""
    config = {
        'environment': {
            'w1': 0.4,  # 信息价值权重
            'w2': 1.0,  # 广播代价权重
            'w3': 0.3,  # 共享率激励权重
            'sharing_threshold': 0.95,
            'max_distance': 150.0
        },
        'agent': {
            'lr': 5e-4,  # 降低学习率提高稳定性
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,  # 保持一定探索
            'epsilon_decay': 0.998,  # 更慢的衰减
            'buffer_size': 50000,
            'batch_size': 128,  # 增加批大小
            'target_update': 500,  # 更频繁更新目标网络
            'hidden_dims': [512, 256, 128]  # 增加网络容量
        },
        'training': {
            'max_episodes': 3000,
            'eval_frequency': 50,
            'save_frequency': 200,
            'early_stopping': True,
            'patience': 200
        }
    }

    return config


if __name__ == "__main__":
    print("改进的DQN训练器 - 支持动作掩码版本")
    print("主要改进:")
    print("1. 支持动作掩码，避免无效动作干扰")
    print("2. 改进的奖励计算和效率监控")
    print("3. 优化的训练配置和收敛检测")
    print("4. 详细的性能分析和可视化")