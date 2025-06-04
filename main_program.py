#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAV协同感知广播优化DQN主程序
整合数据加载、环境创建、智能体训练和结果分析
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 导入自定义模块
from data_loader import MATLABDataLoader
from cav_broadcast_env import CAVBroadcastEnv
from dqn_agent import MaskedDQNAgent
from dqn_trainer import DQNTrainer, create_optimized_config


def setup_matplotlib():
    """设置matplotlib中文显示"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def load_and_validate_data(file_path):
    """加载并验证数据"""
    print("=" * 60)
    print("步骤1: 数据加载与验证")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到数据文件 {file_path}")
        print("请确保.mat文件在正确路径下")
        return None

    # 创建数据加载器
    loader = MATLABDataLoader(file_path)

    # 加载数据
    if loader.load_data():
        print("✅ 数据加载成功!")
        loader.print_cav_info()
        return loader
    else:
        print("❌ 数据加载失败!")
        return None


def create_environment_and_agent(loader, config):
    """创建环境和智能体"""
    print("\n" + "=" * 60)
    print("步骤2: 环境与智能体创建")
    print("=" * 60)

    # 创建环境，包含 w3 参数
    env = CAVBroadcastEnv(
        data_loader=loader,
        w1=config['environment']['w1'],
        w2=config['environment']['w2'],
        w3=config['environment']['w3'],
        sharing_threshold=config['environment']['sharing_threshold'],
        max_distance=config['environment']['max_distance']
    )

    print(f"✅ 环境创建成功!")
    print(f"   - CAV数量: {env.num_cavs}")
    print(f"   - 最大目标数/CAV: {env.max_targets_per_cav}")
    print(f"   - 共享率阈值: {env.sharing_threshold}")
    print(f"   - 权重 w1={env.w1}, w2={env.w2}, w3={env.w3}")

    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.max_targets_per_cav

    print(f"   - 状态空间维度: {state_dim}")
    print(f"   - 动作空间维度: {action_dim}")

    # 创建智能体，使用 MaskedDQNAgent
    agent = MaskedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=config['agent']['lr'],
        gamma=config['agent']['gamma'],
        epsilon_start=config['agent']['epsilon_start'],
        epsilon_end=config['agent']['epsilon_end'],
        epsilon_decay=config['agent']['epsilon_decay'],
        buffer_size=config['agent']['buffer_size'],
        batch_size=config['agent']['batch_size'],
        target_update=config['agent']['target_update']
    )

    print(f"✅ DQN智能体创建成功!")
    print(f"   - 学习率: {config['agent']['lr']}")
    print(f"   - 缓冲区大小: {config['agent']['buffer_size']}")

    return env, agent


def train_model(env, agent, config, save_dir):
    """训练模型"""
    print("\n" + "=" * 60)
    print("步骤3: DQN模型训练")
    print("=" * 60)

    # 创建训练器
    trainer = DQNTrainer(env, agent, save_dir=save_dir)

    # 开始训练
    episode_rewards, objective_values = trainer.train(
        max_episodes=config['training']['max_episodes'],
        eval_frequency=config['training']['eval_frequency'],
        save_frequency=config['training']['save_frequency'],
        early_stopping=config['training']['early_stopping'],
        patience=config['training']['patience']
    )

    print("✅ 模型训练完成!")

    return trainer


def analyze_results(trainer, save_dir):
    """分析训练结果"""
    print("\n" + "=" * 60)
    print("步骤4: 结果分析")
    print("=" * 60)

    # 绘制训练进度
    trainer.plot_training_progress(save_plots=True)

    # 测试最终策略
    test_results, stats = trainer.test_policy(num_episodes=100, verbose=True)

    # 生成分析报告
    generate_analysis_report(trainer, test_results, stats, save_dir)

    return test_results, stats


def generate_analysis_report(trainer, test_results, stats, save_dir):
    """生成分析报告"""
    report = {
        'training_summary': {
            'total_episodes': len(trainer.episode_rewards),
            'best_episode': trainer.best_episode + 1,
            'best_objective_value': trainer.best_objective_value,
            'final_epsilon': trainer.agent.epsilon,
            'convergence_achieved': len(trainer.objective_values) >= trainer.convergence_window
        },
        'test_performance': {
            'success_rate': np.mean(test_results['meets_threshold']),
            'average_sharing_rate': stats['sharing_rates']['mean'],
            'average_broadcast_count': stats['broadcast_counts']['mean'],
            'average_objective_value': stats['objective_values']['mean'],
            'objective_std': stats['objective_values']['std']
        },
        'optimization_analysis': {
            'sharing_threshold_met': stats['sharing_rates']['mean'] >= trainer.env.sharing_threshold,
            'broadcast_efficiency': stats['broadcast_counts']['mean'] / trainer.env.max_targets_per_cav,
            'information_value_per_broadcast': stats['information_values']['mean'] / max(
                stats['broadcast_counts']['mean'], 1)
        }
    }

    # 保存报告
    report_path = os.path.join(save_dir, "analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印关键结果
    print("\n📊 关键性能指标:")
    print(f"   - 共享率阈值满足率: {report['test_performance']['success_rate']:.1%}")
    print(f"   - 平均共享率: {report['test_performance']['average_sharing_rate']:.3f}")
    print(f"   - 平均广播数: {report['test_performance']['average_broadcast_count']:.1f}")
    print(f"   - 平均目标函数值: {report['test_performance']['average_objective_value']:.3f}")
    print(f"   - 广播效率: {report['optimization_analysis']['broadcast_efficiency']:.1%}")

    print(f"\n📋 分析报告已保存到: {report_path}")


def save_config(config, save_dir):
    """保存配置文件"""
    config_path = os.path.join(save_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"📁 配置文件已保存到: {config_path}")


def main():
    """主函数"""
    print("🚗 CAV协同感知广播优化DQN系统")
    print("=" * 60)

    # 设置matplotlib
    setup_matplotlib()

    # 配置参数
    config = create_optimized_config()

    # 文件路径 - 请根据实际情况修改
    data_file_path = "vehicles_density50_penetration0.5.mat"

    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./results/cav_dqn_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 步骤1: 加载数据
        loader = load_and_validate_data(data_file_path)
        if loader is None:
            return

        # 保存配置
        save_config(config, save_dir)

        # 步骤2: 创建环境和智能体
        env, agent = create_environment_and_agent(loader, config)

        # 步骤3: 训练模型
        trainer = train_model(env, agent, config, save_dir)

        # 步骤4: 分析结果
        test_results, stats = analyze_results(trainer, save_dir)

        print("\n" + "=" * 60)
        print("🎉 系统运行完成!")
        print(f"📁 所有结果已保存到: {save_dir}")
        print("=" * 60)

        # 提供后续操作建议
        print("\n💡 后续操作建议:")
        print("1. 查看训练进度图和结果分析")
        print("2. 调整超参数重新训练")
        print("3. 在新数据集上测试模型泛化性能")
        print("4. 分析CAV个体广播策略")

    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()
        return


def quick_test():
    """快速测试(使用较少的训练轮数)"""
    print("🧪 快速测试模式")

    config = create_optimized_config()
    # 减少训练轮数用于测试
    config['training']['max_episodes'] = 100
    config['training']['eval_frequency'] = 20
    config['training']['save_frequency'] = 50

    # 减少网络复杂度
    config['agent']['hidden_dims'] = [128, 64]
    config['agent']['buffer_size'] = 10000

    print("⚙️  使用测试配置 (减少训练轮数)")

    # 运行主程序逻辑
    main()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CAV广播优化DQN系统')
    parser.add_argument('--test', action='store_true',
                        help='运行快速测试模式 (较少训练轮数)')
    parser.add_argument('--data', type=str,
                        default='vehicles_density50_penetration0.5.mat',
                        help='MATLAB数据文件路径')

    args = parser.parse_args()

    # 更新数据文件路径
    if hasattr(args, 'data') and args.data:
        # 这里可以修改全局变量或传递参数
        pass

    if args.test:
        quick_test()
    else:
        main()