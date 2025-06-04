import numpy as np
import scipy.io as sio
import os


class MATLABDataLoader:
    """MATLAB数据加载器"""

    def __init__(self, file_path):
        """
        初始化数据加载器
        Args:
            file_path: .mat文件路径
        """
        self.file_path = file_path
        self.data = None
        self.is_cav = None
        self.los_matrix = None
        self.positions = None
        self.vehicle_ids = None
        self.num_vehicles = 0
        self.num_cavs = 0

    def load_data(self):
        """加载MATLAB数据文件"""
        try:
            # 读取.mat文件
            self.data = sio.loadmat(self.file_path)
            print(f"成功读取文件: {self.file_path}")
            print(f"文件中包含的变量: {list(self.data.keys())}")

            # 提取各个变量
            self.is_cav = self.data['is_cav'].flatten()  # 转为1维数组
            self.los_matrix = self.data['los_matrix']
            self.positions = self.data['positions']
            self.vehicle_ids = self.data['vehicle_ids'].flatten()

            # 计算基本信息
            self.num_vehicles = len(self.is_cav)
            self.num_cavs = np.sum(self.is_cav)

            print(f"\n数据基本信息:")
            print(f"总车辆数: {self.num_vehicles}")
            print(f"CAV数量: {self.num_cavs}")
            print(f"普通车数量: {self.num_vehicles - self.num_cavs}")
            print(f"CAV渗透率: {self.num_cavs / self.num_vehicles:.2%}")

            # 验证数据
            self._validate_data()

            return True

        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.file_path}")
            print("请确保文件路径正确")
            return False
        except Exception as e:
            print(f"读取数据时出错: {e}")
            return False

    def _validate_data(self):
        """验证数据完整性"""
        print(f"\n数据验证:")
        print(f"is_cav形状: {self.is_cav.shape}")
        print(f"los_matrix形状: {self.los_matrix.shape}")
        print(f"positions形状: {self.positions.shape}")
        print(f"vehicle_ids形状: {self.vehicle_ids.shape}")

        # 检查数据一致性
        assert self.los_matrix.shape[0] == self.los_matrix.shape[1], "LOS矩阵应该是方阵"
        assert self.los_matrix.shape[0] == self.num_vehicles, "LOS矩阵大小与车辆数不匹配"
        assert self.positions.shape[0] == self.num_vehicles, "位置数据与车辆数不匹配"
        assert self.positions.shape[1] == 2, "位置数据应该是2维坐标"

        print("✓ 数据验证通过")

    def get_cav_indices(self):
        """获取所有CAV的索引"""
        return np.where(self.is_cav == 1)[0]

    def get_regular_vehicle_indices(self):
        """获取所有普通车的索引"""
        return np.where(self.is_cav == 0)[0]

    def get_cav_los_matrix(self):
        """
        获取CAV的LOS矩阵
        返回: [CAV数量 × 全部车辆数] 的矩阵
        """
        cav_indices = self.get_cav_indices()
        cav_los_matrix = self.los_matrix[cav_indices, :]
        return cav_los_matrix, cav_indices

    def calculate_distances(self):
        """计算所有车辆间的距离矩阵"""
        distances = np.zeros((self.num_vehicles, self.num_vehicles))

        for i in range(self.num_vehicles):
            for j in range(self.num_vehicles):
                if i != j:
                    dist = np.sqrt(np.sum((self.positions[i] - self.positions[j]) ** 2))
                    distances[i, j] = dist

        return distances

    def print_cav_info(self):
        """打印CAV详细信息"""
        cav_indices = self.get_cav_indices()
        cav_los_matrix, _ = self.get_cav_los_matrix()

        print(f"\nCAV详细信息:")
        for i, cav_idx in enumerate(cav_indices):
            los_connections = np.where(cav_los_matrix[i] == 1)[0]
            print(f"CAV {cav_idx} (车辆ID: {int(self.vehicle_ids[cav_idx])}):")
            print(f"  位置: ({self.positions[cav_idx][0]:.2f}, {self.positions[cav_idx][1]:.2f})")
            print(f"  可感知车辆数: {len(los_connections)}")
            print(f"  可感知车辆ID: {[int(self.vehicle_ids[idx]) for idx in los_connections]}")
            print()

    def save_summary(self, output_file="data_summary.txt"):
        """保存数据摘要到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"数据文件: {self.file_path}\n")
            f.write(f"总车辆数: {self.num_vehicles}\n")
            f.write(f"CAV数量: {self.num_cavs}\n")
            f.write(f"普通车数量: {self.num_vehicles - self.num_cavs}\n")
            f.write(f"CAV渗透率: {self.num_cavs / self.num_vehicles:.2%}\n")

            # CAV详细信息
            cav_indices = self.get_cav_indices()
            cav_los_matrix, _ = self.get_cav_los_matrix()

            f.write(f"\nCAV详细信息:\n")
            for i, cav_idx in enumerate(cav_indices):
                los_connections = np.where(cav_los_matrix[i] == 1)[0]
                f.write(f"CAV {cav_idx}: 可感知{len(los_connections)}辆车\n")

        print(f"数据摘要已保存到: {output_file}")


def test_data_loader():
    """测试数据加载器"""
    # 这里你需要替换为你的实际文件路径
    file_path = "vehicles_density50_penetration0.5.mat"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在")
        print("请将你的.mat文件放在当前目录下，或修改file_path变量")
        return None

    # 创建数据加载器
    loader = MATLABDataLoader(file_path)

    # 加载数据
    if loader.load_data():
        # 打印CAV信息
        loader.print_cav_info()

        # 保存摘要
        loader.save_summary()

        return loader
    else:
        return None


if __name__ == "__main__":
    print("=== MATLAB数据加载器测试 ===")
    loader = test_data_loader()

    if loader is not None:
        print("\n✓ 数据加载成功！")
        print("接下来我们可以进行DQN环境搭建...")
    else:
        print("\n✗ 数据加载失败，请检查文件路径和文件格式")