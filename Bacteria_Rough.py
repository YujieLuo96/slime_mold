import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


class SlimeMoldSimulation:
    def __init__(self, size=500, num_nutrients=200, num_agents=1000):
        self.size = size
        self.nutrients = np.random.rand(num_nutrients, 2) * size
        self.agents = np.random.rand(num_agents, 2) * size
        self.agent_angles = np.random.rand(num_agents) * 2 * np.pi
        self.trail_map = np.zeros((size, size))
        self.nutrient_tree = cKDTree(self.nutrients)

        # 参数设置
        self.sensor_angle = np.pi / 4  # 传感器角度
        self.sensor_distance = 5  # 传感器间距
        self.rotation_angle = np.pi / 8  # 最大旋转角度
        self.step_size = 2.5  # 移动步长
        self.deposit_amount = 20  # 信息素沉积量
        self.decay_factor = 0.999  # 信息素衰减系数
        self.branch_prob = 0.05  # 分叉概率

    def sense_environment(self, agent_pos, angle):
        sensor_pos = agent_pos + self.sensor_distance * np.array([
            [np.cos(angle - self.sensor_angle), np.sin(angle - self.sensor_angle)],
            [np.cos(angle), np.sin(angle)],
            [np.cos(angle + self.sensor_angle), np.sin(angle + self.sensor_angle)]
        ])

        sensor_values = []
        for pos in sensor_pos:
            x, y = pos.astype(int)
            x = np.clip(x, 0, self.size - 1)
            y = np.clip(y, 0, self.size - 1)
            sensor_values.append(self.trail_map[x, y])

            # 添加营养吸引因子
            _, idx = self.nutrient_tree.query(pos)
            nutrient_dist = np.linalg.norm(pos - self.nutrients[idx])
            sensor_values[-1] += max(50 - nutrient_dist, 0)

        return sensor_values

    def update_agents(self):
        new_agents = []
        for i in range(len(self.agents)):
            pos = self.agents[i]
            angle = self.agent_angles[i]

            # 传感器检测
            sensor_values = self.sense_environment(pos, angle)

            # 计算转向角度
            if sensor_values[1] < sensor_values[0] and sensor_values[1] < sensor_values[2]:
                angle += self.rotation_angle * (1 if sensor_values[0] > sensor_values[2] else -1)
            else:
                angle += self.rotation_angle * np.random.uniform(-1, 1)

            # 移动并更新位置
            new_pos = pos + self.step_size * np.array([np.cos(angle), np.sin(angle)])
            new_pos = np.clip(new_pos, 0, self.size - 1)

            # 检查是否接触营养
            dist, idx = self.nutrient_tree.query(new_pos)
            if dist < 3:
                self.nutrients = np.delete(self.nutrients, idx, axis=0)
                self.nutrient_tree = cKDTree(self.nutrients)

            # 更新信息素地图
            x, y = new_pos.astype(int)
            self.trail_map[x, y] += self.deposit_amount

            # 分叉逻辑
            if np.random.rand() < self.branch_prob:
                new_angle = angle + np.random.uniform(-np.pi / 2, np.pi / 2)
                new_agents.append((new_pos.copy(), new_angle))

            self.agents[i] = new_pos
            self.agent_angles[i] = angle

        # 添加新分叉
        if new_agents:
            new_pos, new_angles = zip(*new_agents)
            self.agents = np.vstack([self.agents, np.array(new_pos)])
            self.agent_angles = np.concatenate([self.agent_angles, new_angles])

    def update(self, frame):
        self.trail_map *= self.decay_factor  # 信息素衰减
        self.update_agents()

        # 清空画布并重新绘制
        self.ax.clear()
        self.ax.imshow(gaussian_filter(self.trail_map, sigma=1),
                       cmap='plasma', alpha=0.8)
        self.ax.scatter(self.nutrients[:, 0], self.nutrients[:, 1],
                        c='lime', s=10, alpha=0.6)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(f'Slime Mold Simulation - Frame {frame}')
        return self.ax,

    def run_animation(self):
        fig, self.ax = plt.subplots(figsize=(8, 8))
        ani = animation.FuncAnimation(fig, self.update,
                                      frames=500, interval=50,
                                      blit=True)
        plt.show()


# 运行模拟
sim = SlimeMoldSimulation(size=500, num_nutrients=200, num_agents=1)
sim.run_animation()