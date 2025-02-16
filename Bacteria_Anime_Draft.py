import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


class OptimizedSlimeMold:
    def __init__(self, size=500, num_nutrients=200, num_agents=1000):
        self.size = size
        self.nutrients = np.random.rand(num_nutrients, 2) * size
        self.agents = np.random.rand(num_agents, 2) * size
        self.agent_angles = np.random.rand(num_agents) * 2 * np.pi
        self.trail_map = np.zeros((size, size))
        self.nutrient_tree = cKDTree(self.nutrients)

        # 参数设置
        self.sensor_angle = np.pi / 4  # 传感器角度
        self.sensor_distance = 7  # 传感器间距
        self.rotation_angle = np.pi / 10  # 最大旋转角度
        self.step_size = 2  # 移动步长
        self.deposit_amount = 100  # 信息素沉积量
        self.decay_factor = 0.95  # 信息素衰减系数
        self.branch_prob = 0.07  # 分叉概率
        self.sigma_blur = 1.6  # 高斯模糊参数

    def batch_sense(self, positions, angles):
        offsets = np.array([
            [-self.sensor_angle, 0, self.sensor_angle],
        ])
        sensor_angles = angles[:, None] + offsets
        sensor_pos = positions[:, None] + self.sensor_distance * np.stack(
            [np.cos(sensor_angles), np.sin(sensor_angles)], axis=2
        )
        return sensor_pos.reshape(-1, 2)

    def update_agents(self):
        consumed_nutrients = set()
        new_agents = []

        # 批量处理传感器位置
        sensor_positions = self.batch_sense(self.agents, self.agent_angles)
        distances, indices = self.nutrient_tree.query(sensor_positions)
        sensor_values = np.zeros((len(self.agents), 3))

        for i in range(len(self.agents)):
            # 计算三个传感器的值
            for j in range(3):
                idx = i * 3 + j
                pos = sensor_positions[idx]
                x, y = pos.astype(int)
                x = np.clip(x, 0, self.size - 1)
                y = np.clip(y, 0, self.size - 1)
                sensor_values[i, j] = self.trail_map[x, y]

                # 添加营养吸引因子
                nutrient_dist = distances[idx]
                sensor_values[i, j] += max(50 - nutrient_dist, 0)

            # 计算转向角度
            pos = self.agents[i]
            angle = self.agent_angles[i]
            if sensor_values[i, 1] < sensor_values[i, 0] and sensor_values[i, 1] < sensor_values[i, 2]:
                angle += self.rotation_angle * (1 if sensor_values[i, 0] > sensor_values[i, 2] else -1)
            else:
                angle += self.rotation_angle * np.random.uniform(-1, 1)

            # 移动并更新位置
            new_pos = pos + self.step_size * np.array([np.cos(angle), np.sin(angle)])
            new_pos = np.clip(new_pos, 0, self.size - 1)

            # 检查是否接触营养
            dist, idx = self.nutrient_tree.query(new_pos)
            if dist < 3:
                consumed_nutrients.add(idx)

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

        # 批量更新营养物
        if consumed_nutrients:
            mask = np.ones(len(self.nutrients), dtype=bool)
            mask[list(consumed_nutrients)] = False
            self.nutrients = self.nutrients[mask]
            self.nutrient_tree = cKDTree(self.nutrients)

    def update(self, frame):
        self.trail_map *= self.decay_factor
        self.update_agents()

        # 增强的可视化设置
        self.ax.clear()
        processed_map = gaussian_filter(self.trail_map ** 0.6, sigma=self.sigma_blur)

        img = self.ax.imshow(
            processed_map,
            cmap='inferno',  # 使用更高对比度的颜色映射
            alpha=0.98,  # 增加不透明度
            vmin=0.1,
            vmax=25,  # 降低最大值增强对比
            interpolation='hermite',  # 更清晰的插值方式
            rasterized=True
        )

        # 增强营养点显示
        scatter = self.ax.scatter(
            self.nutrients[:, 0], self.nutrients[:, 1],
            c='lime',  # 使用更鲜艳的颜色
            s=55,  # 增大尺寸
            edgecolors='#00ff0044',  # 半透明绿色光晕
            linewidths=2.5,  # 加粗边缘
            alpha=0.97,
            zorder=2,
            marker='*'  # 使用星形标记
        )

        # 动态调整颜色条
        if frame == 0 and not hasattr(self, 'cbar'):
            self.cbar = self.fig.colorbar(
                img,  # 使用img对象创建colorbar
                ax=self.ax,
                shrink=0.75,
                ticks=[0, 25],
                label='Pheromone Intensity'
            )
            self.cbar.outline.set_visible(False)

        self.ax.set_title(f'Optimized Slime Mold - Frame {frame}\nAgents: {len(self.agents):,}', fontsize=10)
        return [img, scatter]  # 返回需要更新的图形元素列表

    def run_animation(self, save_path=None):  # 添加保存路径参数
        self.fig, self.ax = plt.subplots(
            figsize=(10, 10),
            facecolor='#0f0f0f',  # 深色背景增强对比
            dpi=180  # 提高分辨率
        )
        self.ax.set_facecolor('#0f0f0f')
        plt.tight_layout()

        ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=300, interval=33,  # 30 FPS
            blit=True, cache_frame_data=False
        )

        if save_path:
            if save_path.endswith('.gif'):
                # 使用 Pillow 保存为 GIF
                ani.save(
                    save_path,
                    writer='pillow',  # 使用 Pillow 写入器
                    fps=30,
                    dpi=180
                )
                print(f"动画已保存至：{save_path}")
            else:
                print("警告：未安装 ffmpeg，仅支持保存为 GIF 文件。")
        else:
            plt.show()


# 运行优化后的模拟
sim = OptimizedSlimeMold(size=500, num_nutrients=15, num_agents=1)
sim.run_animation(save_path='slime_mold_simulation.gif')  # 保存为 GIF 文件