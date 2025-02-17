import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


class EnhancedSlimeMold:
    def __init__(self, size=500, num_nutrients=200, num_agents=1000):
        self.size = size
        self.nutrient_counter = 0  # 营养物ID生成器

        # 初始化营养物（包含位置、生命周期、唯一ID）
        self.nutrients = np.zeros(num_nutrients, dtype=[
            ('position', 'f8', 2),
            ('lifetime', 'i4'),
            ('active', 'bool'),
            ('id', 'i4')
        ])
        self.nutrients['position'] = np.random.rand(num_nutrients, 2) * size
        self.nutrients['lifetime'] = 0
        self.nutrients['active'] = False
        self.nutrients['id'] = np.arange(num_nutrients)
        self.nutrient_counter = num_nutrients

        # 初始化黏菌代理
        self.agents = np.random.rand(num_agents, 2) * size
        self.agent_angles = np.random.rand(num_agents) * 2 * np.pi
        self.agent_attached = np.full(num_agents, -1, dtype=int)  # 跟踪附着营养物ID

        self.trail_map = np.zeros((size, size))
        self.nutrient_tree = cKDTree(self.nutrients['position'])

        # 参数配置（保持不变）
        self.params = {
            'nutrient': {
                'activation_range': 3,  # 触发激活的接触距离
                'max_lifetime': 45  # 最大存在时间（帧）
            },
            'sensor': {
                'angle': np.pi / 4,  # 传感器角度
                'distance': 7  # 传感器间距
            },
            'movement': {
                'attached_step': 2.8,  # 附着时的移动步长
                'free_step': 0.7,  # 自由状态的移动步长
                'max_rotate': np.pi / 8  # 最大旋转角度
            },
            'pheromone': {
                'attached_deposit': 500,  # 附着状态沉积量
                'free_deposit': 100,  # 自由状态沉积量
                'decay': 0.95  # 信息素衰减系数
            },
            'reproduction': {
                'attached_prob': 0.1,  # 附着时分裂概率
                'free_prob': 0.005,  # 自由时分裂概率
                'max_branch_angle': np.pi / 1.5
            },
            'survival': {
                'attached_survival': 1.0,  # 附着存活率
                'free_survival': 0.93  # 自由存活率
            },
            'gaussian_blur': 1  # 可视化模糊参数
        }

    def batch_sense(self, positions, angles):
        """批量计算传感器位置（保持不变）"""
        offsets = np.array([-self.params['sensor']['angle'],
                            0,
                            self.params['sensor']['angle']])
        sensor_angles = angles[:, None] + offsets
        sensor_pos = positions[:, None] + \
                     self.params['sensor']['distance'] * np.stack(
            [np.cos(sensor_angles), np.sin(sensor_angles)], axis=2)
        return sensor_pos.reshape(-1, 2)

    def update_nutrient_states(self):
        """更新营养物状态并处理过期营养（保持不变）"""
        active_mask = self.nutrients['active']
        self.nutrients['lifetime'][active_mask] += 1

        expire_mask = (self.nutrients['lifetime'] > self.params['nutrient']['max_lifetime']) | \
                      (self.nutrients['position'][:, 0] < 0)
        expired_nutrients = self.nutrients[expire_mask]

        if len(expired_nutrients) > 0:
            expired_ids = expired_nutrients['id']
            attached_mask = np.isin(self.agent_attached, expired_ids)
            self.agent_attached[attached_mask] = -1

        self.nutrients = self.nutrients[~expire_mask]
        if len(self.nutrients) > 0:
            self.nutrient_tree = cKDTree(self.nutrients['position'])

    def update_agents(self):
        """更新代理状态（修正坐标存储顺序）"""
        new_agents = []

        sensor_positions = self.batch_sense(self.agents, self.agent_angles)
        distances, indices = self.nutrient_tree.query(sensor_positions, distance_upper_bound=self.size)
        sensor_values = np.zeros((len(self.agents), 3))

        for i in range(len(self.agents)):
            for j in range(3):
                idx = i * 3 + j
                if indices[idx] >= len(self.nutrients):
                    continue

                pos = sensor_positions[idx]
                x, y = np.clip(pos.astype(int), 0, self.size - 1)
                # 调整坐标访问顺序
                sensor_values[i, j] = self.trail_map[y, x]  # 改为[y,x]

                nutrient_dist = distances[idx]
                sensor_values[i, j] += max(40 - nutrient_dist, 0)

        for i in range(len(self.agents)):
            pos = self.agents[i]
            angle = self.agent_angles[i]
            nutrient_id = self.agent_attached[i]
            attached = nutrient_id != -1

            if sensor_values[i, 1] < sensor_values[i, 0] and \
                    sensor_values[i, 1] < sensor_values[i, 2]:
                rotate_dir = 1 if sensor_values[i, 0] > sensor_values[i, 2] else -1
                angle += rotate_dir * self.params['movement']['max_rotate']
            else:
                angle += np.random.uniform(-1, 1) * self.params['movement']['max_rotate']

            step = self.params['movement']['attached_step'] if attached \
                else self.params['movement']['free_step']
            new_pos = pos + step * np.array([np.cos(angle), np.sin(angle)])
            new_pos = np.clip(new_pos, 0, self.size - 1)

            if len(self.nutrients) > 0:
                dist, idx = self.nutrient_tree.query(new_pos)
                if dist < self.params['nutrient']['activation_range']:
                    contact_id = self.nutrients[idx]['id']
                    self.agent_attached[i] = contact_id

                    if not self.nutrients[idx]['active']:
                        self.nutrients[idx]['active'] = True
                        self.nutrients[idx]['lifetime'] = 0

            deposit = self.params['pheromone']['attached_deposit'] if attached \
                else self.params['pheromone']['free_deposit']
            # 修正坐标存储顺序
            x, y = new_pos.astype(int)  # 改为先x后y
            x = np.clip(x, 0, self.size-1)
            y = np.clip(y, 0, self.size-1)
            self.trail_map[y, x] += deposit  # 存储顺序[y,x]

            branch_prob = self.params['reproduction']['attached_prob'] if attached \
                else self.params['reproduction']['free_prob']
            if np.random.rand() < branch_prob:
                branch_angle = angle + np.random.uniform(
                    -self.params['reproduction']['max_branch_angle'],
                    self.params['reproduction']['max_branch_angle']
                )
                new_agents.append((new_pos.copy(), branch_angle, nutrient_id))

            self.agents[i] = new_pos
            self.agent_angles[i] = angle % (2 * np.pi)

        if new_agents:
            new_pos, new_angles, parent_ids = zip(*new_agents)
            self.agents = np.vstack([self.agents, np.array(new_pos)])
            self.agent_angles = np.concatenate([self.agent_angles, new_angles])
            self.agent_attached = np.concatenate([
                self.agent_attached,
                np.array(parent_ids)
            ])

        survival_probs = np.where(
            self.agent_attached != -1,
            self.params['survival']['attached_survival'],
            self.params['survival']['free_survival']
        )
        survive_mask = np.random.rand(len(self.agents)) < survival_probs
        self._filter_agents(survive_mask)

    def _filter_agents(self, mask):
        """筛选存活的代理（保持不变）"""
        self.agents = self.agents[mask]
        self.agent_angles = self.agent_angles[mask]
        self.agent_attached = self.agent_attached[mask]

    def update(self, frame):
        """更新动画帧（修正坐标系）"""
        self.trail_map *= self.params['pheromone']['decay']
        self.update_agents()
        self.update_nutrient_states()

        self.ax.clear()
        processed_map = gaussian_filter(
            np.log1p(self.trail_map),
            sigma=self.params['gaussian_blur']
        )

        # 添加origin='lower'修正坐标系方向
        self.ax.imshow(
            processed_map,
            cmap='inferno',
            alpha=0.98,
            vmin=0.1,
            vmax=5.0,
            interpolation='hermite',
            rasterized=True,
            origin='lower'  # 新增关键参数
        )

        if len(self.nutrients) > 0:
            active_nutrients = self.nutrients[self.nutrients['active']]
            fading_nutrients = self.nutrients[~self.nutrients['active']]

            if len(active_nutrients) > 0:
                self.ax.scatter(
                    active_nutrients['position'][:, 0],
                    active_nutrients['position'][:, 1],
                    c='#FF4500',
                    s=50 * (1 - active_nutrients['lifetime'] / self.params['nutrient']['max_lifetime']),
                    edgecolors='#FFD70088',
                    linewidths=2,
                    alpha=0.9,
                    zorder=3,
                    marker='*'
                )

            if len(fading_nutrients) > 0:
                self.ax.scatter(
                    fading_nutrients['position'][:, 0],
                    fading_nutrients['position'][:, 1],
                    c='#00FF7F',
                    s=30,
                    alpha=0.6,
                    zorder=2,
                    marker='o'
                )

        stats_text = (
            f"Frame: {frame}\n"
            f"Agents: {len(self.agents):,}\n"
            f"Active Nutrients: {np.sum(self.nutrients['active'])}\n"
            f"Attached Agents: {np.sum(self.agent_attached != -1):,}"
        )
        self.ax.text(
            0.02, 0.95, stats_text,
            transform=self.ax.transAxes,
            color='white',
            fontsize=8,
            verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

        return self.ax,

    def run_animation(self):
        """运行模拟动画（保持不变）"""
        self.fig, self.ax = plt.subplots(
            figsize=(9, 9),
            facecolor='#0F0F0F',
            dpi=150
        )
        self.ax.set_facecolor('#0F0F0F')
        self.ax.axis('off')
        plt.tight_layout()

        ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=300, interval=33,
            blit=True, cache_frame_data=False
        )
        plt.show()


if __name__ == "__main__":
    sim = EnhancedSlimeMold(500, 100, 100)
    sim.run_animation()