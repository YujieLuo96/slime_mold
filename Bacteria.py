import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


class EnhancedSlimeMold:
    def __init__(self, size=500, num_nutrients=200, num_agents=1000):
        self.size = size
        self.nutrient_counter = 0

        # 初始化营养物
        self.nutrients = np.zeros(num_nutrients, dtype=[
            ('position', 'f8', 2),
            ('lifetime', 'i4'),
            ('active', 'bool'),
            ('id', 'i4'),
            ('energy', 'i4')
        ])
        self.nutrients['position'] = np.random.rand(num_nutrients, 2) * size
        self.nutrients['lifetime'] = 0
        self.nutrients['active'] = False
        self.nutrients['id'] = np.arange(num_nutrients)
        self.nutrients['energy'] = 200
        self.nutrient_counter = num_nutrients

        # 初始化代理
        self.agents = np.random.rand(num_agents, 2) * size
        self.agent_angles = np.random.rand(num_agents) * 2 * np.pi
        self.agent_attached = np.full(num_agents, -1, dtype=int)
        self.agent_step_multiplier = np.ones(num_agents)  # 新增步长乘数

        self.trail_map = np.zeros((size, size))
        self.nutrient_tree = cKDTree(self.nutrients['position'])

        # 更新参数配置
        self.params = {
            'nutrient': {
                'activation_range': 3,
                'max_lifetime': 45,
                'consumption_radius': 8  # 新增消耗半径
            },
            'sensor': {
                'angle': np.pi / 4,
                'distance': 9
            },
            'movement': {
                'attached_step': 0.7,
                'free_step': 2.8,
                'max_rotate': np.pi / 5,
                'inactive_rotate': np.pi / 9  # 无营养时的旋转限制
            },
            'pheromone': {
                'attached_deposit': 1200,
                'free_deposit': 250,
                'decay': 0.95
            },
            'reproduction': {
                'attached_prob': 0.25,  # 提高附着分裂概率
                'free_prob': 0.01,
                'explore_prob': 0.2,  # 新增探索概率
                'max_branch_angle': np.pi / 1.5
            },
            'survival': {
                'attached_survival': 0.98,
                'free_survival': 0.91,
                'inactive_survival': 0.84  # 无营养时的存活率
            },
            'gaussian_blur': 1.2,
            'merge_distance': 3  # 代理合并距离
        }

    def batch_sense(self, positions, angles):
        """批量计算传感器位置"""
        offsets = np.array([-self.params['sensor']['angle'],
                            0,
                            self.params['sensor']['angle']])
        sensor_angles = angles[:, None] + offsets
        sensor_pos = positions[:, None] + \
                     self.params['sensor']['distance'] * np.stack(
            [np.cos(sensor_angles), np.sin(sensor_angles)], axis=2)
        return sensor_pos.reshape(-1, 2)

    def update_nutrient_states(self):
        """更新营养物状态"""
        active_mask = self.nutrients['active']
        self.nutrients['lifetime'][active_mask] += 1

        expire_mask = (self.nutrients['lifetime'] > self.params['nutrient']['max_lifetime']) | \
                      (self.nutrients['energy'] <= 0) | \
                      (self.nutrients['position'][:, 0] < 0)
        expired_nutrients = self.nutrients[expire_mask]

        if len(expired_nutrients) > 0:
            expired_ids = expired_nutrients['id']
            attached_mask = np.isin(self.agent_attached, expired_ids)
            self.agent_attached[attached_mask] = -1

        self.nutrients = self.nutrients[~expire_mask]
        if len(self.nutrients) > 0:
            self.nutrient_tree = cKDTree(self.nutrients['position'])

    def process_agent_merging(self):
        """处理代理合并（新功能）"""
        if len(self.agents) < 2:
            return

        agent_tree = cKDTree(self.agents)
        pairs = agent_tree.query_pairs(self.params['merge_distance'])

        for i, j in pairs:
            ai, aj = self.agent_attached[i], self.agent_attached[j]

            # 合并附着状态
            if ai != aj:
                if ai == -1 and aj != -1:
                    self.agent_attached[i] = aj
                elif aj == -1 and ai != -1:
                    self.agent_attached[j] = ai
                elif ai != -1 and aj != -1:
                    if np.random.rand() < 0.5:
                        self.agent_attached[j] = ai
                    else:
                        self.agent_attached[i] = aj

    def update_agents(self):
        """更新代理状态"""
        new_agents = []
        has_active_nutrients = np.any(self.nutrients['active'])

        # 动态参数调整
        if has_active_nutrients:
            move_params = self.params['movement']
            repro_params = self.params['reproduction']
            survival_params = self.params['survival']
        else:
            move_params = {
                'free_step': 0.8,
                'max_rotate': self.params['movement']['inactive_rotate']
            }
            repro_params = {'free_prob': 0.01}
            survival_params = {'free_survival': self.params['survival']['inactive_survival']}

        # 传感器处理
        sensor_positions = self.batch_sense(self.agents, self.agent_angles)
        if len(self.nutrients) > 0:
            distances, indices = self.nutrient_tree.query(sensor_positions, distance_upper_bound=self.size)
        else:
            distances, indices = np.full(len(sensor_positions), np.inf), []

        sensor_values = np.zeros((len(self.agents), 3))
        for i in range(len(self.agents)):
            for j in range(3):
                idx = i * 3 + j
                if idx >= len(indices):  # 防止索引越界
                    continue

                pos = sensor_positions[idx]
                x, y = np.clip(pos.astype(int), 0, self.size - 1)
                sensor_values[i, j] = self.trail_map[y, x]

                if not (self.agent_attached[i] != -1):
                    nutrient_dist = distances[idx]
                    sensor_values[i, j] += max(50 - nutrient_dist, 0)

        # 代理移动逻辑
        for i in range(len(self.agents)):
            pos = self.agents[i]
            angle = self.agent_angles[i]
            nutrient_id = self.agent_attached[i]
            attached = nutrient_id != -1

            # 方向调整
            if sensor_values[i, 1] < sensor_values[i, 0] and \
                    sensor_values[i, 1] < sensor_values[i, 2]:
                rotate_dir = 1 if sensor_values[i, 0] > sensor_values[i, 2] else -1
                max_rotate = move_params['max_rotate'] if has_active_nutrients else np.pi / 16
                angle += rotate_dir * max_rotate
            else:
                angle += np.random.uniform(-1, 1) * move_params['max_rotate']

            # 移动步长
            if attached:
                step = self.params['movement']['attached_step']
            else:
                step = move_params['free_step'] * self.agent_step_multiplier[i]
            new_pos = pos + step * np.array([np.cos(angle), np.sin(angle)])
            new_pos = np.clip(new_pos, 0, self.size - 1)

            # 营养物接触检测
            if not attached and len(self.nutrients) > 0:
                dist, idx = self.nutrient_tree.query(new_pos)
                if dist < self.params['nutrient']['activation_range']:
                    contact_id = self.nutrients[idx]['id']
                    self.agent_attached[i] = contact_id
                    if not self.nutrients[idx]['active']:
                        self.nutrients[idx]['active'] = True
                        self.nutrients[idx]['lifetime'] = 0

            # 信息素沉积
            deposit = self.params['pheromone']['attached_deposit'] if attached \
                else self.params['pheromone']['free_deposit']
            x, y = np.clip(new_pos.astype(int), 0, self.size - 1)
            self.trail_map[y, x] += deposit

            # 繁殖逻辑
            if attached:
                branch_prob = self.params['reproduction']['attached_prob']
            else:
                branch_prob = repro_params['free_prob']

            if np.random.rand() < branch_prob:
                branch_angle = angle + np.random.uniform(
                    -self.params['reproduction']['max_branch_angle'],
                    self.params['reproduction']['max_branch_angle']
                )
                # 80%保持附着，20%快速扩散
                if attached and np.random.rand() < 0.8:
                    new_parent = nutrient_id
                    new_multiplier = 1
                else:
                    new_parent = -1
                    new_multiplier = 2 if has_active_nutrients else 1
                new_agents.append((new_pos.copy(), branch_angle, new_parent, new_multiplier))

            self.agents[i] = new_pos
            self.agent_angles[i] = angle % (2 * np.pi)

        # 添加新代理
        if new_agents:
            new_pos, new_angles, parent_ids, multipliers = zip(*new_agents)
            self.agents = np.vstack([self.agents, np.array(new_pos)])
            self.agent_angles = np.concatenate([self.agent_angles, new_angles])
            self.agent_attached = np.concatenate([self.agent_attached, parent_ids])
            self.agent_step_multiplier = np.concatenate([
                self.agent_step_multiplier,
                np.array(multipliers)
            ])

        # 营养消耗（基于周围代理数量）
        active_nutrients = self.nutrients[self.nutrients['active']]
        if len(active_nutrients) > 0 and len(self.agents) > 0:
            agent_tree = cKDTree(self.agents)
            for idx in np.where(self.nutrients['active'])[0]:
                pos = self.nutrients[idx]['position']
                count = agent_tree.query_ball_point(
                    pos,
                    self.params['nutrient']['consumption_radius'],
                    return_length=True
                )
                self.nutrients['energy'][idx] -= count
                if self.nutrients['energy'][idx] <= 0:
                    self.nutrients['active'][idx] = False

        # 生存率筛选
        survival_probs = np.where(
            self.agent_attached != -1,
            self.params['survival']['attached_survival'],
            survival_params['free_survival']
        )
        survive_mask = np.random.rand(len(self.agents)) < survival_probs
        self._filter_agents(survive_mask)

        # 代理合并处理
        self.process_agent_merging()

    def _filter_agents(self, mask):
        """筛选代理"""
        self.agents = self.agents[mask]
        self.agent_angles = self.agent_angles[mask]
        self.agent_attached = self.agent_attached[mask]
        self.agent_step_multiplier = self.agent_step_multiplier[mask]

    def update(self, frame):
        """更新动画帧"""
        self.trail_map *= self.params['pheromone']['decay']
        self.update_agents()
        self.update_nutrient_states()

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.axis('off')

        processed_map = gaussian_filter(
            np.log1p(self.trail_map),
            sigma=self.params['gaussian_blur']
        )

        self.ax.imshow(
            processed_map,
            cmap='inferno',
            alpha=0.98,
            vmin=0.1,
            vmax=5.0,
            interpolation='hermite',
            rasterized=True,
            origin='lower',
            extent=[0, self.size, 0, self.size]
        )

        if len(self.nutrients) > 0:
            active_nutrients = self.nutrients[self.nutrients['active']]
            fading_nutrients = self.nutrients[~self.nutrients['active']]

            if len(active_nutrients) > 0:
                self.ax.scatter(
                    active_nutrients['position'][:, 0],
                    active_nutrients['position'][:, 1],
                    c='#FF4500',
                    s=80 * np.sqrt(active_nutrients['energy'] / 100),
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
            f"Remaining Energy: {np.sum(self.nutrients['energy']):,}"
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
        """运行动画"""
        self.fig, self.ax = plt.subplots(
            figsize=(9, 9),
            facecolor='#0F0F0F',
            dpi=150
        )
        self.ax.set_facecolor('#0F0F0F')
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        plt.tight_layout()

        ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=300, interval=33,
            blit=True, cache_frame_data=False
        )
        plt.show()


if __name__ == "__main__":
    sim = EnhancedSlimeMold(500, 100, 1000)
    sim.run_animation()