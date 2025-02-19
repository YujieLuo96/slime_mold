"""
多物种黏菌模拟器

本程序模拟多个物种的黏菌行为，包含以下主要功能：
1. 多物种差异化参数配置（移动速度、信息素沉积、颜色等）
2. 基于传感器探测的自主移动行为
3. 营养物激活与消耗机制
4. 代理繁殖与合并机制
5. 物种竞争与密度优势淘汰
6. 实时可视化界面与参数调节

主要类说明：
- EnhancedSlimeMold: 模拟核心逻辑，处理代理行为和环境更新
- SlimeMoldUI: 图形用户界面，提供可视化与交互控制
"""

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

BACKGROUND_COLOR = '#2E2E2E'
BASE_COLOR = '#1A1A1A' # 屏幕外空白处颜色

class EnhancedSlimeMold:
    """黏菌模拟核心类，处理多物种代理的行为逻辑和环境状态"""

    def __init__(self, size=500, num_nutrients=100, num_agents=1000, num_species=3):
        """
        初始化模拟环境
        参数：
            size: 模拟环境尺寸（正方形边长）
            num_nutrients: 初始营养物数量
            num_agents: 初始代理数量
            num_species: 物种数量
        """
        self.size = size
        self.num_species = num_species  # 物种数量
        self.nutrient_counter = 0

        # 初始化营养物（结构化数组）
        self.nutrients = np.zeros(num_nutrients, dtype=[
            ('position', 'f8', 2),  # 位置坐标
            ('lifetime', 'i4'),  # 存在时间
            ('active', 'bool'),  # 激活状态
            ('id', 'i4'),  # 唯一标识
            ('energy', 'i4')  # 剩余能量
        ])
        self.nutrients['position'] = np.random.rand(num_nutrients, 2) * size
        self.nutrients['lifetime'] = 0
        self.nutrients['active'] = False
        self.nutrients['id'] = np.arange(num_nutrients)
        self.nutrients['energy'] = 400
        self.nutrient_counter = num_nutrients

        # 初始化代理属性
        self.agents = np.random.rand(num_agents, 2) * size  # 代理位置
        self.agent_angles = np.random.rand(num_agents) * 2 * np.pi  # 移动方向
        self.agent_attached = np.full(num_agents, -1, dtype=int)  # 附着营养物ID
        self.agent_step_multiplier = np.ones(num_agents)  # 移动步长乘数
        self.agent_species = np.random.randint(0, self.num_species, num_agents)  # 代理所属物种

        # 初始化轨迹图（每个物种一个通道）
        self.trail_map = np.zeros((size, size, self.num_species))
        self.nutrient_tree = cKDTree(self.nutrients['position'])  # 营养物空间索引

        # 参数配置（物种参数+全局参数）
        self.params = {
            'species_params': self._generate_species_params(num_species),  # 物种差异化参数
            'global_params': {  # 全局共享参数
                'nutrient': {
                    'activation_range': 5.0,  # 营养物激活半径
                    'max_lifetime': 150,  # 最大存活时间
                    'consumption_radius': 10  # 营养消耗半径
                },
                'sensor': {
                    'angle': np.pi / 3,  # 传感器夹角
                    'distance': 11  # 传感器探测距离
                },
                'movement': {
                    'max_rotate': np.pi / 4,  # 最大旋转角度（激活状态）
                    'inactive_rotate': np.pi / 8  # 非激活状态旋转角度
                },
                'reproduction': {
                    'attached_prob': 0.06,  # 附着状态繁殖概率
                    'free_prob': 0.005,  # 自由状态繁殖概率
                    'explore_prob': 0.35,  # 探索分支概率
                    'max_branch_angle': np.pi / 2.2  # 最大分支角度
                },
                'survival': {
                    'attached_survival': 0.995,  # 附着状态生存率
                    'free_survival': 0.95,  # 自由状态生存率
                    'inactive_survival': 0.88  # 非激活状态生存率
                },
                'pheromone_decay': 0.91,  # 信息素衰减率
                'gaussian_blur': 0.8,  # 轨迹模糊强度
                'merge_distance': 2,  # 代理合并距离
                'dominance_radius': 8,  # 物种竞争检测半径
                'dominance_threshold': 0.6  # 数量优势阈值
            }
        }

    def _generate_species_params(self, num_species):
        """生成物种差异化参数"""
        species_params = []
        colors = [  # 预定义物种颜色
            (1.0, 0.2, 0.2),  # 红色
            (0.2, 1.0, 0.2),  # 绿色
            (0.3, 0.3, 1.0),  # 蓝色
            (1.0, 0.5, 0.0),  # 橙色
            (0.5, 0.0, 1.0),  # 紫色
            (0.0, 1.0, 1.0)  # 青色
        ]
        for i in range(num_species):
            species_params.append({
                'pheromone': {
                    'attached_deposit':  800,  # 400 + i * 50,  # 附着状态信息素沉积量
                    'free_deposit':  80  # 50 + i * 5  # 自由状态沉积量
                },
                'movement': {
                    'free_step':  4.4,  # 3.5 + i * 0.5,  # 自由移动步长
                    'attached_step': 1.4  # 1.2 + i * 0.1  # 附着移动步长
                },
                'color': colors[i % len(colors)],  # 显示颜色
                'inhibition': 0.5 + i * 0.1  # 其他物种信息素抑制系数
            })
        return species_params

    def batch_sense(self, positions, angles):
        """批量计算代理传感器位置"""
        offsets = np.array([
            -self.params['global_params']['sensor']['angle'] * 1.2,  # 左传感器偏移
            0,  # 正前方传感器
            self.params['global_params']['sensor']['angle'] * 1.2  # 右传感器偏移
        ])
        sensor_angles = angles[:, None] + offsets  # 为每个代理计算三个传感器角度
        sensor_pos = positions[:, None] + \
                     self.params['global_params']['sensor']['distance'] * np.stack(
            [np.cos(sensor_angles), np.sin(sensor_angles)], axis=2)
        return sensor_pos.reshape(-1, 2)

    def update_nutrient_states(self):
        """更新营养物状态"""
        active_mask = self.nutrients['active']
        self.nutrients['lifetime'][active_mask] += 1

        # 检测过期营养物（超过寿命或能量耗尽）
        expire_mask = (self.nutrients['lifetime'] > self.params['global_params']['nutrient']['max_lifetime']) | \
                      (self.nutrients['energy'] <= 0)
        expired_nutrients = self.nutrients[expire_mask]

        # 解除相关代理的附着状态
        if len(expired_nutrients) > 0:
            expired_ids = expired_nutrients['id']
            attached_mask = np.isin(self.agent_attached, expired_ids)
            self.agent_attached[attached_mask] = -1

        # 移除过期营养物并更新空间索引
        self.nutrients = self.nutrients[~expire_mask]
        if len(self.nutrients) > 0:
            self.nutrient_tree = cKDTree(self.nutrients['position'])

    def process_agent_merging(self):
        """处理同物种代理合并"""
        if len(self.agents) < 2:
            return

        # 建立空间索引查找邻近代理
        agent_tree = cKDTree(self.agents)
        pairs = agent_tree.query_pairs(self.params['global_params']['merge_distance'])

        for i, j in pairs:
            # 仅合并同物种代理
            if self.agent_species[i] == self.agent_species[j]:
                ai, aj = self.agent_attached[i], self.agent_attached[j]
                if ai != aj:
                    # 优先合并到已附着的代理
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
        """更新所有代理状态（含物种竞争逻辑）"""
        new_agents = []
        has_active_nutrients = np.any(self.nutrients['active'])
        global_params = self.params['global_params']

        # 传感器处理（多物种信息素综合计算）
        sensor_positions = self.batch_sense(self.agents, self.agent_angles)
        if len(self.nutrients) > 0:
            distances, indices = self.nutrient_tree.query(sensor_positions, distance_upper_bound=self.size)
        else:
            distances, indices = np.full(len(sensor_positions), np.inf), []

        # 计算各传感器的综合值（考虑自身物种和其他物种的信息素）
        sensor_values = np.zeros((len(self.agents), 3))
        for i in range(len(self.agents)):
            species = self.agent_species[i]
            species_params = self.params['species_params'][species]
            for j in range(3):
                idx = i * 3 + j
                if idx >= len(indices):
                    continue

                pos = sensor_positions[idx]
                x, y = np.clip(pos.astype(int), 0, self.size - 1)

                # 计算传感器值：自身信息素 - 其他物种信息素*抑制系数
                own_pheromone = self.trail_map[y, x, species]
                other_pheromone = np.sum(self.trail_map[y, x, :]) - own_pheromone
                sensor_value = own_pheromone - other_pheromone * species_params['inhibition']

                # 自由状态时考虑营养物距离
                if self.agent_attached[i] == -1:
                    nutrient_dist = distances[idx]
                    sensor_value += max(50 - nutrient_dist, 0)

                sensor_values[i, j] = sensor_value

        # 代理移动与行为逻辑
        for i in range(len(self.agents)):
            species = self.agent_species[i]
            species_params = self.params['species_params'][species]
            pos = self.agents[i]
            angle = self.agent_angles[i]
            nutrient_id = self.agent_attached[i]
            attached = nutrient_id != -1

            # 根据传感器值调整方向
            if (sensor_values[i, 0] - sensor_values[i, 2]) > 0.1 * sensor_values[i, 1]:
                rotate_dir = 1  # 左转
            elif (sensor_values[i, 2] - sensor_values[i, 0]) > 0.1 * sensor_values[i, 1]:
                rotate_dir = -1  # 右转
            else:
                rotate_dir = 0  # 直行

            # 根据环境状态调整最大旋转角度
            max_rotate = global_params['movement']['max_rotate'] if has_active_nutrients else global_params['movement'][
                'inactive_rotate']
            angle += rotate_dir * max_rotate

            # 计算移动步长（使用物种特定参数）
            self.agent_step_multiplier[i] = 0.5 + 1.5 * (sensor_values[i].max() / 100)
            if attached:
                step = species_params['movement']['attached_step']
            else:
                step = species_params['movement']['free_step'] * self.agent_step_multiplier[i]

            # 更新位置并限制在边界内
            new_pos = pos + step * np.array([np.cos(angle), np.sin(angle)])
            new_pos = np.clip(new_pos, 0, self.size - 1)

            # 营养物接触检测与附着
            if not attached and len(self.nutrients) > 0:
                dist, idx = self.nutrient_tree.query(new_pos)
                if dist < global_params['nutrient']['activation_range']:
                    contact_id = self.nutrients[idx]['id']
                    self.agent_attached[i] = contact_id
                    if not self.nutrients[idx]['active']:
                        self.nutrients[idx]['active'] = True
                        self.nutrients[idx]['lifetime'] = 0

            # 信息素沉积（到对应物种的通道）
            x, y = np.clip(new_pos.astype(int), 0, self.size - 1)
            deposit = species_params['pheromone']['attached_deposit'] if attached else species_params['pheromone'][
                'free_deposit']
            self.trail_map[y, x, species] += deposit

            # 繁殖逻辑（继承父代物种）
            branch_prob = global_params['reproduction']['attached_prob'] if attached else global_params['reproduction'][
                'free_prob']
            if np.random.rand() < branch_prob:
                branch_angle = angle + np.random.uniform(
                    -global_params['reproduction']['max_branch_angle'],
                    global_params['reproduction']['max_branch_angle']
                )
                new_parent = nutrient_id if attached and np.random.rand() < 0.8 else -1
                new_multiplier = 1 if attached else 2
                new_agents.append((new_pos.copy(), branch_angle, new_parent, new_multiplier, species))

            # 更新代理属性
            self.agents[i] = new_pos
            self.agent_angles[i] = angle % (2 * np.pi)

        # 添加新生代理
        if new_agents:
            new_pos, new_angles, parent_ids, multipliers, species = zip(*new_agents)
            self.agents = np.vstack([self.agents, np.array(new_pos)])
            self.agent_angles = np.concatenate([self.agent_angles, new_angles])
            self.agent_attached = np.concatenate([self.agent_attached, parent_ids])
            self.agent_step_multiplier = np.concatenate([self.agent_step_multiplier, np.array(multipliers)])
            self.agent_species = np.concatenate([self.agent_species, species])

        # 营养物能量消耗
        active_nutrients = self.nutrients[self.nutrients['active']]
        if len(active_nutrients) > 0 and len(self.agents) > 0:
            agent_tree = cKDTree(self.agents)
            for idx in np.where(self.nutrients['active'])[0]:
                pos = self.nutrients[idx]['position']
                count = agent_tree.query_ball_point(
                    pos,
                    global_params['nutrient']['consumption_radius'],
                    return_length=True
                )
                self.nutrients['energy'][idx] -= count
                if self.nutrients['energy'][idx] <= 0:
                    self.nutrients['active'][idx] = False

        # 物种竞争淘汰机制
        if len(self.agents) > 0:
            agent_tree = cKDTree(self.agents)
            survive_mask = np.ones(len(self.agents), dtype=bool)

            for i in range(len(self.agents)):
                pos = self.agents[i]
                current_species = self.agent_species[i]

                # 在竞争半径内统计物种分布
                neighbors = agent_tree.query_ball_point(pos, global_params['dominance_radius'])
                neighbor_species = self.agent_species[neighbors]

                current_count = np.sum(neighbor_species == current_species)
                total_count = len(neighbors)

                if total_count > 0 and (current_count / total_count) < global_params['dominance_threshold']:
                    survive_mask[i] = False  # 淘汰处于劣势的代理

            self._filter_agents(survive_mask)

        # 常规生存率筛选
        survival_probs = np.where(
            self.agent_attached != -1,
            global_params['survival']['attached_survival'],
            global_params['survival']['free_survival'] if has_active_nutrients else global_params['survival'][
                'inactive_survival']
        )
        survive_mask = np.random.rand(len(self.agents)) < survival_probs
        self._filter_agents(survive_mask)

        self.process_agent_merging()

    def _filter_agents(self, mask):
        """筛选代理并更新相关属性"""
        self.agents = self.agents[mask]
        self.agent_angles = self.agent_angles[mask]
        self.agent_attached = self.agent_attached[mask]
        self.agent_step_multiplier = self.agent_step_multiplier[mask]
        self.agent_species = self.agent_species[mask]

    def update(self, frame):
        """执行单步模拟更新"""
        # 各物种信息素独立衰减
        for s in range(self.num_species):
            self.trail_map[:, :, s] *= self.params['global_params']['pheromone_decay']

        self.update_agents()
        self.update_nutrient_states()

        return {
            "trail_map": self.trail_map,
            "nutrients": self.nutrients,
            "agents": len(self.agents),
            "energy": np.sum(self.nutrients['energy'])
        }


class SlimeMoldUI:
    """黏菌模拟器的图形用户界面类，负责可视化与用户交互"""

    def __init__(self, master):
        """
        初始化UI界面
        参数：
            master: Tkinter根窗口
        """
        self.master = master
        master.title("多物种黏菌模拟器")
        master.configure(bg = BACKGROUND_COLOR)  # 设置背景颜色

        # 初始化模拟参数
        self.sim_params = {
            'size': 500,
            'num_nutrients': 200,
            'num_agents': 1000,
            'num_species': 3,  # 默认物种数量
            'sensor_distance': 11,
            'sensor_angle': np.pi / 3,
            'free_step': 4.8,
            'attached_step': 1.2,
            'pheromone_decay': 0.88,
            'blur_sigma': 0.9
        }

        # 初始化模拟器
        self.sim = EnhancedSlimeMold(
            size=self.sim_params['size'],
            num_nutrients=self.sim_params['num_nutrients'],
            num_agents=self.sim_params['num_agents'],
            num_species=self.sim_params['num_species']
        )

        self.setup_ui()  # 设置UI布局
        self.is_running = False  # 模拟运行状态

    def setup_ui(self):
        """设置UI布局"""
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.setup_canvas(main_frame)  # 设置画布
        self.setup_control_panel(main_frame)  # 设置控制面板

    def setup_canvas(self, parent):
        """设置可视化画布"""
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建Matplotlib图形
        self.fig = Figure(figsize=(6, 6), facecolor= BASE_COLOR)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        # 将Matplotlib图形嵌入Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.add_nutrient)  # 绑定点击事件

    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        self.setup_buttons(control_frame)  # 设置按钮
        self.setup_sliders(control_frame)  # 设置滑动条
        self.setup_stats(control_frame)  # 设置统计信息

    def setup_buttons(self, parent):
        """设置控制按钮"""
        btn_style = ttk.Style()
        btn_style.configure('TButton', font=('Arial', 9), padding=5)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)

        # 开始/暂停按钮
        self.start_btn = ttk.Button(btn_frame, text="开始", command=self.toggle_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        # 重置按钮
        ttk.Button(btn_frame, text="重置", command=self.reset_simulation).pack(side=tk.LEFT, padx=2)

        # 清空按钮
        ttk.Button(btn_frame, text="清空", command=self.clear_simulation).pack(side=tk.LEFT, padx=2)

    def setup_sliders(self, parent):
        """设置参数滑动条"""
        slider_frame = ttk.LabelFrame(parent, text="参数调节", padding=10)
        slider_frame.pack(fill=tk.X, pady=5)

        # 可调节参数列表
        params = [
            ('num_species', '物种数量', 1, 6),  # 物种数量
            ('sensor_distance', '传感器距离', 1, 20),
            ('sensor_angle', '传感器角度', 0, np.pi / 2),
            ('free_step', '自由步长', 0.1, 5),
            ('attached_step', '附着步长', 0.1, 2),
            ('pheromone_decay', '信息素衰减', 0.8, 0.99),
            ('blur_sigma', '模糊强度', 0.5, 3)
        ]

        self.sliders = {}
        for param in params:
            frame = ttk.Frame(slider_frame)
            frame.pack(fill=tk.X, pady=2)

            # 参数标签
            label = ttk.Label(frame, text=param[1], width=12)
            label.pack(side=tk.LEFT)

            # 滑动条
            slider = ttk.Scale(frame, from_=param[2], to=param[3],
                               command=lambda v, p=param[0]: self.update_param(p, float(v)))
            slider.set(self.sim_params.get(param[0], 1))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.sliders[param[0]] = slider

    def setup_stats(self, parent):
        """设置实时统计信息"""
        stats_frame = ttk.LabelFrame(parent, text="实时统计", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)

        # 统计信息标签
        self.stats_labels = {
            'agents': ttk.Label(stats_frame, text="代理数量: 0"),
            'nutrients': ttk.Label(stats_frame, text="活跃营养物: 0"),
            'energy': ttk.Label(stats_frame, text="剩余能量: 0")
        }

        for label in self.stats_labels.values():
            label.pack(anchor=tk.W)

    def update_param(self, param, value):
        """更新模拟参数"""
        self.sim_params[param] = value
        if param == 'num_species':
            # 重置模拟以应用新的物种数量
            self.reset_simulation()
        elif param == 'sensor_distance':
            self.sim.params['global_params']['sensor']['distance'] = value
        elif param == 'sensor_angle':
            self.sim.params['global_params']['sensor']['angle'] = value
        elif param == 'free_step':
            for species_params in self.sim.params['species_params']:
                species_params['movement']['free_step'] = value
        elif param == 'attached_step':
            for species_params in self.sim.params['species_params']:
                species_params['movement']['attached_step'] = value
        elif param == 'pheromone_decay':
            self.sim.params['global_params']['pheromone_decay'] = value
        elif param == 'blur_sigma':
            self.sim.params['global_params']['gaussian_blur'] = value

    def toggle_simulation(self):
        """切换模拟运行状态"""
        self.is_running = not self.is_running
        self.start_btn.config(text="暂停" if self.is_running else "开始")
        if self.is_running:
            self.animate()

    def animate(self):
        """执行模拟动画并进行可视化渲染"""
        if self.is_running:
            # 执行模拟计算并获取当前状态数据
            data = self.sim.update(0)  # 注意：这里传入的frame参数未被使用

            # 清空画布并设置坐标系
            self.ax.clear()
            self.ax.set_xlim(0, self.sim.size)
            self.ax.set_ylim(0, self.sim.size)
            self.ax.axis('off')

            """
            轨迹图渲染说明：
            - 每个物种独立渲染一个颜色通道，最终叠加显示
            - 修改以下参数会影响轨迹显示效果：
              1. species_params['color'] : 改变各物种的显示颜色
              2. species_params['pheromone']中的沉积量参数 : 
                 * attached_deposit - 附着状态时沉积量，值越大轨迹越亮
                 * free_deposit - 自由移动时沉积量，值越大基础轨迹越明显
              3. global_params['gaussian_blur'] : 
                 * 控制高斯模糊的sigma值（通过界面"模糊强度"滑块调节）
                 * 值越大轨迹边缘越模糊，范围建议0.5-3.0
              4. global_params['pheromone_decay'] : 
                 * 信息素衰减率（通过"信息素衰减"滑块调节）
                 * 值越小衰减越快，轨迹存留时间越短
            """
            for s in range(self.sim.num_species):
                species_params = self.sim.params['species_params'][s]
                # 对轨迹图进行预处理（对数变换+高斯模糊）
                processed_map = gaussian_filter(
                    np.log1p(data['trail_map'][:, :, s] ** 1.2),  # 非线性增强
                    sigma=self.sim.params['global_params']['gaussian_blur']  # 模糊强度参数
                )
                # 创建颜色映射（从黑色到物种颜色）
                self.ax.imshow(
                    processed_map,
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                        'custom', ['#000000', species_params['color']]),
                    alpha=0.8,  # 整体透明度，值越小越透明
                    vmin=0.3,  # 颜色映射下限，低于此值不显示
                    vmax=7.0,  # 颜色映射上限，高于此值饱和
                    interpolation='bicubic',  # 插值方式影响平滑度
                    origin='lower',  # 坐标系原点在左下角
                    extent=[0, self.sim.size, 0, self.sim.size]
                )

            """
            营养物渲染说明：
            - 激活状态显示为橙色星形，未激活显示为绿色圆形
            - 修改以下参数会影响营养物显示：
              1. global_params['nutrient']['activation_range'] : 
                 * 控制激活半径（通过传感器距离参数间接影响）
                 * 值越大营养物激活范围越大，显示尺寸也越大
              2. global_params['nutrient']['max_lifetime'] : 
                 * 最大存活时间，值越大营养物存在时间越长
              3. nutrient['energy'] : 
                 * 剩余能量，能量越高显示尺寸越大
            """
            if len(data['nutrients']) > 0:
                active_nutrients = data['nutrients'][data['nutrients']['active']]
                fading_nutrients = data['nutrients'][~data['nutrients']['active']]

                # 渲染激活状态的营养物（橙色星形）
                if len(active_nutrients) > 0:
                    activation_radius = self.sim.params['global_params']['nutrient']['activation_range']
                    base_size = (activation_radius) ** 2  # 基础尺寸与激活半径相关
                    self.ax.scatter(
                        active_nutrients['position'][:, 0],
                        active_nutrients['position'][:, 1],
                        c='#FF4500',  # 填充颜色
                        s=base_size * np.sqrt(active_nutrients['energy'] / 10),  # 尺寸与能量平方根成正比
                        edgecolors='#FFD70088',  # 边缘颜色（金色带透明度）
                        linewidths=2,  # 边缘线宽
                        alpha=0.9,  # 整体透明度
                        zorder=3,  # 绘制层级（在上层）
                        marker='*'  # 星形标记
                    )

                # 渲染未激活/衰减中的营养物（绿色圆形）
                if len(fading_nutrients) > 0:
                    self.ax.scatter(
                        fading_nutrients['position'][:, 0],
                        fading_nutrients['position'][:, 1],
                        c='#00FF7F',  # 填充颜色
                        s=30,  # 固定尺寸
                        alpha=0.6,  # 较低透明度
                        zorder=2,  # 绘制层级（在下层）
                        marker='o'  # 圆形标记
                    )

            """
            统计信息显示说明：
            - 显示代理数量、活跃营养物数量、剩余总能量
            - 以下参数会影响统计值：
              1. reproduction参数中的概率设置 : 影响代理数量增长速率
              2. survival参数中的生存率 : 影响代理数量衰减速率
              3. nutrient['energy']消耗速率 : 影响剩余能量变化速度
            """
            stats_text = (
                f"Agents: {data['agents']:,}\n"
                f"Active Nutrients: {np.sum(data['nutrients']['active'])}\n"
                f"Remaining Energy: {data['energy']:,}"
            )
            self.ax.text(
                0.02, 0.95, stats_text,
                transform=self.ax.transAxes,  # 使用相对坐标系
                color='white',
                fontsize=8,
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')  # 半透明黑色背景
            )

            # 刷新画布并准备下一帧
            self.canvas.draw()
            self.master.after(33, self.animate)  # 约30FPS（1000/33≈30帧/秒）

    def update_stats(self):
        """更新统计信息"""
        self.stats_labels['agents'].config(
            text=f"代理数量: {len(self.sim.agents):,}")
        self.stats_labels['nutrients'].config(
            text=f"活跃营养物: {np.sum(self.sim.nutrients['active'])}")
        self.stats_labels['energy'].config(
            text=f"剩余能量: {np.sum(self.sim.nutrients['energy']):,}")

    def add_nutrient(self, event):
        """添加营养物（点击事件）"""
        if event.inaxes is None:
            return

        # 创建新营养物
        new_nutrient = np.zeros(1, dtype=self.sim.nutrients.dtype)
        new_nutrient['position'] = [[event.xdata, event.ydata]]
        new_nutrient['active'] = True
        new_nutrient['id'] = self.sim.nutrient_counter
        new_nutrient['energy'] = 400

        # 更新营养物列表
        self.sim.nutrients = np.concatenate([self.sim.nutrients, new_nutrient])
        self.sim.nutrient_counter += 1
        self.sim.nutrient_tree = cKDTree(self.sim.nutrients['position'])

    def reset_simulation(self):
        """重置模拟"""
        self.sim = EnhancedSlimeMold(
            size=self.sim_params['size'],
            num_nutrients=self.sim_params['num_nutrients'],
            num_agents=self.sim_params['num_agents'],
            num_species=int(self.sim_params['num_species'])
        )
        self.ax.clear()
        self.ax.axis('off')
        self.canvas.draw()

    def clear_simulation(self):
        """清空模拟"""
        self.sim.trail_map.fill(0)
        self.sim.nutrients = self.sim.nutrients[:0]
        self.sim.agents = np.empty((0, 2))
        self.ax.clear()
        self.ax.axis('off')
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = SlimeMoldUI(root)
    root.mainloop()