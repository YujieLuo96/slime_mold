import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置matplotlib使用TkAgg作为后端
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 用于在Tkinter中嵌入matplotlib图形
from matplotlib.figure import Figure  # 用于创建图形
import tkinter as tk  # 用于创建GUI
from tkinter import ttk  # 提供更美观的UI组件
from tkinter import filedialog, messagebox  # 用于文件对话框和消息框
from scipy.spatial import cKDTree  # 用于高效的空间搜索
from scipy.ndimage import gaussian_filter  # 用于图像的高斯模糊处理
from PIL import Image  # 用于生成GIF
import tempfile  # 用于创建临时目录
import shutil  # 用于清理临时文件
import os  # 用于路径操作


class EnhancedSlimeMold:
    def __init__(self, size=500, num_nutrients=100, num_agents=1000):
        self.size = size  # 模拟区域的大小
        self.nutrient_counter = 0  # 用于生成唯一的营养物ID

        # 初始化营养物
        self.nutrients = np.zeros(num_nutrients, dtype=[
            ('position', 'f8', 2),  # 营养物的位置
            ('lifetime', 'i4'),  # 营养物的生命周期
            ('active', 'bool'),  # 营养物是否活跃
            ('id', 'i4'),  # 营养物的唯一ID
            ('energy', 'i4')  # 营养物的能量
        ])
        self.nutrients['position'] = np.random.rand(num_nutrients, 2) * size  # 随机初始化营养物位置
        self.nutrients['lifetime'] = 0  # 初始化生命周期为0
        self.nutrients['active'] = False  # 初始状态为不活跃
        self.nutrients['id'] = np.arange(num_nutrients)  # 分配唯一ID
        self.nutrients['energy'] = 400  # 初始化能量
        self.nutrient_counter = num_nutrients  # 更新营养物计数器

        # 初始化代理
        self.agents = np.random.rand(num_agents, 2) * size  # 随机初始化代理位置
        self.agent_angles = np.random.rand(num_agents) * 2 * np.pi  # 随机初始化代理角度
        self.agent_attached = np.full(num_agents, -1, dtype=int)  # 代理是否附着在营养物上，-1表示未附着
        self.agent_step_multiplier = np.ones(num_agents)  # 代理的步长乘数

        self.trail_map = np.zeros((size, size))  # 用于记录代理留下的信息素轨迹
        self.nutrient_tree = cKDTree(self.nutrients['position'])  # 创建营养物的空间搜索树

        # 更新参数配置
        self.params = {
            'nutrient': {
                'activation_range': 5.0,  # 增大激活范围以匹配视觉大小
                'max_lifetime': 150,      # 延长营养物生命周期
                'consumption_radius': 10  # 增大消耗半径
            },
            'sensor': {
                'angle': np.pi / 3,  # 加宽传感器角度增强环境感知
                'distance': 11  # 增加感知距离
            },
            'movement': {
                'attached_step': 1.2,  # 附着时小步移动增加沉积密度
                'free_step': 3.5,  # 自由时大步探索新区域
                'max_rotate': np.pi / 4,  # 减少最大转向角度
                'inactive_rotate': np.pi / 8
            },
            'pheromone': {
                'attached_deposit': 400,  # 大幅增加附着沉积量
                'free_deposit': 40,  # 适当减少自由沉积量
                'decay': 0.91  # 降低衰减率延长路径持续时间
            },
            'reproduction': {
                'attached_prob': 0.06,  # 提高附着繁殖概率
                'free_prob': 0.005,  # 降低自由繁殖概率
                'explore_prob': 0.35,
                'max_branch_angle': np.pi / 2.2  # 缩小分支角度形成更紧密结构
            },
            'survival': {
                'attached_survival': 0.995,  # 大幅提高附着生存率
                'free_survival': 0.95,
                'inactive_survival': 0.88
            },
            'gaussian_blur': 0.8,  # 减少模糊保持路径锐度
            'merge_distance': 2  # 缩小合并距离促进主干形成
        }

    def batch_sense(self, positions, angles):
        """批量计算传感器位置（增强边缘检测）"""
        offsets = np.array([
            -self.params['sensor']['angle'] * 1.2,  # 扩大两侧传感器偏移
            0,
            self.params['sensor']['angle'] * 1.2
        ])
        sensor_angles = angles[:, None] + offsets  # 计算传感器的角度
        sensor_pos = positions[:, None] + \
                     self.params['sensor']['distance'] * np.stack(
            [np.cos(sensor_angles), np.sin(sensor_angles)], axis=2)  # 计算传感器的位置
        return sensor_pos.reshape(-1, 2)  # 返回传感器位置的二维数组

    def update_nutrient_states(self):
        """更新营养物状态"""
        active_mask = self.nutrients['active']  # 获取活跃的营养物
        self.nutrients['lifetime'][active_mask] += 1  # 更新活跃营养物的生命周期

        # 判断营养物是否过期
        expire_mask = (self.nutrients['lifetime'] > self.params['nutrient']['max_lifetime']) | \
                      (self.nutrients['energy'] <= 0)  # 移除冗余的位置检查
        expired_nutrients = self.nutrients[expire_mask]  # 获取过期的营养物

        # 处理过期的营养物
        if len(expired_nutrients) > 0:
            expired_ids = expired_nutrients['id']
            attached_mask = np.isin(self.agent_attached, expired_ids)  # 找到附着在过期营养物上的代理
            self.agent_attached[attached_mask] = -1  # 解除代理的附着状态

        self.nutrients = self.nutrients[~expire_mask]  # 移除过期的营养物
        if len(self.nutrients) > 0:
            self.nutrient_tree = cKDTree(self.nutrients['position'])  # 更新营养物的空间搜索树

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
            if (sensor_values[i, 0] - sensor_values[i, 2]) > 0.1 * sensor_values[i, 1]:
                rotate_dir = 1
            elif (sensor_values[i, 2] - sensor_values[i, 0]) > 0.1 * sensor_values[i, 1]:
                rotate_dir = -1
            else:
                rotate_dir = 0

            max_rotate = move_params['max_rotate'] if has_active_nutrients else np.pi / 16
            angle += rotate_dir * max_rotate

            # 自适应步长调整
            self.agent_step_multiplier[i] = 0.5 + 1.5 * (sensor_values[i].max() / 100)

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
        """更新模拟状态，返回需要绘制的数据"""
        self.trail_map *= self.params['pheromone']['decay']
        self.update_agents()
        self.update_nutrient_states()

        # 返回绘图数据
        return {
            "trail_map": self.trail_map,
            "nutrients": self.nutrients,
            "agents": len(self.agents),
            "energy": np.sum(self.nutrients['energy'])
        }


class SlimeMoldUI:
    def __init__(self, master):
        self.master = master
        master.title("智能黏菌模拟器")
        master.configure(bg='#2E2E2E')

        # 初始化模拟参数
        self.sim_params = {
            'size': 500,
            'num_nutrients': 50,
            'num_agents': 1000,
            'sensor_distance': 11,
            'sensor_angle': np.pi / 3,
            'free_step': 4.8,
            'attached_step': 1.2,
            'pheromone_decay': 0.88,
            'blur_sigma': 0.9
        }

        # 创建模拟实例
        self.sim = EnhancedSlimeMold(
            size=self.sim_params['size'],
            num_nutrients=self.sim_params['num_nutrients'],
            num_agents=self.sim_params['num_agents']
        )

        # 设置UI布局
        self.setup_ui()
        self.is_running = False

        # 录制相关属性
        self.is_recording = False
        self.recording_frames = []
        self.temp_dir = tempfile.mkdtemp()
        self.frame_counter = 0

    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧画布区域
        self.setup_canvas(main_frame)

        # 右侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # 控制按钮
        self.setup_buttons(control_frame)

        # 参数调节滑动条
        self.setup_sliders(control_frame)

        # 状态显示
        self.setup_stats(control_frame)

    def setup_canvas(self, parent):
        """设置绘图画布"""
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(6, 6), facecolor='#1A1A1A')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.add_nutrient)

    def setup_buttons(self, parent):
        """设置控制按钮"""
        btn_style = ttk.Style()
        btn_style.configure('TButton', font=('Arial', 9), padding=5)

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(btn_frame, text="开始", command=self.toggle_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="重置", command=self.reset_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清空", command=self.clear_simulation).pack(side=tk.LEFT, padx=2)

        self.record_btn = ttk.Button(btn_frame, text="开始录制", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=2)

    def toggle_recording(self):
        """切换录制状态"""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_btn.config(text="停止录制")
            self.recording_frames = []
            self.frame_counter = 0
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            os.makedirs(self.temp_dir, exist_ok=True)
        else:
            self.record_btn.config(text="开始录制")
            self.save_gif()

    def capture_frame(self):
        """捕获当前帧并保存到临时目录"""
        if self.is_recording and self.frame_counter % 3 == 0:  # 每3帧捕获一次
            path = os.path.join(self.temp_dir, f"frame_{self.frame_counter:05d}.png")
            self.fig.savefig(path, dpi=80, facecolor=self.fig.get_facecolor(), bbox_inches='tight')
            self.recording_frames.append(path)
        self.frame_counter += 1

    def save_gif(self):
        """将捕获的帧转换为GIF动画"""
        if not self.recording_frames:
            return

        # 让用户选择保存路径
        filename = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF动画", "*.gif"), ("所有文件", "*.*")]
        )

        if filename:
            try:
                # 使用PIL创建GIF
                images = []
                for path in self.recording_frames:
                    img = Image.open(path)
                    images.append(img.copy())
                    img.close()

                # 保存为GIF（优化颜色表）
                images[0].save(
                    filename,
                    save_all=True,
                    append_images=images[1:],
                    optimize=True,
                    duration=100,  # 每帧100ms（约10fps）
                    loop=0,
                    disposal=2  # 每帧处理方式：恢复背景色
                )
                messagebox.showinfo("保存成功", f"GIF动画已保存至：\n{filename}")
            except Exception as e:
                messagebox.showerror("保存失败", f"生成GIF时出错：\n{str(e)}")

        # 清理临时文件
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def setup_sliders(self, parent):
        """创建参数调节滑动条"""
        slider_frame = ttk.LabelFrame(parent, text="参数调节", padding=10)
        slider_frame.pack(fill=tk.X, pady=5)

        params = [
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

            label = ttk.Label(frame, text=param[1], width=12)
            label.pack(side=tk.LEFT)

            slider = ttk.Scale(frame, from_=param[2], to=param[3],
                               command=lambda v, p=param[0]: self.update_param(p, float(v)))
            slider.set(self.sim_params.get(param[0], 1))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.sliders[param[0]] = slider

    def setup_stats(self, parent):
        """设置统计信息显示"""
        stats_frame = ttk.LabelFrame(parent, text="实时统计", padding=10)
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_labels = {
            'agents': ttk.Label(stats_frame, text="代理数量: 0"),
            'nutrients': ttk.Label(stats_frame, text="活跃营养物: 0"),
            'energy': ttk.Label(stats_frame, text="剩余能量: 0")
        }

        for label in self.stats_labels.values():
            label.pack(anchor=tk.W)

    def update_param(self, param, value):
        """更新参数值"""
        self.sim_params[param] = value
        if param == 'sensor_distance':
            self.sim.params['sensor']['distance'] = value
        elif param == 'sensor_angle':
            self.sim.params['sensor']['angle'] = value
        elif param == 'free_step':
            self.sim.params['movement']['free_step'] = value
        elif param == 'attached_step':
            self.sim.params['movement']['attached_step'] = value
        elif param == 'pheromone_decay':
            self.sim.params['pheromone']['decay'] = value
        elif param == 'blur_sigma':
            self.sim.params['gaussian_blur'] = value

    def toggle_simulation(self):
        """切换模拟状态"""
        self.is_running = not self.is_running
        self.start_btn.config(text="暂停" if self.is_running else "开始")
        if self.is_running:
            self.animate()

    def animate(self):
        """更新动画帧"""
        if self.is_running:
            data = self.sim.update(0)  # 获取模拟数据

            # 清除旧绘图
            self.ax.clear()
            self.ax.set_xlim(0, self.sim.size)
            self.ax.set_ylim(0, self.sim.size)
            self.ax.axis('off')

            # 绘制轨迹图
            processed_map = gaussian_filter(
                np.log1p(data['trail_map'] ** 1.5),  # 增加非线性增强
                sigma=self.sim.params['gaussian_blur']
            )
            self.ax.imshow(
                processed_map,
                cmap='magma',  # 改用更高对比度的颜色映射
                alpha=0.92,
                vmin=0.3,  # 提高显示阈值
                vmax=7.0,  # 扩展最大值范围
                interpolation='bicubic',  # 改用更高阶插值
                origin='lower',
                extent=[0, self.sim.size, 0, self.sim.size]
            )

            # 绘制营养物
            if len(data['nutrients']) > 0:
                active_nutrients = data['nutrients'][data['nutrients']['active']]
                fading_nutrients = data['nutrients'][~data['nutrients']['active']]

                if len(active_nutrients) > 0:
                    # 根据激活范围计算绘制大小
                    activation_radius = self.sim.params['nutrient']['activation_range']
                    base_size = ( activation_radius) ** 2  # 基础面积与激活范围相关
                    self.ax.scatter(
                        active_nutrients['position'][:, 0],
                        active_nutrients['position'][:, 1],
                        c='#FF4500',
                        s=base_size * np.sqrt(active_nutrients['energy'] / 10),  # 调整能量影响系数
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

            # 更新统计信息
            stats_text = (
                f"Agents: {data['agents']:,}\n"
                f"Active Nutrients: {np.sum(data['nutrients']['active'])}\n"
                f"Remaining Energy: {data['energy']:,}"
            )
            self.ax.text(
                0.02, 0.95, stats_text,
                transform=self.ax.transAxes,
                color='white',
                fontsize=8,
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
            )

            self.canvas.draw()
            self.master.after(33, self.animate)
            self.capture_frame()  # 添加这行

    def update_stats(self):
        """更新统计信息"""
        self.stats_labels['agents'].config(
            text=f"代理数量: {len(self.sim.agents):,}")
        self.stats_labels['nutrients'].config(
            text=f"活跃营养物: {np.sum(self.sim.nutrients['active'])}")
        self.stats_labels['energy'].config(
            text=f"剩余能量: {np.sum(self.sim.nutrients['energy']):,}")

    def add_nutrient(self, event):
        """通过鼠标点击添加营养物"""
        if event.inaxes is None:
            return

        new_nutrient = np.zeros(1, dtype=self.sim.nutrients.dtype)
        new_nutrient['position'] = [[event.xdata, event.ydata]]
        new_nutrient['active'] = True
        new_nutrient['id'] = self.sim.nutrient_counter
        new_nutrient['energy'] = 400

        self.sim.nutrients = np.concatenate([self.sim.nutrients, new_nutrient])
        self.sim.nutrient_counter += 1
        self.sim.nutrient_tree = cKDTree(self.sim.nutrients['position'])

    def reset_simulation(self):
        """重置整个模拟"""
        self.sim = EnhancedSlimeMold(
            size=self.sim_params['size'],
            num_nutrients=self.sim_params['num_nutrients'],
            num_agents=self.sim_params['num_agents']
        )
        self.ax.clear()
        self.ax.axis('off')
        self.canvas.draw()

    def clear_simulation(self):
        """清空当前模拟"""
        self.sim.trail_map.fill(0)
        self.sim.nutrients = self.sim.nutrients[:0]  # 清空营养物
        self.sim.agents = np.empty((0, 2))
        self.ax.clear()
        self.ax.axis('off')
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = SlimeMoldUI(root)
    root.mainloop()