import pygame
import numpy as np
import torch
from mouse_trajectory_model import TrajectoryGenerator, generate_trajectory
import json
import random
import sys
import os

class TrajectoryVisualizer:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("轨迹对比测试 (按空格键生成新轨迹)")
        
        self.clock = pygame.time.Clock()
        
        # 设置中文字体
        if sys.platform.startswith('win'):
            # Windows系统字体路径
            font_paths = [
                'C:\\Windows\\Fonts\\simhei.ttf',  # 黑体
                'C:\\Windows\\Fonts\\msyh.ttc',    # 微软雅黑
                'C:\\Windows\\Fonts\\simsun.ttc',  # 宋体
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        self.font = pygame.font.Font(font_path, 36)
                        print(f"成功加载字体: {font_path}")
                        break
                    except:
                        continue
            else:  # 如果所有字体都失败了
                print("尝试使用系统字体...")
                try:
                    self.font = pygame.font.SysFont('simsun', 36)  # 尝试使用系统宋体
                except:
                    print("无法加载中文字体，使用默认字体")
                    self.font = pygame.font.SysFont(None, 36)
        else:
            # Linux/Mac系统
            try:
                self.font = pygame.font.SysFont('notosanscjk', 36)  # 尝试使用Noto Sans CJK
            except:
                print("无法加载中文字体，使用默认字体")
                self.font = pygame.font.SysFont(None, 36)
        
        # 修改图例文本，使用英文作为备选
        self.ai_text = 'AI Trajectory (Blue)' if self.font == pygame.font.SysFont(None, 36) else 'AI轨迹 (蓝色)'
        self.real_text = 'Real Trajectory (Pink)' if self.font == pygame.font.SysFont(None, 36) else '真实轨迹 (粉色)'
        
        # 加载模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = TrajectoryGenerator().to(self.device)
        try:
            self.model.load_state_dict(torch.load('trajectory_model.pth', map_location=self.device))
            print(u"模型加载成功！")
        except FileNotFoundError:
            print(u"错误：找不到模型文件 trajectory_model.pth")
            raise
        self.model.eval()
        
        # 加载真实轨迹数据用于对比
        try:
            with open('mouse_trajectories_20250128_124044.json', 'r') as f:
                self.real_trajectories = json.load(f)['trajectories']
            print(f"加载了 {len(self.real_trajectories)} 条真实轨迹")
        except FileNotFoundError:
            print(u"错误：找不到轨迹数据文件")
            raise

    def draw_trajectory(self, points, color, width=2):
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, width)
            # 绘制起点和终点
            pygame.draw.circle(self.screen, (0, 255, 0), points[0], 5)  # 绿色起点
            pygame.draw.circle(self.screen, (255, 0, 0), points[-1], 5)  # 红色终点

    def generate_random_points(self):
        start_x = random.randint(50, self.width - 50)
        start_y = random.randint(50, self.height - 50)
        end_x = random.randint(50, self.width - 50)
        end_y = random.randint(50, self.height - 50)
        return np.array([start_x, start_y]), np.array([end_x, end_y])

    def get_similar_real_trajectory(self, start_point, end_point):
        target_vector = end_point - start_point
        target_distance = np.linalg.norm(target_vector)
        
        best_trajectory = None
        min_diff = float('inf')
        
        for traj in self.real_trajectories:
            points = np.array(traj['points'])
            if len(points) < 2:
                continue
                
            real_vector = points[-1] - points[0]
            real_distance = np.linalg.norm(real_vector)
            
            if real_distance < 1e-6:  # 避免除零错误
                continue
                
            # 计算方向和距离的差异
            cos_sim = np.dot(target_vector, real_vector) / (target_distance * real_distance)
            if cos_sim > 1:  # 处理数值误差
                cos_sim = 1
            elif cos_sim < -1:
                cos_sim = -1
                
            direction_diff = np.arccos(cos_sim) / np.pi  # 归一化到 [0,1]
            distance_diff = abs(target_distance - real_distance) / max(target_distance, real_distance)
            
            # 综合考虑方向和距离的差异
            total_diff = direction_diff * 0.7 + distance_diff * 0.3
            
            if total_diff < min_diff:
                min_diff = total_diff
                best_trajectory = points
        
        if best_trajectory is not None and min_diff < 0.5:  # 只有当差异不太大时才返回轨迹
            # 对轨迹进行缩放和平移，使其匹配目标起点和终点
            trajectory = best_trajectory.copy()
            
            # 计算缩放比例
            original_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            if original_distance > 1e-6:
                # 计算缩放和旋转
                scale = target_distance / original_distance
                
                # 先将轨迹移动到原点
                centered_trajectory = trajectory - trajectory[0]
                
                # 计算旋转角度
                original_direction = (trajectory[-1] - trajectory[0]) / original_distance
                target_direction = target_vector / target_distance
                
                # 计算旋转矩阵
                cos_theta = np.dot(original_direction, target_direction)
                sin_theta = np.cross(original_direction, target_direction)
                rotation_matrix = np.array([
                    [cos_theta, -sin_theta],
                    [sin_theta, cos_theta]
                ])
                
                # 应用变换：缩放、旋转、平移
                transformed_trajectory = np.dot(centered_trajectory * scale, rotation_matrix.T) + start_point
                
                # 确保起点和终点完全匹配
                transformed_trajectory[0] = start_point
                transformed_trajectory[-1] = end_point
                
                return transformed_trajectory.astype(int)
        
        return None

    def run(self):
        running = True
        show_trajectory = False
        ai_trajectory = None
        real_trajectory = None
        
        print(u"程序已启动，按空格键生成新的轨迹对比")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print(u"生成新轨迹...")
                        # 生成新的随机起点和终点
                        start_point, end_point = self.generate_random_points()
                        
                        # 生成AI轨迹
                        try:
                            ai_points = generate_trajectory(self.model, start_point, end_point, device=self.device)
                            ai_trajectory = ai_points.astype(int)
                            print(u"AI轨迹生成成功")
                            print(f"AI轨迹起点: {ai_trajectory[0]}")
                        except Exception as e:
                            print(f"生成AI轨迹时出错: {e}")
                            continue
                        
                        # 获取相似的真实轨迹
                        real_trajectory = self.get_similar_real_trajectory(start_point, end_point)
                        if real_trajectory is not None:
                            print(u"找到匹配的真实轨迹")
                            print(f"真实轨迹起点: {real_trajectory[0]}")
                            
                            # 验证起点是否完全一致
                            if not np.array_equal(ai_trajectory[0], real_trajectory[0]):
                                print("警告：轨迹起点不一致！")
                                print(f"AI起点: {ai_trajectory[0]}")
                                print(f"真实起点: {real_trajectory[0]}")
                        else:
                            print(u"未找到匹配的真实轨迹")
                        
                        show_trajectory = True

            # 绘制
            self.screen.fill((255, 255, 255))
            
            if show_trajectory:
                if ai_trajectory is not None:
                    self.draw_trajectory(ai_trajectory, (0, 0, 255))
                if real_trajectory is not None:
                    self.draw_trajectory(real_trajectory, (255, 0, 255))
                
                # 使用预设的文本
                ai_text = self.font.render(self.ai_text, True, (0, 0, 255))
                real_text = self.font.render(self.real_text, True, (255, 0, 255))
                self.screen.blit(ai_text, (10, 10))
                self.screen.blit(real_text, (10, 50))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    visualizer = TrajectoryVisualizer()
    visualizer.run() 