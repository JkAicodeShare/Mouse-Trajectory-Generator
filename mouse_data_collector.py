import pygame
import numpy as np
import time
import json
from datetime import datetime

class MouseTracker:
    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("鼠标轨迹收集器")
        
        self.target_radius = 20
        self.target_pos = self.generate_random_position()
        self.tracking = False
        self.trajectory = []
        self.all_trajectories = []
        self.clock = pygame.time.Clock()
        
        # 增加采样频率
        self.sample_rate = 144  # 提高采样率
        self.last_sample_time = 0

    def generate_random_position(self):
        x = np.random.randint(self.target_radius, self.width - self.target_radius)
        y = np.random.randint(self.target_radius, self.height - self.target_radius)
        return (x, y)

    def save_trajectories(self):
        if not self.all_trajectories:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mouse_trajectories_{timestamp}.json"
        
        data = {
            "trajectories": self.all_trajectories
        }
        
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"轨迹已保存到 {filename}")

    def update_trajectory(self):
        if self.tracking:
            current_time = time.time()
            if current_time - self.last_sample_time >= 1.0 / self.sample_rate:
                self.trajectory.append(list(pygame.mouse.get_pos()))
                self.last_sample_time = current_time

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    distance = np.sqrt((mouse_pos[0] - self.target_pos[0])**2 + 
                                    (mouse_pos[1] - self.target_pos[1])**2)
                    
                    if distance <= self.target_radius:
                        self.tracking = True
                        self.trajectory = [list(mouse_pos)]
                        self.last_sample_time = time.time()
                        self.target_pos = self.generate_random_position()
                        
                elif event.type == pygame.MOUSEBUTTONUP:
                    if self.tracking:
                        self.tracking = False
                        if len(self.trajectory) > 1:
                            self.all_trajectories.append({
                                "points": self.trajectory,
                                "timestamp": time.time()
                            })
            
            # 更新轨迹采样
            self.update_trajectory()

            # 绘制
            self.screen.fill((255, 255, 255))
            pygame.draw.circle(self.screen, (255, 0, 0), self.target_pos, self.target_radius)
            
            if len(self.trajectory) > 1:
                pygame.draw.lines(self.screen, (0, 0, 255), False, self.trajectory, 2)

            pygame.display.flip()
            self.clock.tick(144)  # 提高刷新率

        self.save_trajectories()
        pygame.quit()

if __name__ == "__main__":
    tracker = MouseTracker()
    tracker.run() 