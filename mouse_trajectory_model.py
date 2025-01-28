import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, max_length=50):
        self.samples = []
        self.max_length = max_length
        
        for traj in trajectories:
            points = np.array(traj['points'])
            if len(points) < 5:  # 忽略太短的轨迹
                continue
                
            # 计算相对位移
            relative_movements = points[1:] - points[:-1]
            
            # 计算起点到终点的向量
            start_to_end = points[-1] - points[0]
            distance = np.linalg.norm(start_to_end)
            
            if distance < 1e-6:  # 忽略原地不动的轨迹
                continue
            
            # 对轨迹进行重采样
            if len(relative_movements) > self.max_length:
                # 如果轨迹太长，进行降采样
                indices = np.linspace(0, len(relative_movements)-1, self.max_length, dtype=int)
                relative_movements = relative_movements[indices]
            else:
                # 如果轨迹太短，进行线性插值
                num_points = len(relative_movements)
                x = np.linspace(0, num_points-1, num_points)
                x_new = np.linspace(0, num_points-1, self.max_length)
                relative_movements = np.array([
                    np.interp(x_new, x, relative_movements[:, 0]),
                    np.interp(x_new, x, relative_movements[:, 1])
                ]).T
            
            # 标准化相对移动
            movement_distances = np.linalg.norm(relative_movements, axis=1)
            mask = movement_distances > 1e-6
            relative_movements[mask] = relative_movements[mask] / movement_distances[mask, np.newaxis]
            
            self.samples.append(relative_movements)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx])

class TrajectoryGenerator(nn.Module):
    def __init__(self, input_size=2, hidden_size=128):
        super(TrajectoryGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden

def train_model(model, train_loader, num_epochs=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_loss = float('inf')
    patience_counter = 0
    
    # 记录训练历史
    history = {
        'loss': [],
        'main_loss': [],
        'smoothness_loss': [],
        'learning_rate': []
    }
    
    print("开始训练...")
    start_time = datetime.now()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_main_loss = 0
        total_smoothness_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # 创建输入序列和目标序列
            input_seq = batch[:, :-1, :]
            target_seq = batch[:, 1:, :]
            
            optimizer.zero_grad()
            output, _ = model(input_seq)
            
            # 添加平滑损失
            smoothness_loss = torch.mean(torch.norm(output[:, 1:] - output[:, :-1], dim=2))
            
            # 组合重建损失和平滑损失
            main_loss = criterion(output, target_seq)
            loss = main_loss + 0.1 * smoothness_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_smoothness_loss += smoothness_loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_main_loss = total_main_loss / len(train_loader)
        avg_smoothness_loss = total_smoothness_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['loss'].append(avg_loss)
        history['main_loss'].append(avg_main_loss)
        history['smoothness_loss'].append(avg_smoothness_loss)
        history['learning_rate'].append(current_lr)
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_trajectory_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 30:
            print("Early stopping triggered")
            break
            
        if (epoch + 1) % 10 == 0:
            time_elapsed = datetime.now() - start_time
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {avg_loss:.4f}, '
                  f'Main Loss: {avg_main_loss:.4f}, '
                  f'Smoothness Loss: {avg_smoothness_loss:.4f}, '
                  f'LR: {current_lr:.6f}, '
                  f'Time: {time_elapsed}')
    
    # 保存训练历史图表
    plt.figure(figsize=(15, 10))
    
    # 绘制总损失
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制主损失
    plt.subplot(2, 2, 2)
    plt.plot(history['main_loss'])
    plt.title('Main Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制平滑损失
    plt.subplot(2, 2, 3)
    plt.plot(history['smoothness_loss'])
    plt.title('Smoothness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制学习率
    plt.subplot(2, 2, 4)
    plt.plot(history['learning_rate'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    
    # 调整子图布局
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_history_{timestamp}.png')
    print(f"训练历史已保存到 training_history_{timestamp}.png")
    
    # 同时保存历史数据
    history_data = {
        'loss': history['loss'],
        'main_loss': history['main_loss'],
        'smoothness_loss': history['smoothness_loss'],
        'learning_rate': history['learning_rate']
    }
    with open(f'training_history_{timestamp}.json', 'w') as f:
        json.dump(history_data, f)
    print(f"训练数据已保存到 training_history_{timestamp}.json")
    
    total_time = datetime.now() - start_time
    print(f"训练完成！总用时: {total_time}")
    
    return history

def generate_trajectory(model, start_point, end_point, num_points=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    with torch.no_grad():
        target_vector = end_point - start_point
        distance = np.linalg.norm(target_vector)
        if distance < 1e-6:
            return np.array([start_point])
            
        normalized_vector = target_vector / distance
        
        # 初始化轨迹
        trajectory = [start_point]
        current_input = torch.zeros((1, 1, 2)).to(device)
        current_input[0, 0] = torch.FloatTensor(normalized_vector).to(device)
        
        # 生成轨迹点
        for _ in range(num_points - 1):
            output, _ = model(current_input)
            delta = output[0, 0].cpu().numpy()
            next_point = trajectory[-1] + delta * (distance / num_points)
            trajectory.append(next_point)
            
            current_direction = (next_point - trajectory[-2]) / np.linalg.norm(next_point - trajectory[-2])
            current_input[0, 0] = torch.FloatTensor(current_direction).to(device)
    
    trajectory = np.array(trajectory)
    
    # 确保轨迹的起点和终点正确
    trajectory[0] = start_point
    trajectory[-1] = end_point
    
    # 使用更强的平滑处理
    smoothed = np.zeros_like(trajectory)
    smoothed[0] = trajectory[0]
    smoothed[-1] = trajectory[-1]
    
    # 多次平滑
    for _ in range(3):
        alpha = 0.4
        for i in range(1, len(trajectory)-1):
            smoothed[i] = trajectory[i] * (1-alpha) + (trajectory[i-1] + trajectory[i+1]) * alpha/2
        trajectory = smoothed.copy()
    
    return smoothed

def load_and_prepare_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return TrajectoryDataset(data['trajectories'])

if __name__ == "__main__":
    # 加载数据
    dataset = load_and_prepare_data('mouse_trajectories_20250128_124044.json')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建和训练模型
    model = TrajectoryGenerator()
    history = train_model(model, train_loader)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'trajectory_model.pth') 