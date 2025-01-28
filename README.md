# Mouse Trajectory Generator

基于深度学习的鼠标移动轨迹生成器，使用LSTM神经网络学习并模拟真实的人类鼠标移动特征。

[English](README_EN.md) | 简体中文

## ✨ 特性

- 144Hz高精度轨迹采集
- LSTM深度学习模型
- 平滑自然的轨迹生成
- 实时训练过程可视化
- 轨迹对比分析工具

## 📦 安装

```bash
# 克隆本项目
git clone https://github.com/JkAicodeShare/Mouse-Trajectory-Generator.git

# 安装依赖
pip install -r requirements.txt
```

## 🔨 使用方法

### 1. 采集数据

```bash
python mouse_data_collector.py
```

- 点击红色目标点开始记录
- 移动鼠标到新目标点
- 自动保存轨迹数据

### 2. 训练模型

```bash
python mouse_trajectory_model.py
```

输出文件：
- `trajectory_model.pth`: 训练模型
- `training_history_*.png`: 训练过程图
- `training_history_*.json`: 训练数据

### 3. 测试生成

```bash
python test_trajectory.py
```

- 空格键：生成新轨迹
- 蓝色线：AI轨迹
- 粉色线：真实轨迹
- 绿点：起点
- 红点：终点

## 🛠️ 核心功能

### 数据采集器 (mouse_data_collector.py)
- 144Hz采样率
- 实时轨迹显示
- 自动保存数据
- 异常轨迹过滤

### 轨迹模型 (mouse_trajectory_model.py)
- 3层LSTM (128维)
- 自适应学习率
- 平滑度损失函数
- 早停机制

### 测试工具 (test_trajectory.py)
- 实时轨迹生成
- 真实轨迹对比
- 可视化分析
- 轨迹平滑处理

## 📈 模型参数

```python
# 网络结构
input_size = 2          # 坐标维度
hidden_size = 128       # 隐藏层大小
num_layers = 3          # LSTM层数
fc_layers = [64, 32, 2] # 全连接层

# 训练参数
batch_size = 32         # 批次大小
learning_rate = 0.001   # 学习率
max_epochs = 500        # 最大轮数
early_stop = 30         # 早停轮数
```

## 💡 常见问题

#### Q: 最少需要多少训练数据？
A: 建议至少100条轨迹，越多越好

#### Q: 如何调整轨迹平滑度？
A: 修改 generate_trajectory 中的 alpha 参数

#### Q: 训练时间多久？
A: GPU约10秒，CPU约30秒

## 🚀 开发计划

- [ ] 速度曲线控制
- [ ] 多样化轨迹生成
- [ ] 数据增强优化
- [ ] 轨迹评估指标
- [ ] 特征提取支持

## 📄 许可证

MIT License

