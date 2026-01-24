import torch
import torch.nn as nn
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
# 在 [0, 2π] 范围内生成均匀分布的点
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# 生成sin函数值，并添加一些噪声
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)

# 转换为torch tensor
X = torch.from_numpy(X_numpy).float()  # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"X范围: [{X.min().item():.2f}, {X.max().item():.2f}]")
print(f"y范围: [{y.min().item():.2f}, {y.max().item():.2f}]")
print("---" * 10)

# 2. 定义多层神经网络模型
# 使用nn.Module定义网络结构
class SinNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 128, 64], output_dim=1):
        """
        多层神经网络，用于拟合sin函数
        Args:
            input_dim: 输入维度（这里是1，因为只有一个x值）
            hidden_dims: 隐藏层维度列表，例如 [64, 128, 64] 表示三层隐藏层
            output_dim: 输出维度（这里是1，因为只输出一个y值）
        """
        super(SinNet, self).__init__()
        layers = []
        
        # 输入层到第一个隐藏层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 使用ReLU激活函数
            prev_dim = hidden_dim
        
        # 最后一个隐藏层到输出层（不使用激活函数，因为这是回归任务）
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 创建模型实例
# 可以调整hidden_dims来改变网络结构
model = SinNet(input_dim=1, hidden_dims=[64, 128, 64], output_dim=1)

# 打印模型结构
print("模型结构:")
print(model)
print("---" * 10)

# 统计模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 回归任务使用均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，学习率0.001

# 4. 训练模型
num_epochs = 2000
loss_history = []  # 记录loss历史，用于可视化

print("开始训练...")
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度，torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 记录loss
    loss_history.append(loss.item())
    
    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print(f"最终Loss: {loss_history[-1]:.6f}")
print("---" * 10)

# 5. 使用训练好的模型进行预测
model.eval()  # 设置为评估模式
with torch.no_grad():
    y_predicted = model(X)

# 为了绘制更平滑的曲线，生成更多的测试点
X_test_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
X_test = torch.from_numpy(X_test_numpy).float()
with torch.no_grad():
    y_test_predicted = model(X_test)

# 6. 可视化结果
# 创建两个子图：一个显示拟合效果，一个显示loss变化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 子图1：拟合效果对比
ax1.scatter(X_numpy, y_numpy, label='训练数据 (带噪声)', color='blue', alpha=0.5, s=20)
ax1.plot(X_numpy, np.sin(X_numpy), label='真实sin函数', color='green', linewidth=2, linestyle='--')
ax1.plot(X_test_numpy, y_test_predicted.numpy(), label='神经网络拟合', color='red', linewidth=2)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('多层神经网络拟合sin函数', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：Loss变化曲线
ax2.plot(range(1, num_epochs + 1), loss_history, color='blue', linewidth=1.5)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (MSE)', fontsize=12)
ax2.set_title('训练过程中的Loss变化', fontsize=14, fontweight='bold')
ax2.set_yscale('log')  # 使用对数刻度，更好地观察loss下降
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sin_function_fitting.png', dpi=300, bbox_inches='tight')
print("可视化结果已保存为: sin_function_fitting.png")
plt.show()

# 7. 计算拟合误差
mse = loss_fn(y_predicted, y).item()
print(f"\n均方误差 (MSE): {mse:.6f}")
print(f"均方根误差 (RMSE): {np.sqrt(mse):.6f}")
