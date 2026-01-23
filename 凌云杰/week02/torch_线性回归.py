import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数的模拟数据
X_numpy = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)  # 生成0到2π的100个点，形状(100,1)
Y_numpy = np.sin(X_numpy) + 0.1*np.random.randn(100,1)  # sin函数+少量噪声

# 转换为torch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(Y_numpy).float()

print('数据生成完成')
print('---'*10)

# 2. 创建sin函数的拟合参数（标准sin方程：y = a * sin(b * x + c) + d ）
# requires_grad=True 保持不变，告诉pytorch计算梯度
a = torch.randn(1, requires_grad=True, dtype=torch.float)  # 振幅
b = torch.randn(1, requires_grad=True, dtype=torch.float)  # 频率
c = torch.randn(1, requires_grad=True, dtype=torch.float)  # 相位
d = torch.randn(1, requires_grad=True, dtype=torch.float)  # 偏移

# 3. 损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 回归任务仍用均方误差
optimizer = torch.optim.SGD([a,b,c,d], lr=0.01)  # 学习率调大一点，sin拟合需要稍大lr

# 4.训练模型
num_epochs = 5000  # 训练轮数稍增，sin拟合比线性慢一点
for epoch in range(num_epochs):
    # 前向传播：把线性公式改成sin公式
    y_pred = a * torch.sin(b * X + c) + d

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100轮打印一次（原代码是每1轮，改成每100轮减少输出）
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] , loss: {loss.item():.4f}')

# 5. 打印最终学习到的sin参数
print('\n 训练完成！ ')
a_learned = a.item()
b_learned = b.item()
c_learned = c.item()
d_learned = d.item()
print(f'拟合的振幅 a: {a_learned:.4f} (真实值≈1)')
print(f'拟合的频率 b: {b_learned:.4f} (真实值≈1)')
print(f'拟合的相位 c: {c_learned:.4f} (真实值≈0)')
print(f'拟合的偏移 d: {d_learned:.4f} (真实值≈0)')
print('---'*10)

# 6.绘制结果
with torch.no_grad():
    y_predicted = a_learned * np.sin(b_learned * X_numpy + c_learned) + d_learned  # 拟合y

plt.figure(figsize=(10,6))
plt.scatter(X_numpy,Y_numpy,label = 'Raw data', color ='blue', alpha=0.6)
plt.plot(X_numpy,y_predicted,label = f'Model: y ={a_learned:.2f}*sin({b_learned:.2f}x+{c_learned:.2f})+{d_learned:.2f}',color='red',linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()