import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist() # 数据集第一列 文本
string_labels = dataset[1].tolist() # 数据集第二列 类别

# 类别转换数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 原始的文本构建一个词典，字 -》 数字
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 取文本的前40个字符
max_len = 40

tokenized_texts = []
for text in texts:
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    tokenized_texts.append(tokenized)

labels_tensor = torch.tensor(numerical_labels, dtype=torch.long)

# term frequency
def create_bow_vectors(tokenized_texts, vocab_size):
    bow_vectors = []
    for text_indices in tokenized_texts:
        bow_vector = torch.zeros(vocab_size) # 词典个数长度的向量，存储每个字在这个文本中间出现的次数
        for index in text_indices:
            if index != 0:  # Ignore padding
                bow_vector[index] += 1
        bow_vectors.append(bow_vector)
    return torch.stack(bow_vectors)

bow_matrix = create_bow_vectors(tokenized_texts, vocab_size)
input_size = vocab_size

# 灵活的模型类，支持动态层数和节点数
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，例如 [64, 128] 表示两层，第一层64节点，第二层128节点
            output_dim: 输出维度（类别数）
        """
        super(FlexibleClassifier, self).__init__()
        layers = []
        
        # 构建所有层
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

output_dim = len(label_to_index)

# 定义不同的模型配置：[(配置名称, 隐藏层维度列表), ...]
model_configs = [
    ("1层-64节点", [64]),
    ("1层-128节点", [128]),
    ("1层-256节点", [256]),
    ("2层-64-32节点", [64, 32]),
    ("2层-128-64节点", [128, 64]),
    ("2层-256-128节点", [256, 128]),
    ("3层-128-64-32节点", [128, 64, 32]),
    ("3层-256-128-64节点", [256, 128, 64]),
]

# 存储所有模型的训练loss历史
all_losses = {}
num_epochs = 20

print("=" * 80)
print("开始训练不同配置的模型，对比loss变化")
print("=" * 80)

for config_name, hidden_dims in model_configs:
    print(f"\n训练模型: {config_name} (隐藏层: {hidden_dims})")
    
    # 创建模型
    model = FlexibleClassifier(input_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 记录loss历史
    loss_history = []
    
    # 训练
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(bow_matrix)
        loss = criterion(outputs, labels_tensor)
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    all_losses[config_name] = loss_history
    print(f"  最终Loss: {loss_history[-1]:.4f}")

# 打印对比结果
print("\n" + "=" * 80)
print("模型Loss对比总结")
print("=" * 80)
print(f"{'模型配置':<25} {'初始Loss':<12} {'最终Loss':<12} {'Loss下降':<12}")
print("-" * 80)

for config_name, loss_history in all_losses.items():
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = initial_loss - final_loss
    print(f"{config_name:<25} {initial_loss:<12.4f} {final_loss:<12.4f} {loss_reduction:<12.4f}")

# 绘制loss变化曲线
plt.figure(figsize=(12, 8))
for config_name, loss_history in all_losses.items():
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label=config_name, markersize=3)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('不同模型配置的Loss变化对比', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_loss_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nLoss对比图已保存为: model_loss_comparison.png")
plt.show()

# 使用最佳模型进行预测（选择最终loss最小的模型）
best_config = min(all_losses.items(), key=lambda x: x[1][-1])[0]
# 从model_configs中找到对应的hidden_dims
best_hidden_dims = None
for name, dims in model_configs:
    if name == best_config:
        best_hidden_dims = dims
        break

print(f"\n最佳模型配置: {best_config} (隐藏层: {best_hidden_dims})")
best_model = FlexibleClassifier(input_size, best_hidden_dims, output_dim)
best_optimizer = optim.SGD(best_model.parameters(), lr=0.01)
for epoch in range(num_epochs):
    best_model.train()
    best_optimizer.zero_grad()
    outputs = best_model(bow_matrix)
    loss = nn.CrossEntropyLoss()(outputs, labels_tensor)
    loss.backward()
    best_optimizer.step()
model = best_model  # 用于后续的分类测试


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    # 正向传播，11 神经元的输出
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
