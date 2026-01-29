import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 1. 数据预处理
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签编码
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

# 字符编码
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40

# 划分训练/测试集（8:2）
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels
)


# 2. 自定义Dataset
class CharSeqDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# 构建DataLoader
train_loader = DataLoader(CharSeqDataset(train_texts, train_labels, char_to_index, max_len),
                          batch_size=32, shuffle=True)
test_loader = DataLoader(CharSeqDataset(test_texts, test_labels, char_to_index, max_len),
                         batch_size=32, shuffle=False)


# 3. 通用序列分类模型
class SeqClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_type="lstm"):
        super().__init__()
        self.model_type = model_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.model_type == "rnn":
            self.seq_layer = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        elif self.model_type == "lstm":
            self.seq_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.model_type == "gru":
            self.seq_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be 'rnn'/'lstm'/'gru'")

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        if self.model_type == "lstm":
            _, (hidden, _) = self.seq_layer(embedded)
        else:
            _, hidden = self.seq_layer(embedded)
        out = self.fc(hidden.squeeze(0))
        return out


# 4. 训练与评估函数
def train_and_evaluate(model_type, epochs=4):
    """训练指定模型，记录训练时间、损失和精度"""
    # 模型初始化
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)
    model = SeqClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, model_type)

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练计时：记录总起始时间
    total_train_start = time.perf_counter()  # 高精度计时
    epoch_times = []  # 存储每轮训练时间

    print(f"\n===== 训练 {model_type.upper()} 模型 =====")
    for epoch in range(epochs):
        # 单轮训练计时
        epoch_start = time.perf_counter()

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 单轮结束计时
        epoch_end = time.perf_counter()
        epoch_duration = epoch_end - epoch_start  # 单轮耗时（秒）
        epoch_times.append(epoch_duration)

        # 打印每轮信息（含耗时）
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f} | 耗时: {epoch_duration:.2f}s")

    # 总训练时间计算
    total_train_end = time.perf_counter()
    total_train_duration = total_train_end - total_train_start  # 总耗时
    avg_epoch_duration = sum(epoch_times) / len(epoch_times)  # 平均每轮耗时

    # 评估精度
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)

    # 打印模型核心指标
    print(f"\n{model_type.upper()} 训练总结:")
    print(f"总训练时间: {total_train_duration:.2f}s")
    print(f"平均每轮时间: {avg_epoch_duration:.2f}s")
    print(f"测试精度: {accuracy:.4f}")
    print(f"最终训练损失: {avg_loss:.4f}")

    # 单条文本预测示例
    def predict(text):
        indices = [char_to_index.get(char, 0) for char in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        return index_to_label[pred_idx.item()]

    test_texts = ["帮我导航到北京", "查询明天北京的天气"]
    for text in test_texts:
        print(f"输入 '{text}' → 预测标签: {predict(text)}")

    # 返回包含时间的结果字典
    return {
        "model_type": model_type,
        "total_train_time": total_train_duration,  # 总训练时间（s）
        "avg_epoch_time": avg_epoch_duration,  # 平均每轮时间（s）
        "final_loss": avg_loss,
        "accuracy": accuracy
    }


# 5. 对比实验
results = []
for model_type in ["rnn", "lstm", "gru"]:
    res = train_and_evaluate(model_type)
    results.append(res)

# 6. 结果汇总
print("\n" + "=" * 60)
print("                      RNN/LSTM/GRU 对比实验结果")
print("=" * 60)
# 格式化输出表格（左对齐，保留2位小数）
print(f"{'模型':<8} {'总训练时间(s)':<15} {'平均每轮时间(s)':<18} {'最终损失':<12} {'测试精度':<10}")
print("-" * 60)
for res in results:
    print(f"{res['model_type'].upper():<8} "
          f"{res['total_train_time']:<15.2f} "
          f"{res['avg_epoch_time']:<18.2f} "
          f"{res['final_loss']:<12.4f} "
          f"{res['accuracy']:<10.4f}")
print("=" * 60)

# 7. 关键结论（自动输出）
print("\n【实验结论】")
# 按精度排序
acc_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)
# 按总训练时间排序
speed_sorted = sorted(results, key=lambda x: x["total_train_time"])

print(f"1. 精度最高: {acc_sorted[0]['model_type'].upper()} (精度: {acc_sorted[0]['accuracy']:.4f})")
print(f"2. 训练最快: {speed_sorted[0]['model_type'].upper()} (总耗时: {speed_sorted[0]['total_train_time']:.2f}s)")
print(
    f"3. 效率最优（平衡精度与速度）: {acc_sorted[1]['model_type'].upper() if acc_sorted[0]['model_type'] != speed_sorted[0]['model_type'] else acc_sorted[0]['model_type'].upper()}")