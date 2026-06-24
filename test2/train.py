import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建保存图片的文件夹
save_dir = "bert_training_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==================== 设备检测和配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 显示 GPU 信息（如果可用）
if device.type == 'cuda':
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"当前显存占用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# 根据设备调整参数
if device.type == 'cuda':
    batch_size = 32           # GPU 可用较大批次（根据显存调整，如果显存不足可减到16或8）
    sample_size = None        # None 表示使用全部数据
    num_epochs = 3            # 训练轮数
    print("✓ GPU 模式：使用较大批次和全部数据")
else:
    batch_size = 4
    sample_size = 5000
    num_epochs = 2
    print("⚠️ CPU 模式：已自动减小批次大小和数据量以提高训练速度")

print("=" * 70)
print("实验二：新闻文本分类 - Bert 模型训练")
print("=" * 70)

# ==================== 1. 加载数据 ====================
print("\n[1/8] 加载数据...")
train_df = pd.read_csv("TrainSet.csv", sep='\t')
test_df = pd.read_csv("TestSet.csv", sep='\t')

print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")
print(f"类别数量: {train_df['label'].nunique()}")

# 获取标签
labels = sorted(train_df['label'].unique())
num_labels = len(labels)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# 使用部分数据加速训练（如果 sample_size 不是 None）
if sample_size is not None and sample_size < len(train_df):
    train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    print(f"使用 {len(train_df)} 条数据进行训练")
else:
    print(f"使用全部 {len(train_df)} 条数据进行训练")

# ==================== 2. 数据分布可视化 ====================
print("\n[2/8] 生成数据分布分析图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2.1 训练集标签分布
train_label_counts = train_df['label'].value_counts().sort_index()
axes[0, 0].bar(train_label_counts.index, train_label_counts.values, color='steelblue', alpha=0.7)
axes[0, 0].set_title('训练集标签分布', fontsize=14)
axes[0, 0].set_xlabel('类别标签')
axes[0, 0].set_ylabel('样本数量')
axes[0, 0].grid(True, alpha=0.3)

# 2.2 测试集标签分布
test_label_counts = test_df['label'].value_counts().sort_index()
axes[0, 1].bar(test_label_counts.index, test_label_counts.values, color='coral', alpha=0.7)
axes[0, 1].set_title('测试集标签分布', fontsize=14)
axes[0, 1].set_xlabel('类别标签')
axes[0, 1].set_ylabel('样本数量')
axes[0, 1].grid(True, alpha=0.3)

# 2.3 文本长度分布
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x).split()))
axes[1, 0].hist(train_df['text_length'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(train_df['text_length'].mean(), color='red', linestyle='--', 
                   label=f'均值: {train_df["text_length"].mean():.1f}')
axes[1, 0].set_title('文本长度分布', fontsize=14)
axes[1, 0].set_xlabel('文本长度（词数）')
axes[1, 0].set_ylabel('样本数量')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 2.4 标签占比饼图
axes[1, 1].pie(train_label_counts.values, labels=train_label_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('训练集标签占比', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '1_data_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {save_dir}/1_data_analysis.png")

# ==================== 3. 准备 Bert 数据集 ====================
print("\n[3/8] 准备 Bert 数据集...")

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.labels = labels.tolist() if isinstance(labels, pd.Series) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载 Bert tokenizer（添加超时和镜像源支持）
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print("✓ Tokenizer 加载成功")
except Exception as e:
    print(f"⚠️ 网络问题，尝试离线模式...")
    # 如果网络不行，可以手动下载后使用本地路径
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese-local', local_files_only=True)

# 划分验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'], train_df['label'],
    test_size=0.1,
    random_state=42,
    stratify=train_df['label']
)

print(f"训练样本数: {len(train_texts)}")
print(f"验证样本数: {len(val_texts)}")

# 创建 DataLoader（batch_size 已根据设备调整）
train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==================== 4. 加载 Bert 模型 ====================
print("\n[4/8] 加载 Bert 模型...")

try:
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-chinese',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    print("✓ 模型加载成功")
except Exception as e:
    print(f"⚠️ 网络问题，尝试离线模式...")
    model = BertForSequenceClassification.from_pretrained(
        './bert-base-chinese-local',
        num_labels=num_labels,
        local_files_only=True
    )

model.to(device)

# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 如果需要混合精度训练（可选的加速选项）
use_amp = False
if device.type == 'cuda':
    try:
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        print("✓ 启用混合精度训练（AMP）加速")
    except:
        print("⚠️ 混合精度训练不可用，使用标准训练")

# ==================== 5. 训练模型 ====================
print("\n[5/8] 开始训练...")
print(f"训练配置: {num_epochs} 轮, batch_size={batch_size}, 使用 {device}")

train_losses = []
val_losses = []
val_accuracies = []

best_val_accuracy = 0

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    total_train_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # 使用混合精度训练（如果启用）
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        total_train_loss += loss.item()
        
        # 打印进度
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 清理 GPU 缓存
        if device.type == 'cuda' and batch_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 验证阶段
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_accuracy)
    
    print(f"\nEpoch {epoch+1}/{num_epochs} 完成!")
    print(f"  训练损失: {avg_train_loss:.4f}")
    print(f"  验证损失: {avg_val_loss:.4f}")
    print(f"  验证准确率: {val_accuracy:.4f}")
    
    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        print(f"  ✓ 保存最佳模型 (准确率: {val_accuracy:.4f})")
    print("-" * 50)
    
    # 每个 epoch 后清理 GPU 缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ==================== 6. 训练过程可视化 ====================
print("\n[6/8] 生成训练过程对比图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 6.1 损失曲线
axes[0].plot(range(1, num_epochs+1), train_losses, 'o-', label='训练损失', color='steelblue', linewidth=2, markersize=8)
axes[0].plot(range(1, num_epochs+1), val_losses, 's-', label='验证损失', color='coral', linewidth=2, markersize=8)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('损失值', fontsize=12)
axes[0].set_title('训练与验证损失曲线对比', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 6.2 准确率曲线
axes[1].plot(range(1, num_epochs+1), val_accuracies, 'D-', label='验证准确率', color='green', linewidth=2, markersize=8)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('准确率', fontsize=12)
axes[1].set_title('验证准确率变化曲线', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '2_training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {save_dir}/2_training_curves.png")

# ==================== 7. 在测试集上评估 ====================
print("\n[7/8] 在测试集上评估模型...")

# 加载最佳模型
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device))

# 创建测试集 DataLoader
test_dataset = NewsDataset(test_df['text'], test_df['label'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 预测
model.eval()
all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
print(f"\n测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 分类报告
print("\n分类报告:")
print(classification_report(all_test_labels, all_test_preds))

# ==================== 8. 生成评估对比图 ====================
print("\n[8/8] 生成模型评估对比图...")

# 8.1 混淆矩阵
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(all_test_labels, all_test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Bert 模型混淆矩阵 (测试集准确率: {test_accuracy:.4f})', fontsize=14)
ax.set_xlabel('预测标签')
ax.set_ylabel('真实标签')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '3_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {save_dir}/3_confusion_matrix.png")

# 8.2 各类别性能对比
fig, ax = plt.subplots(figsize=(14, 6))
report_dict = classification_report(all_test_labels, all_test_preds, output_dict=True)
categories = [str(i) for i in range(num_labels)]

# 安全获取指标
precision = []
recall = []
f1 = []
for cat in categories:
    if cat in report_dict:
        precision.append(report_dict[cat]['precision'])
        recall.append(report_dict[cat]['recall'])
        f1.append(report_dict[cat]['f1-score'])
    else:
        precision.append(0)
        recall.append(0)
        f1.append(0)

x = np.arange(len(categories))
width = 0.25

ax.bar(x - width, precision, width, label='精确率', color='steelblue', alpha=0.8)
ax.bar(x, recall, width, label='召回率', color='coral', alpha=0.8)
ax.bar(x + width, f1, width, label='F1分数', color='green', alpha=0.8)

ax.set_xlabel('类别标签', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('Bert 模型各类别性能指标对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '4_category_performance.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {save_dir}/4_category_performance.png")

# 8.3 模型整体性能对比图
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['准确率', '宏平均精确率', '宏平均召回率', '宏平均F1']
values = [
    test_accuracy,
    report_dict.get('macro avg', {}).get('precision', 0),
    report_dict.get('macro avg', {}).get('recall', 0),
    report_dict.get('macro avg', {}).get('f1-score', 0)
]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylim([0, 1])
ax.set_ylabel('分数', fontsize=12)
ax.set_title('Bert 模型整体性能指标', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '5_model_performance_summary.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {save_dir}/5_model_performance_summary.png")

# 8.4 注意力可视化
print("\n生成注意力机制示意图...")
fig, ax = plt.subplots(figsize=(10, 8))
attention_matrix = np.random.rand(8, 8)
for i in range(8):
    attention_matrix[i, i] = 0.8
    for j in range(8):
        if abs(i-j) <= 2:
            attention_matrix[i, j] = 0.5

sns.heatmap(attention_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            xticklabels=[f'词{i+1}' for i in range(8)],
            yticklabels=[f'词{i+1}' for i in range(8)])
ax.set_title('多头注意力机制示意图（示例）', fontsize=14)
ax.set_xlabel('Key 位置')
ax.set_ylabel('Query 位置')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '6_attention_visualization.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {save_dir}/6_attention_visualization.png")

# ==================== 9. 保存实验结果 ====================
print("\n生成实验报告汇总...")

results_summary = pd.DataFrame({
    '指标': ['测试集准确率', '宏平均精确率', '宏平均召回率', '宏平均F1分数', '最佳验证准确率'],
    '数值': [
        f"{test_accuracy:.4f}",
        f"{report_dict.get('macro avg', {}).get('precision', 0):.4f}",
        f"{report_dict.get('macro avg', {}).get('recall', 0):.4f}",
        f"{report_dict.get('macro avg', {}).get('f1-score', 0):.4f}",
        f"{best_val_accuracy:.4f}"
    ]
})

print("\n实验结果汇总:")
print(results_summary.to_string(index=False))

results_summary.to_csv(os.path.join(save_dir, 'experiment_results.csv'), index=False, encoding='utf-8-sig')

with open(os.path.join(save_dir, 'training_log.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("实验二：新闻文本分类 - Bert 模型训练日志\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"使用设备: {device}\n")
    if device.type == 'cuda':
        f.write(f"GPU 型号: {torch.cuda.get_device_name(0)}\n")
        f.write(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    f.write(f"训练样本数: {len(train_texts)}\n")
    f.write(f"验证样本数: {len(val_texts)}\n")
    f.write(f"测试样本数: {len(test_df)}\n")
    f.write(f"类别数量: {num_labels}\n")
    f.write(f"训练轮数: {num_epochs}\n")
    f.write(f"批次大小: {batch_size}\n")
    f.write(f"混合精度训练: {use_amp}\n\n")
    f.write("训练损失:\n")
    for i, loss in enumerate(train_losses):
        f.write(f"  Epoch {i+1}: {loss:.4f}\n")
    f.write("\n验证损失:\n")
    for i, loss in enumerate(val_losses):
        f.write(f"  Epoch {i+1}: {loss:.4f}\n")
    f.write("\n验证准确率:\n")
    for i, acc in enumerate(val_accuracies):
        f.write(f"  Epoch {i+1}: {acc:.4f}\n")
    f.write(f"\n最终测试准确率: {test_accuracy:.4f}\n")

print(f"\n✓ 所有结果已保存到: {save_dir}/")

print("\n" + "=" * 70)
print("实验完成！")
print(f"生成的图片保存在: {save_dir}/")
print("图片列表:")
for img in os.listdir(save_dir):
    if img.endswith(('.png', '.jpg')):
        print(f"  - {img}")
print("=" * 70)

# 显示 GPU 使用统计
if device.type == 'cuda':
    print("\nGPU 训练统计:")
    print(f"  最大显存占用: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    print(f"  显存占用峰值: {torch.cuda.max_memory_reserved(0) / 1e9:.2f} GB")