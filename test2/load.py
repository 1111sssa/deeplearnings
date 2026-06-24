import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 修复中文乱码 ====================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 如果中文还是乱码，设置为 True 使用英文标签
USE_ENGLISH = False

# ==================== 设置保存路径 ====================
# 创建 picture 文件夹（如果不存在）
picture_dir = "picture"
if not os.path.exists(picture_dir):
    os.makedirs(picture_dir)
    print(f"✓ 创建文件夹: {picture_dir}")

# 原来的训练结果文件夹（用于读取模型）
train_dir = "bert_training_results"

print("=" * 70)
print("加载已训练的 Bert 模型并生成评估图表")
print(f"图片保存路径: {picture_dir}/")
print("=" * 70)

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ==================== 1. 加载数据 ====================
print("\n[1/7] 加载数据...")
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

# ==================== 2. 数据分布可视化（生成 1_data_analysis.png）====================
print("\n[2/7] 生成数据分布分析图...")

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
plt.savefig(os.path.join(picture_dir, '1_data_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {picture_dir}/1_data_analysis.png")

# ==================== 3. 加载模型和分词器 ====================
print("\n[3/7] 加载模型和分词器...")

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载分词器
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print("✓ Tokenizer 加载成功")
except:
    print("⚠️ 尝试离线模式...")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese-local', local_files_only=True)

# 加载模型
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False
)

# 加载训练好的权重
model_path = os.path.join(train_dir, 'best_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ 成功加载模型权重: {model_path}")
else:
    print(f"❌ 找不到模型文件: {model_path}")
    exit()

model.to(device)
model.eval()
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 4. 准备数据集 ====================
print("\n[4/7] 准备数据集...")

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

# 创建 DataLoader
batch_size = 32
test_dataset = NewsDataset(test_df['text'], test_df['label'], tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 5. 在测试集上评估 ====================
print("\n[5/7] 在测试集上评估模型...")

all_test_preds = []
all_test_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  处理进度: {batch_idx + 1}/{len(test_loader)}")

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
print(f"\n测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 分类报告
print("\n分类报告:")
print(classification_report(all_test_labels, all_test_preds))

# ==================== 6. 生成评估图表 ====================
print("\n[6/7] 生成评估图表...")

# 根据是否使用英文来设置标签
if USE_ENGLISH:
    title_labels = {
        'confusion_matrix': f'Confusion Matrix - BERT Model (Test Acc: {test_accuracy:.4f})',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'accuracy': 'Accuracy',
        'macro_avg_precision': 'Macro Avg Precision',
        'macro_avg_recall': 'Macro Avg Recall',
        'macro_avg_f1': 'Macro Avg F1',
        'categories': 'Categories',
        'scores': 'Scores'
    }
else:
    title_labels = {
        'confusion_matrix': f'Bert 模型混淆矩阵 (测试集准确率: {test_accuracy:.4f})',
        'precision': '精确率',
        'recall': '召回率',
        'f1': 'F1分数',
        'accuracy': '准确率',
        'macro_avg_precision': '宏平均精确率',
        'macro_avg_recall': '宏平均召回率',
        'macro_avg_f1': '宏平均F1分数',
        'categories': '类别标签',
        'scores': '分数'
    }

# 6.1 混淆矩阵（对应原来的 3_confusion_matrix.png）
fig, ax = plt.subplots(figsize=(12, 10))
cm = confusion_matrix(all_test_labels, all_test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            annot_kws={'size': 10} if cm.max() > 100 else {'size': 12})
ax.set_title(title_labels['confusion_matrix'], fontsize=14, fontweight='bold')
ax.set_xlabel('预测标签', fontsize=12)
ax.set_ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, '3_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {picture_dir}/3_confusion_matrix.png")

# 6.2 各类别性能对比（对应原来的 4_category_performance.png）
fig, ax = plt.subplots(figsize=(14, 6))
report_dict = classification_report(all_test_labels, all_test_preds, output_dict=True)
categories = [str(i) for i in range(num_labels)]

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

ax.bar(x - width, precision, width, label=title_labels['precision'], color='steelblue', alpha=0.8)
ax.bar(x, recall, width, label=title_labels['recall'], color='coral', alpha=0.8)
ax.bar(x + width, f1, width, label=title_labels['f1'], color='green', alpha=0.8)

ax.set_xlabel(title_labels['categories'], fontsize=12)
ax.set_ylabel(title_labels['scores'], fontsize=12)
ax.set_title('Bert 模型各类别性能指标对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, '4_category_performance.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {picture_dir}/4_category_performance.png")

# 6.3 模型整体性能对比图（对应原来的 5_model_performance_summary.png）
fig, ax = plt.subplots(figsize=(10, 6))
metrics = [title_labels['accuracy'], title_labels['macro_avg_precision'], 
           title_labels['macro_avg_recall'], title_labels['macro_avg_f1']]
values = [
    test_accuracy,
    report_dict.get('macro avg', {}).get('precision', 0),
    report_dict.get('macro avg', {}).get('recall', 0),
    report_dict.get('macro avg', {}).get('f1-score', 0)
]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylim([0, 1])
ax.set_ylabel(title_labels['scores'], fontsize=12)
ax.set_title('Bert 模型整体性能指标', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(picture_dir, '5_model_performance_summary.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {picture_dir}/5_model_performance_summary.png")

# 6.4 注意力可视化（对应原来的 6_attention_visualization.png）
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
ax.set_title('多头注意力机制示意图', fontsize=14)
ax.set_xlabel('Key 位置')
ax.set_ylabel('Query 位置')
plt.tight_layout()
plt.savefig(os.path.join(picture_dir, '6_attention_visualization.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ 已保存: {picture_dir}/6_attention_visualization.png")

# ==================== 7. 尝试还原训练曲线（如果存在日志文件）====================
print("\n[7/7] 尝试还原训练曲线...")

# 检查是否有训练日志文件
log_file = os.path.join(train_dir, 'training_log.txt')
if os.path.exists(log_file):
    print("找到训练日志文件，正在提取训练数据...")
    
    # 从日志文件中读取训练损失和验证准确率
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        reading_train_loss = False
        reading_val_loss = False
        reading_val_acc = False
        
        for line in lines:
            if '训练损失:' in line:
                reading_train_loss = True
                continue
            elif '验证损失:' in line:
                reading_train_loss = False
                reading_val_loss = True
                continue
            elif '验证准确率:' in line:
                reading_val_loss = False
                reading_val_acc = True
                continue
            elif reading_train_loss and line.strip() and 'Epoch' in line:
                # 提取训练损失
                parts = line.split(':')
                if len(parts) == 2:
                    train_losses.append(float(parts[1].strip()))
            elif reading_val_loss and line.strip() and 'Epoch' in line:
                # 提取验证损失
                parts = line.split(':')
                if len(parts) == 2:
                    val_losses.append(float(parts[1].strip()))
            elif reading_val_acc and line.strip() and 'Epoch' in line:
                # 提取验证准确率
                parts = line.split(':')
                if len(parts) == 2:
                    val_accuracies.append(float(parts[1].strip()))
    
    if train_losses and val_losses and val_accuracies:
        num_epochs = len(train_losses)
        
        # 生成训练曲线图（对应原来的 2_training_curves.png）
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(range(1, num_epochs+1), train_losses, 'o-', label='训练损失', color='steelblue', linewidth=2, markersize=8)
        axes[0].plot(range(1, num_epochs+1), val_losses, 's-', label='验证损失', color='coral', linewidth=2, markersize=8)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('损失值', fontsize=12)
        axes[0].set_title('训练与验证损失曲线对比', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(range(1, num_epochs+1), val_accuracies, 'D-', label='验证准确率', color='green', linewidth=2, markersize=8)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('准确率', fontsize=12)
        axes[1].set_title('验证准确率变化曲线', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, '2_training_curves.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ 已保存: {picture_dir}/2_training_curves.png")
    else:
        print("⚠️ 无法从日志中提取完整的训练数据")
else:
    print("⚠️ 未找到训练日志文件，无法还原训练曲线图")

# ==================== 保存实验结果 ====================
results_summary = pd.DataFrame({
    '指标': ['测试集准确率', '宏平均精确率', '宏平均召回率', '宏平均F1分数'],
    '数值': [
        f"{test_accuracy:.4f}",
        f"{report_dict.get('macro avg', {}).get('precision', 0):.4f}",
        f"{report_dict.get('macro avg', {}).get('recall', 0):.4f}",
        f"{report_dict.get('macro avg', {}).get('f1-score', 0):.4f}"
    ]
})

results_summary.to_csv(os.path.join(picture_dir, 'experiment_results.csv'), index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("所有图片生成完成！")
print(f"图片保存位置: {picture_dir}/")
print("\n生成的图片列表:")
for img in sorted(os.listdir(picture_dir)):
    if img.endswith(('.png', '.jpg', '.csv')):
        print(f"  - {img}")
print("=" * 70)