import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 读取数据
# 真实标签 (111.csv)
true_labels_df = pd.read_csv('111.csv')
# 预测标签 (Net1_predict_20260602_165518.csv)
pred_labels_df = pd.read_csv('Net1_predict_20260602_165518.csv')

print("真实标签数据前5行:")
print(true_labels_df.head())
print("\n预测标签数据前5行:")
print(pred_labels_df.head())

# 2. 数据预处理
# 将预测标签从one-hot格式转换为单标签格式
pred_labels_df['pred_label'] = pred_labels_df[['label_0', 'label_1', 'label_2', 'label_3']].idxmax(axis=1).str.replace('label_', '').astype(int)

# 合并真实标签和预测标签
merged_df = pd.merge(true_labels_df, pred_labels_df[['id', 'pred_label']], on='id', how='inner')

print(f"\n合并后数据形状: {merged_df.shape}")
print("\n合并后数据前5行:")
print(merged_df.head())

# 3. 计算评估指标
y_true = merged_df['label']
y_pred = merged_df['pred_label']

# 计算各项指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n" + "="*50)
print("模型评估指标:")
print("="*50)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数 (F1-Score): {f1:.4f}")
print("="*50)

# 分类报告
print("\n分类报告:")
print(classification_report(y_true, y_pred, digits=4))

# 4. 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
# 获取所有类别
classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

# 5. 可视化图表

# 创建图形
fig = plt.figure(figsize=(18, 12))

# 5.1 混淆矩阵热力图
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, ax=ax1)
ax1.set_title('混淆矩阵 (Confusion Matrix)', fontsize=14, fontweight='bold')
ax1.set_xlabel('预测标签 (Predicted Label)')
ax1.set_ylabel('真实标签 (True Label)')

# 5.2 各类别准确率
ax2 = plt.subplot(2, 3, 2)
class_accuracy = cm.diagonal() / cm.sum(axis=1)
# 处理除零情况
class_accuracy = np.nan_to_num(class_accuracy)
bars = ax2.bar(classes, class_accuracy, color='skyblue', edgecolor='black')
ax2.set_title('各类别准确率 (Per-class Accuracy)', fontsize=14, fontweight='bold')
ax2.set_xlabel('类别 (Class)')
ax2.set_ylabel('准确率 (Accuracy)')
ax2.set_ylim([0, 1.1])
# 在柱状图上显示数值
for bar, acc in zip(bars, class_accuracy):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.3f}', ha='center', va='bottom')

# 5.3 真实标签分布
ax3 = plt.subplot(2, 3, 3)
true_counts = pd.Series(y_true).value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax3.bar(true_counts.index, true_counts.values, color=colors[:len(true_counts)], edgecolor='black')
ax3.set_title('真实标签分布 (True Label Distribution)', fontsize=14, fontweight='bold')
ax3.set_xlabel('类别 (Class)')
ax3.set_ylabel('数量 (Count)')
# 显示数值
for bar, count in zip(bars, true_counts.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom')

# 5.4 预测标签分布
ax4 = plt.subplot(2, 3, 4)
pred_counts = pd.Series(y_pred).value_counts().sort_index()
bars = ax4.bar(pred_counts.index, pred_counts.values, color=colors[:len(pred_counts)], edgecolor='black')
ax4.set_title('预测标签分布 (Predicted Label Distribution)', fontsize=14, fontweight='bold')
ax4.set_xlabel('类别 (Class)')
ax4.set_ylabel('数量 (Count)')
# 显示数值
for bar, count in zip(bars, pred_counts.values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom')

# 5.5 混淆矩阵归一化 (百分比)
ax5 = plt.subplot(2, 3, 5)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=classes, yticklabels=classes, ax=ax5)
ax5.set_title('混淆矩阵 (归一化百分比)', fontsize=14, fontweight='bold')
ax5.set_xlabel('预测标签 (Predicted Label)')
ax5.set_ylabel('真实标签 (True Label)')

# 5.6 评估指标条形图
ax6 = plt.subplot(2, 3, 6)
metrics = ['准确率', '精确率', '召回率', 'F1分数']
values = [accuracy, precision, recall, f1]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax6.bar(metrics, values, color=colors_bar, edgecolor='black')
ax6.set_title('模型评估指标', fontsize=14, fontweight='bold')
ax6.set_ylabel('分数 (Score)')
ax6.set_ylim([0, 1.1])
# 显示数值
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n图表已保存为: model_evaluation_results.png")

# 6. 额外分析 - 错误预测分析
print("\n" + "="*50)
print("错误预测分析:")
print("="*50)
# 找出预测错误的样本
merged_df['correct'] = merged_df['label'] == merged_df['pred_label']
error_df = merged_df[~merged_df['correct']]

if len(error_df) > 0:
    print(f"错误预测样本数量: {len(error_df)}")
    print(f"错误率: {len(error_df)/len(merged_df):.4f}")
    print("\n错误预测样本示例:")
    print(error_df[['id', 'label', 'pred_label']].head(10))
    
    # 错误类型分布
    error_df['error_type'] = error_df.apply(lambda x: f"{int(x['label'])}→{int(x['pred_label'])}", axis=1)
    error_counts = error_df['error_type'].value_counts()
    print("\n错误类型分布:")
    print(error_counts)
    
    # 绘制错误类型分布图
    fig2, ax = plt.subplots(figsize=(10, 6))
    error_counts.head(10).plot(kind='bar', color='coral', edgecolor='black')
    ax.set_title('错误预测类型分布 (Top 10)', fontsize=14, fontweight='bold')
    ax.set_xlabel('真实→预测 (True→Predicted)')
    ax.set_ylabel('数量 (Count)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n错误分析图已保存为: error_analysis.png")
else:
    print("所有样本预测正确！")

# 7. 保存详细结果到CSV
merged_df.to_csv('evaluation_results.csv', index=False)
print(f"\n详细结果已保存为: evaluation_results.csv")
print(f"包含 {len(merged_df)} 个样本的预测和真实标签对比")

# 8. 生成统计摘要
print("\n" + "="*50)
print("统计摘要:")
print("="*50)
print(f"总样本数: {len(merged_df)}")
print(f"类别数量: {len(classes)}")
print(f"正确预测数: {merged_df['correct'].sum()}")
print(f"错误预测数: {len(merged_df) - merged_df['correct'].sum()}")
print(f"总体准确率: {accuracy:.4f}")