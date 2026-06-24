import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("心电图(ECG)模型预测评估系统")
print("="*60)

# 1. 读取数据
try:
    true_labels_df = pd.read_csv('111.csv')
    print(f"✓ 成功加载真实标签文件: 111.csv (共{len(true_labels_df)}条记录)")
    pred_labels_df = pd.read_csv('Net1_predict_20260602_165518.csv')
    print(f"✓ 成功加载预测标签文件: Net1_predict_20260602_165518.csv (共{len(pred_labels_df)}条记录)")
except FileNotFoundError as e:
    print(f"✗ 文件加载失败: {e}")
    exit()

print("\n真实标签数据前5行:")
print(true_labels_df.head())
print("\n预测标签数据前5行:")
print(pred_labels_df.head())

# 2. 数据预处理
pred_labels_df['pred_label'] = pred_labels_df[['label_0', 'label_1', 'label_2', 'label_3']].idxmax(axis=1).str.replace('label_', '').astype(int)
merged_df = pd.merge(true_labels_df, pred_labels_df[['id', 'pred_label']], on='id', how='inner')
print(f"\n✓ 数据合并完成: 共{len(merged_df)}个匹配样本")

y_true = merged_df['label']
y_pred = merged_df['pred_label']
classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

# 3. 计算评估指标
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

print("\n分类报告:")
print(classification_report(y_true, y_pred, digits=4))

# 4. 混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# ==================== 单独生成每张图 ====================

# 图1: 混淆矩阵热力图
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, ax=ax1,
            annot_kws={'size': 14})
ax1.set_title('混淆矩阵 (Confusion Matrix)', fontsize=16, fontweight='bold')
ax1.set_xlabel('预测标签 (Predicted Label)', fontsize=13)
ax1.set_ylabel('真实标签 (True Label)', fontsize=13)
plt.tight_layout()
plt.savefig('1_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图1已保存: 1_confusion_matrix.png")

# 图2: 各类别准确率
fig2, ax2 = plt.subplots(figsize=(8, 6))
class_accuracy = np.nan_to_num(cm.diagonal() / cm.sum(axis=1))
bars = ax2.bar(classes, class_accuracy, color='skyblue', edgecolor='black', linewidth=1.5)
ax2.set_title('各类别准确率 (Per-class Accuracy)', fontsize=16, fontweight='bold')
ax2.set_xlabel('类别 (Class)', fontsize=13)
ax2.set_ylabel('准确率 (Accuracy)', fontsize=13)
ax2.set_ylim([0, 1.1])
ax2.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'总体准确率={accuracy:.3f}')
ax2.legend()
for bar, acc in zip(bars, class_accuracy):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('2_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图2已保存: 2_class_accuracy.png")

# 图3: 真实标签分布
fig3, ax3 = plt.subplots(figsize=(8, 6))
true_counts = pd.Series(y_true).value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax3.bar(true_counts.index, true_counts.values, color=colors[:len(true_counts)], edgecolor='black', linewidth=1.5)
ax3.set_title('真实标签分布 (True Label Distribution)', fontsize=16, fontweight='bold')
ax3.set_xlabel('类别 (Class)', fontsize=13)
ax3.set_ylabel('样本数量 (Count)', fontsize=13)
for bar, count in zip(bars, true_counts.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('3_true_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图3已保存: 3_true_distribution.png")

# 图4: 预测标签分布
fig4, ax4 = plt.subplots(figsize=(8, 6))
pred_counts = pd.Series(y_pred).value_counts().sort_index()
bars = ax4.bar(pred_counts.index, pred_counts.values, color=colors[:len(pred_counts)], edgecolor='black', linewidth=1.5)
ax4.set_title('预测标签分布 (Predicted Label Distribution)', fontsize=16, fontweight='bold')
ax4.set_xlabel('类别 (Class)', fontsize=13)
ax4.set_ylabel('样本数量 (Count)', fontsize=13)
for bar, count in zip(bars, pred_counts.values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('4_pred_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图4已保存: 4_pred_distribution.png")

# 图5: 归一化混淆矩阵
fig5, ax5 = plt.subplots(figsize=(8, 6))
cm_normalized = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=classes, yticklabels=classes, ax=ax5,
            annot_kws={'size': 13})
ax5.set_title('归一化混淆矩阵 (Normalized CM)', fontsize=16, fontweight='bold')
ax5.set_xlabel('预测标签 (Predicted Label)', fontsize=13)
ax5.set_ylabel('真实标签 (True Label)', fontsize=13)
plt.tight_layout()
plt.savefig('5_normalized_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图5已保存: 5_normalized_confusion_matrix.png")

# 图6: 评估指标条形图
fig6, ax6 = plt.subplots(figsize=(8, 6))
metrics = ['准确率', '精确率', '召回率', 'F1分数']
values = [accuracy, precision, recall, f1]
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax6.bar(metrics, values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax6.set_title('模型评估指标', fontsize=16, fontweight='bold')
ax6.set_ylabel('分数 (Score)', fontsize=13)
ax6.set_ylim([0, 1.1])
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('6_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 图6已保存: 6_metrics.png")

# ==================== 生成汇总图 (6张图合在一起) ====================
fig = plt.figure(figsize=(20, 14))
fig.suptitle('心电图(ECG)分类模型完整评估报告', fontsize=20, fontweight='bold', y=0.98)

# 1. 混淆矩阵
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, ax=ax1,
            annot_kws={'size': 11})
ax1.set_title('混淆矩阵', fontsize=14, fontweight='bold')
ax1.set_xlabel('预测标签', fontsize=11)
ax1.set_ylabel('真实标签', fontsize=11)

# 2. 各类别准确率
ax2 = plt.subplot(2, 3, 2)
class_accuracy = np.nan_to_num(cm.diagonal() / cm.sum(axis=1))
bars = ax2.bar(classes, class_accuracy, color='skyblue', edgecolor='black', linewidth=1)
ax2.set_title('各类别准确率', fontsize=14, fontweight='bold')
ax2.set_xlabel('类别', fontsize=11)
ax2.set_ylabel('准确率', fontsize=11)
ax2.set_ylim([0, 1.1])
ax2.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'总体={accuracy:.3f}')
ax2.legend()
for bar, acc in zip(bars, class_accuracy):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 3. 真实标签分布
ax3 = plt.subplot(2, 3, 3)
bars = ax3.bar(true_counts.index, true_counts.values, color=colors[:len(true_counts)], edgecolor='black', linewidth=1)
ax3.set_title('真实标签分布', fontsize=14, fontweight='bold')
ax3.set_xlabel('类别', fontsize=11)
ax3.set_ylabel('数量', fontsize=11)
for bar, count in zip(bars, true_counts.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom', fontsize=10)

# 4. 预测标签分布
ax4 = plt.subplot(2, 3, 4)
bars = ax4.bar(pred_counts.index, pred_counts.values, color=colors[:len(pred_counts)], edgecolor='black', linewidth=1)
ax4.set_title('预测标签分布', fontsize=14, fontweight='bold')
ax4.set_xlabel('类别', fontsize=11)
ax4.set_ylabel('数量', fontsize=11)
for bar, count in zip(bars, pred_counts.values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom', fontsize=10)

# 5. 归一化混淆矩阵
ax5 = plt.subplot(2, 3, 5)
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=classes, yticklabels=classes, ax=ax5,
            annot_kws={'size': 10})
ax5.set_title('归一化混淆矩阵', fontsize=14, fontweight='bold')
ax5.set_xlabel('预测标签', fontsize=11)
ax5.set_ylabel('真实标签', fontsize=11)

# 6. 评估指标
ax6 = plt.subplot(2, 3, 6)
bars = ax6.bar(metrics, values, color=colors_bar, edgecolor='black', linewidth=1)
ax6.set_title('评估指标', fontsize=14, fontweight='bold')
ax6.set_ylabel('分数', fontsize=11)
ax6.set_ylim([0, 1.1])
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('0_summary_report.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 汇总图已保存: 0_summary_report.png")

# ==================== 错误分析 ====================
print("\n" + "="*50)
print("错误预测分析:")
print("="*50)
merged_df['correct'] = merged_df['label'] == merged_df['pred_label']
error_df = merged_df[~merged_df['correct']]

if len(error_df) > 0:
    print(f"错误预测样本数量: {len(error_df)}")
    print(f"错误率: {len(error_df)/len(merged_df):.4f}")
    print("\n错误预测样本示例 (前10个):")
    print(error_df[['id', 'label', 'pred_label']].head(10))
    
    error_df['error_type'] = error_df.apply(lambda x: f"{int(x['label'])}→{int(x['pred_label'])}", axis=1)
    error_counts = error_df['error_type'].value_counts()
    print("\n错误类型分布:")
    print(error_counts)
    
    # 单独生成错误分析图
    fig7, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig7.suptitle('错误预测分析', fontsize=16, fontweight='bold')
    
    # 错误类型分布
    error_counts.head(10).plot(kind='bar', ax=ax1, color='coral', edgecolor='black', linewidth=1.5)
    ax1.set_title('错误类型分布 (Top 10)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('真实→预测', fontsize=12)
    ax1.set_ylabel('错误数量', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 各类别错误率
    class_error_rate = []
    for c in classes:
        total = sum(y_true == c)
        error = sum((y_true == c) & (y_pred != c))
        class_error_rate.append(error/total if total > 0 else 0)
    
    bars = ax2.bar(classes, class_error_rate, color='lightblue', edgecolor='black', linewidth=1.5)
    ax2.set_title('各类别错误率', fontsize=14, fontweight='bold')
    ax2.set_xlabel('类别', fontsize=12)
    ax2.set_ylabel('错误率', fontsize=12)
    ax2.set_ylim([0, 1])
    for bar, rate in zip(bars, class_error_rate):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('7_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图7已保存: 7_error_analysis.png")
else:
    print("🎉 所有样本预测正确！")

# ==================== 保存结果 ====================
merged_df.to_csv('evaluation_results.csv', index=False)
print(f"\n✓ 详细结果已保存为: evaluation_results.csv")

print("\n" + "="*50)
print("统计摘要:")
print("="*50)
print(f"总样本数: {len(merged_df)}")
print(f"类别数量: {len(classes)}")
print(f"正确预测数: {merged_df['correct'].sum()}")
print(f"错误预测数: {len(merged_df) - merged_df['correct'].sum()}")
print(f"总体准确率: {accuracy:.4f}")
print("="*50)

print("\n✅ 评估完成！生成的文件列表:")
print("  📊 0_summary_report.png - 汇总报告 (6图合一)")
print("  📊 1_confusion_matrix.png - 混淆矩阵")
print("  📊 2_class_accuracy.png - 各类别准确率")
print("  📊 3_true_distribution.png - 真实标签分布")
print("  📊 4_pred_distribution.png - 预测标签分布")
print("  📊 5_normalized_confusion_matrix.png - 归一化混淆矩阵")
print("  📊 6_metrics.png - 评估指标")
if len(error_df) > 0:
    print("  📊 7_error_analysis.png - 错误分析")
print("  📄 evaluation_results.csv - 详细结果数据")