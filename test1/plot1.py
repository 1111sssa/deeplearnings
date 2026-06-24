import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - 解决中文显示问题
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 支持中文
except:
    plt.rcParams['font.sans-serif'] = ['Arial']  # 备用字体
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("心电图(ECG)模型预测评估系统")
print("="*60)

# 1. 读取数据
try:
    # 真实标签 (111.csv)
    true_labels_df = pd.read_csv('111.csv')
    print(f"✓ 成功加载真实标签文件: 111.csv (共{len(true_labels_df)}条记录)")
    
    # 预测标签 (Net1_predict_20260602_165518.csv)
    pred_labels_df = pd.read_csv('Net1_predict_20260602_165518.csv')
    print(f"✓ 成功加载预测标签文件: Net1_predict_20260602_165518.csv (共{len(pred_labels_df)}条记录)")
except FileNotFoundError as e:
    print(f"✗ 文件加载失败: {e}")
    print("请确保111.csv和Net1_predict_20260602_165518.csv在当前目录下")
    exit()

print("\n" + "-"*40)
print("数据预览:")
print("-"*40)
print("\n真实标签 (前5行):")
print(true_labels_df.head())
print("\n预测标签 (前5行):")
print(pred_labels_df.head())

# 2. 数据预处理
# 将预测标签从one-hot格式转换为单标签格式
label_columns = ['label_0', 'label_1', 'label_2', 'label_3']
if all(col in pred_labels_df.columns for col in label_columns):
    pred_labels_df['pred_label'] = pred_labels_df[label_columns].idxmax(axis=1).str.replace('label_', '').astype(int)
    print(f"\n✓ 成功将one-hot编码转换为标签: {pred_labels_df['pred_label'].unique()}")
else:
    print("\n✗ 预测标签格式不正确，请检查列名")
    exit()

# 合并真实标签和预测标签
merged_df = pd.merge(true_labels_df, pred_labels_df[['id', 'pred_label']], on='id', how='inner')
print(f"\n✓ 数据合并完成: 共{len(merged_df)}个匹配样本")

if len(merged_df) == 0:
    print("\n✗ 没有匹配的样本，请检查ID是否一致")
    exit()

# 3. 计算评估指标
y_true = merged_df['label']
y_pred = merged_df['pred_label']
classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

# 计算各项指标
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print("\n" + "="*60)
print("模型评估指标")
print("="*60)
print(f"{'指标':<15} {'Macro平均':<15} {'Weighted平均':<15}")
print("-"*60)
print(f"{'准确率 (Accuracy)':<15} {accuracy:<15.4f} {accuracy:<15.4f}")
print(f"{'精确率 (Precision)':<15} {precision_macro:<15.4f} {precision_weighted:<15.4f}")
print(f"{'召回率 (Recall)':<15} {recall_macro:<15.4f} {recall_weighted:<15.4f}")
print(f"{'F1分数 (F1-Score)':<15} {f1_macro:<15.4f} {f1_weighted:<15.4f}")
print("="*60)

# 分类报告
print("\n分类报告 (Classification Report):")
print("-"*60)
print(classification_report(y_true, y_pred, digits=4))

# 4. 混淆矩阵
cm = confusion_matrix(y_true, y_pred)

print("\n混淆矩阵 (Confusion Matrix):")
print("-"*60)
print("真实\\预测  ", end="")
for c in classes:
    print(f" {c:>6}", end="")
print()
for i, c in enumerate(classes):
    print(f"类别 {c}     ", end="")
    for j in range(len(classes)):
        print(f" {cm[i][j]:>6}", end="")
    print()

# 5. 创建可视化图表
fig = plt.figure(figsize=(20, 12))
fig.suptitle('心电图(ECG)分类模型评估报告', fontsize=18, fontweight='bold', y=0.98)

# 5.1 混淆矩阵热力图
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, ax=ax1,
            annot_kws={'size': 12})
ax1.set_title('混淆矩阵 (Confusion Matrix)', fontsize=14, fontweight='bold')
ax1.set_xlabel('预测标签 (Predicted Label)', fontsize=12)
ax1.set_ylabel('真实标签 (True Label)', fontsize=12)

# 5.2 各类别准确率
ax2 = plt.subplot(2, 3, 2)
class_accuracy = cm.diagonal() / cm.sum(axis=1)
class_accuracy = np.nan_to_num(class_accuracy)
bars = ax2.bar(classes, class_accuracy, color=sns.color_palette("Blues_d", len(classes)), 
               edgecolor='black', linewidth=1.5)
ax2.set_title('各类别准确率 (Per-class Accuracy)', fontsize=14, fontweight='bold')
ax2.set_xlabel('类别 (Class)', fontsize=12)
ax2.set_ylabel('准确率 (Accuracy)', fontsize=12)
ax2.set_ylim([0, 1.1])
ax2.axhline(y=accuracy, color='red', linestyle='--', linewidth=2, label=f'总体准确率={accuracy:.3f}')
ax2.legend()
for bar, acc in zip(bars, class_accuracy):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

# 5.3 真实标签分布
ax3 = plt.subplot(2, 3, 3)
true_counts = pd.Series(y_true).value_counts().sort_index()
colors = sns.color_palette("Set3", len(true_counts))
bars = ax3.bar(true_counts.index, true_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_title('真实标签分布 (True Label Distribution)', fontsize=14, fontweight='bold')
ax3.set_xlabel('类别 (Class)', fontsize=12)
ax3.set_ylabel('样本数量 (Sample Count)', fontsize=12)
for bar, count in zip(bars, true_counts.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom', fontsize=10)

# 5.4 预测标签分布
ax4 = plt.subplot(2, 3, 4)
pred_counts = pd.Series(y_pred).value_counts().sort_index()
bars = ax4.bar(pred_counts.index, pred_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_title('预测标签分布 (Predicted Label Distribution)', fontsize=14, fontweight='bold')
ax4.set_xlabel('类别 (Class)', fontsize=12)
ax4.set_ylabel('样本数量 (Sample Count)', fontsize=12)
for bar, count in zip(bars, pred_counts.values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}', ha='center', va='bottom', fontsize=10)

# 5.5 归一化混淆矩阵
ax5 = plt.subplot(2, 3, 5)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=classes, yticklabels=classes, ax=ax5,
            annot_kws={'size': 11})
ax5.set_title('归一化混淆矩阵 (Normalized CM)', fontsize=14, fontweight='bold')
ax5.set_xlabel('预测标签 (Predicted Label)', fontsize=12)
ax5.set_ylabel('真实标签 (True Label)', fontsize=12)

# 5.6 评估指标对比图
ax6 = plt.subplot(2, 3, 6)
metrics = ['准确率', '精确率', '召回率', 'F1分数']
macro_values = [accuracy, precision_macro, recall_macro, f1_macro]
weighted_values = [accuracy, precision_weighted, recall_weighted, f1_weighted]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax6.bar(x - width/2, macro_values, width, label='Macro平均', 
                color='#4ECDC4', edgecolor='black', linewidth=1)
bars2 = ax6.bar(x + width/2, weighted_values, width, label='Weighted平均',
                color='#FF6B6B', edgecolor='black', linewidth=1)

ax6.set_title('评估指标对比', fontsize=14, fontweight='bold')
ax6.set_xlabel('评估指标', fontsize=12)
ax6.set_ylabel('分数 (Score)', fontsize=12)
ax6.set_ylim([0, 1.1])
ax6.set_xticks(x)
ax6.set_xticklabels(metrics)
ax6.legend(loc='upper right')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
print("\n✓ 评估图表已保存为: model_evaluation_results.png")
plt.show()

# 6. 错误分析
print("\n" + "="*60)
print("错误预测分析")
print("="*60)

merged_df['correct'] = merged_df['label'] == merged_df['pred_label']
error_df = merged_df[~merged_df['correct']]

if len(error_df) > 0:
    print(f"错误预测样本数量: {len(error_df)} (占比: {len(error_df)/len(merged_df):.2%})")
    print(f"正确预测样本数量: {merged_df['correct'].sum()} (占比: {merged_df['correct'].sum()/len(merged_df):.2%})")
    
    print("\n前10个错误预测样本:")
    print(error_df[['id', 'label', 'pred_label']].head(10))
    
    # 错误类型分布
    error_df['error_type'] = error_df.apply(lambda x: f"类别{x['label']}→{x['pred_label']}", axis=1)
    error_counts = error_df['error_type'].value_counts()
    
    print("\n错误类型分布 (Top 10):")
    for error_type, count in error_counts.head(10).items():
        print(f"  {error_type}: {count}次 ({count/len(error_df):.1%})")
    
    # 绘制错误类型图
    if len(error_counts) > 1:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('错误预测分析', fontsize=16, fontweight='bold')
        
        # 错误类型分布
        error_counts.head(10).plot(kind='bar', ax=ax1, color='coral', edgecolor='black', linewidth=1.5)
        ax1.set_title('错误类型分布', fontsize=13, fontweight='bold')
        ax1.set_xlabel('真实类别→预测类别', fontsize=11)
        ax1.set_ylabel('错误数量', fontsize=11)
        ax1.tick_params(axis='x', rotation=45)
        
        # 错误率按类别
        class_error_rate = []
        for c in classes:
            total = sum(y_true == c)
            error = sum((y_true == c) & (y_pred != c))
            class_error_rate.append(error/total if total > 0 else 0)
        
        bars = ax2.bar(classes, class_error_rate, color='lightblue', edgecolor='black', linewidth=1.5)
        ax2.set_title('各类别错误率', fontsize=13, fontweight='bold')
        ax2.set_xlabel('类别', fontsize=11)
        ax2.set_ylabel('错误率', fontsize=11)
        ax2.set_ylim([0, 1])
        for bar, rate in zip(bars, class_error_rate):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.2%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ 错误分析图已保存为: error_analysis.png")
        plt.show()
else:
    print("🎉 所有样本预测正确！模型表现完美！")

# 7. 保存详细结果
merged_df.to_csv('evaluation_detailed_results.csv', index=False)
print(f"\n✓ 详细结果已保存为: evaluation_detailed_results.csv")
print(f"  包含 {len(merged_df)} 个样本的完整对比信息")

# 8. 生成统计摘要
print("\n" + "="*60)
print("统计摘要")
print("="*60)
print(f"📊 总样本数: {len(merged_df)}")
print(f"📊 类别数量: {len(classes)}")
print(f"📊 类别分布: {dict(zip(classes, pd.Series(y_true).value_counts().sort_index().values))}")
print(f"✅ 正确预测: {merged_df['correct'].sum()} ({merged_df['correct'].sum()/len(merged_df):.2%})")
print(f"❌ 错误预测: {len(error_df)} ({len(error_df)/len(merged_df):.2%})")
print(f"🎯 总体准确率: {accuracy:.2%}")
print("="*60)

print("\n✅ 评估完成！请查看生成的图表和报告文件。")