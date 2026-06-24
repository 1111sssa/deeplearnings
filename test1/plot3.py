import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("心电图(ECG)信号可视化 - 按类别展示")
print("="*60)

# 1. 读取数据
try:
    # 读取真实标签 (111.csv)
    labels_df = pd.read_csv('111.csv')
    print(f"✓ 成功加载标签文件: 111.csv (共{len(labels_df)}条记录)")
    
    # 读取测试集信号 (TestSet.csv)
    testset_df = pd.read_csv('TestSet.csv')
    print(f"✓ 成功加载信号文件: TestSet.csv (共{len(testset_df)}条记录)")
except FileNotFoundError as e:
    print(f"✗ 文件加载失败: {e}")
    print("请确保 111.csv 和 TestSet.csv 在当前目录下")
    exit()

print("\n标签数据前5行:")
print(labels_df.head())
print("\n信号数据前5行:")
print(testset_df.head())

# 2. 合并数据
merged_df = pd.merge(testset_df, labels_df, on='id', how='inner')
print(f"\n✓ 数据合并完成: 共{len(merged_df)}个样本")

# 3. 解析心电信号
def parse_signal(signal_str):
    """将字符串格式的信号转换为numpy数组"""
    if isinstance(signal_str, str):
        # 分割字符串并转换为浮点数
        values = signal_str.split(',')
        return np.array([float(x) for x in values if x.strip()])
    else:
        return np.array([])

# 应用解析函数
merged_df['signal_array'] = merged_df['heartbeat_signals'].apply(parse_signal)

# 检查信号长度
signal_lengths = merged_df['signal_array'].apply(len)
print(f"\n信号长度统计:")
print(f"  最小长度: {signal_lengths.min()}")
print(f"  最大长度: {signal_lengths.max()}")
print(f"  平均长度: {signal_lengths.mean():.1f}")
print(f"  所有信号长度一致: {signal_lengths.nunique() == 1}")

# 获取信号的统一长度
signal_len = signal_lengths.mode()[0] if signal_lengths.nunique() > 1 else signal_lengths.iloc[0]
print(f"  使用信号长度: {signal_len}")

# 4. 按类别分组
categories = sorted(merged_df['label'].unique())
print(f"\n类别分布:")
for cat in categories:
    count = len(merged_df[merged_df['label'] == cat])
    print(f"  类别 {cat}: {count} 个样本")

# 5. 为每个类别选择5个样本
samples_per_class = 5
selected_samples = {}

for cat in categories:
    cat_data = merged_df[merged_df['label'] == cat]
    n_samples = min(samples_per_class, len(cat_data))
    # 选择前n_samples个样本（也可以随机选择）
    selected_samples[cat] = cat_data.head(n_samples)
    print(f"\n类别 {cat}: 选择了 {n_samples} 个样本")

# 6. 创建可视化 - 每个类别5个信号在同一张图上
n_categories = len(categories)
fig, axes = plt.subplots(n_categories, 1, figsize=(15, 4 * n_categories))
fig.suptitle('心电图(ECG)信号分类展示 - 每类5个样本', fontsize=18, fontweight='bold', y=0.98)

# 如果只有一个类别，axes不是数组，需要处理
if n_categories == 1:
    axes = [axes]

# 颜色方案
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', 
          '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471']

for idx, cat in enumerate(categories):
    ax = axes[idx]
    cat_samples = selected_samples[cat]
    
    # 绘制该类别下每个样本的信号
    for i, (_, row) in enumerate(cat_samples.iterrows()):
        signal = row['signal_array']
        # 如果信号长度不一致，需要截断或填充
        if len(signal) != signal_len:
            if len(signal) > signal_len:
                signal = signal[:signal_len]
            else:
                signal = np.pad(signal, (0, signal_len - len(signal)), 'constant', constant_values=0)
        
        # 绘制信号，使用不同颜色和透明度
        ax.plot(signal, color=colors[i % len(colors)], alpha=0.8, 
                linewidth=1.5, label=f'样本 {i+1} (ID:{row["id"]})')
    
    # 设置图表属性
    ax.set_title(f'类别 {int(cat)} - {len(cat_samples)}个样本', fontsize=14, fontweight='bold')
    ax.set_xlabel('采样点 (Sample Point)', fontsize=11)
    ax.set_ylabel('信号幅值 (Amplitude)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax.text(0.02, 0.98, f'样本数: {len(cat_samples)}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('ecg_signals_by_class.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ 图表已保存为: ecg_signals_by_class.png")

# ==================== 额外：在网格中展示所有信号 ====================
print("\n" + "="*60)
print("生成网格视图 - 每个类别单独一行的详细展示")
print("="*60)

# 创建更详细的网格图：每行一个类别，每列一个样本
fig2, axes2 = plt.subplots(n_categories, samples_per_class, 
                           figsize=(4 * samples_per_class, 3 * n_categories))
fig2.suptitle('心电图(ECG)信号网格展示 - 每行一个类别', fontsize=16, fontweight='bold')

# 处理单行或单列的情况
if n_categories == 1:
    axes2 = axes2.reshape(1, -1)
if samples_per_class == 1:
    axes2 = axes2.reshape(-1, 1)

for i, cat in enumerate(categories):
    cat_samples = selected_samples[cat]
    for j in range(samples_per_class):
        ax = axes2[i, j]
        if j < len(cat_samples):
            row = cat_samples.iloc[j]
            signal = row['signal_array']
            
            # 处理信号长度
            if len(signal) != signal_len:
                if len(signal) > signal_len:
                    signal = signal[:signal_len]
                else:
                    signal = np.pad(signal, (0, signal_len - len(signal)), 'constant', constant_values=0)
            
            ax.plot(signal, color='steelblue', linewidth=1.5)
            ax.set_title(f'ID:{row["id"]}', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            # 如果没有足够的样本，显示空白
            ax.set_visible(False)
        
        if j == 0:
            ax.set_ylabel(f'类别 {int(cat)}', fontsize=11, fontweight='bold')
        if i == n_categories - 1:
            ax.set_xlabel('采样点', fontsize=9)

plt.tight_layout()
plt.savefig('ecg_signals_grid.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 网格图已保存为: ecg_signals_grid.png")

# ==================== 额外：统计信息 ====================
print("\n" + "="*60)
print("数据统计摘要")
print("="*60)
print(f"总样本数: {len(merged_df)}")
print(f"类别数量: {len(categories)}")
print(f"信号长度: {signal_len} 个采样点")
print("\n每个类别的样本数量:")
for cat in categories:
    count = len(merged_df[merged_df['label'] == cat])
    print(f"  类别 {int(cat)}: {count} 个样本")

print("\n✅ 可视化完成！生成的文件:")
print("  📊 ecg_signals_by_class.png - 每类5个信号叠加展示")
print("  📊 ecg_signals_grid.png - 网格展示 (每行一个类别)")

# ==================== 可选：保存预处理后的数据 ====================
# 将信号数组转换为DataFrame并保存
# merged_df_export = merged_df.copy()
# merged_df_export['signal_length'] = merged_df_export['signal_array'].apply(len)
# merged_df_export.to_csv('processed_ecg_data.csv', index=False)
# print("  📄 processed_ecg_data.csv - 处理后的数据")