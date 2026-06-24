import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("心电图(ECG)信号可视化 - 每类选取1个样本，显示前5个数据点")
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
        values = signal_str.split(',')
        return np.array([float(x) for x in values if x.strip()])
    else:
        return np.array([])

merged_df['signal_array'] = merged_df['heartbeat_signals'].apply(parse_signal)

# 显示信号长度信息
merged_df['signal_length'] = merged_df['signal_array'].apply(len)
print(f"\n信号长度统计:")
print(f"  最小长度: {merged_df['signal_length'].min()}")
print(f"  最大长度: {merged_df['signal_length'].max()}")
print(f"  平均长度: {merged_df['signal_length'].mean():.1f}")

# 4. 按类别分组，每个类别选择第一个样本
categories = sorted(merged_df['label'].unique())
print(f"\n类别分布: {categories}")

selected_samples = {}
for cat in categories:
    cat_data = merged_df[merged_df['label'] == cat]
    # 选择该类别的第一个样本
    first_sample = cat_data.iloc[0]
    selected_samples[cat] = first_sample
    print(f"  类别 {int(cat)}: 选择 ID={first_sample['id']}, 信号长度={first_sample['signal_length']}")

# 5. 创建可视化 - 每个类别选1个样本，显示前5个数据点
fig, ax = plt.subplots(figsize=(12, 7))

# 定义颜色和标记
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
markers = ['o', 's', '^', 'D']
labels = ['类别 0', '类别 1', '类别 2', '类别 3']

# 显示前N个数据点
n_points = 5

for idx, cat in enumerate(categories):
    sample = selected_samples[cat]
    signal = sample['signal_array']
    sample_id = sample['id']
    
    # 取前5个数据点
    signal_first_5 = signal[:n_points]
    
    # 创建x轴坐标 (0, 1, 2, 3, 4)
    x_values = np.arange(len(signal_first_5))
    
    # 绘制折线图和散点图
    ax.plot(x_values, signal_first_5, 
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            markersize=10,
            linewidth=2.5,
            label=f'类别 {int(cat)} (ID:{sample_id})',
            alpha=0.8)
    
    # 在每个数据点上显示数值
    for i, (x, y) in enumerate(zip(x_values, signal_first_5)):
        ax.annotate(f'{y:.3f}', 
                   (x, y),
                   textcoords="offset points",
                   xytext=(0, 10),
                   ha='center',
                   fontsize=8,
                   color=colors[idx % len(colors)])
    
    # 打印详细信息
    print(f"\n类别 {int(cat)} (ID:{sample_id}) 前5个数据点:")
    for i, val in enumerate(signal_first_5):
        print(f"  第{i+1}个点: {val:.6f}")

# 设置图表属性
ax.set_title('心电图(ECG)信号 - 每类选取1个样本(显示前5个数据点)', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('采样点索引 (Sample Point Index)', fontsize=13)
ax.set_ylabel('信号幅值 (Amplitude)', fontsize=13)
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xticks(np.arange(n_points))
ax.set_xticklabels([f'点{i+1}' for i in range(n_points)])

# 添加背景色
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('ecg_first_5_points_by_class.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ 图表已保存为: ecg_first_5_points_by_class.png")

# ==================== 额外：同时显示完整信号对比 ====================
print("\n" + "="*60)
print("生成完整信号对比图 (作为补充)")
print("="*60)

fig2, ax2 = plt.subplots(figsize=(14, 6))

for idx, cat in enumerate(categories):
    sample = selected_samples[cat]
    signal = sample['signal_array']
    sample_id = sample['id']
    
    # 绘制完整信号
    x_values_full = np.arange(len(signal))
    ax2.plot(x_values_full, signal, 
             color=colors[idx % len(colors)],
             linewidth=1.5,
             label=f'类别 {int(cat)} (ID:{sample_id})',
             alpha=0.7)
    
    # 标记前5个点
    ax2.scatter(x_values_full[:5], signal[:5], 
               color=colors[idx % len(colors)],
               s=50,
               zorder=5,
               marker=markers[idx % len(markers)])

ax2.set_title('心电图(ECG)完整信号对比 (红色圆圈标记前5个数据点)', 
             fontsize=16, fontweight='bold')
ax2.set_xlabel('采样点索引 (Sample Point Index)', fontsize=13)
ax2.set_ylabel('信号幅值 (Amplitude)', fontsize=13)
ax2.legend(loc='best', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('ecg_full_signals_by_class.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 完整信号对比图已保存为: ecg_full_signals_by_class.png")

# ==================== 统计摘要 ====================
print("\n" + "="*60)
print("统计摘要")
print("="*60)
print(f"总样本数: {len(merged_df)}")
print(f"类别数量: {len(categories)}")
print("\n选取的样本:")
for cat in categories:
    sample = selected_samples[cat]
    print(f"  类别 {int(cat)}: ID={sample['id']}, 信号长度={sample['signal_length']}")
print("\n显示的是每个样本的前5个数据点")
print("\n✅ 可视化完成！生成的文件:")
print("  📊 ecg_first_5_points_by_class.png - 每类1个样本，显示前5个数据点")
print("  📊 ecg_full_signals_by_class.png - 完整信号对比 (标记前5个点)")

# ==================== 导出选中的数据 ====================
selected_data = []
for cat in categories:
    sample = selected_samples[cat]
    signal_first_5 = sample['signal_array'][:5]
    selected_data.append({
        '类别': int(cat),
        'ID': sample['id'],
        '点1': signal_first_5[0] if len(signal_first_5) > 0 else None,
        '点2': signal_first_5[1] if len(signal_first_5) > 1 else None,
        '点3': signal_first_5[2] if len(signal_first_5) > 2 else None,
        '点4': signal_first_5[3] if len(signal_first_5) > 3 else None,
        '点5': signal_first_5[4] if len(signal_first_5) > 4 else None,
    })

selected_df = pd.DataFrame(selected_data)
selected_df.to_csv('selected_samples_first_5_points.csv', index=False)
print("  📄 selected_samples_first_5_points.csv - 选中的样本数据")