import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("训练集心电图(ECG)信号可视化 - 每类选取1个样本，显示前5个数据点")
print("="*60)

# 1. 读取训练集数据
try:
    # 读取训练集CSV (假设文件名为 Trainset.csv 或 train.csv)
    # 根据您的截图，文件名可能是 "Trainsetcsv" 但实际应该是 "Trainset.csv"
    import os
    
    # 尝试可能的文件名
    possible_names = ['Trainset.csv', 'trainset.csv', 'Train.csv', 'train.csv', 'Trainsetcsv']
    train_file = None
    
    for name in possible_names:
        if os.path.exists(name):
            train_file = name
            break
    
    if train_file is None:
        # 如果都没有，列出当前目录下的csv文件
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        print(f"当前目录下的CSV文件: {csv_files}")
        # 尝试使用第一个找到的CSV文件
        if csv_files:
            train_file = csv_files[0]
            print(f"使用文件: {train_file}")
        else:
            raise FileNotFoundError("未找到任何CSV文件")
    
    train_df = pd.read_csv(train_file)
    print(f"✓ 成功加载训练集文件: {train_file} (共{len(train_df)}条记录)")
    
except FileNotFoundError as e:
    print(f"✗ 文件加载失败: {e}")
    print("请确保训练集CSV文件在当前目录下")
    print("提示: 文件名可能是 Trainset.csv, trainset.csv, Train.csv 或 train.csv")
    exit()

print("\n训练集数据前5行:")
print(train_df.head())
print("\n训练集数据列名:")
print(train_df.columns.tolist())
print(f"\n数据形状: {train_df.shape}")

# 2. 识别列名
# 尝试识别标签列和信号列
label_col = None
signal_col = None

for col in train_df.columns:
    col_lower = col.lower()
    if 'label' in col_lower or 'class' in col_lower or 'type' in col_lower:
        label_col = col
    if 'signal' in col_lower or 'heartbeat' in col_lower or 'ecg' in col_lower:
        signal_col = col

# 如果没找到明确的列名，使用默认猜测
if label_col is None:
    # 可能是第一列或最后一列
    if train_df.shape[1] == 2:
        label_col = train_df.columns[1]  # 假设第二列是标签
    else:
        label_col = train_df.columns[-1]  # 假设最后一列是标签
        signal_col = train_df.columns[0]  # 假设第一列是信号

if signal_col is None:
    # 如果signal_col还是None，检查列的数据类型
    for col in train_df.columns:
        if col != label_col and train_df[col].dtype == 'object':
            # 如果是字符串类型，可能是信号
            signal_col = col
            break
    if signal_col is None:
        signal_col = train_df.columns[0]  # 默认第一列

print(f"\n识别到的列:")
print(f"  标签列: {label_col}")
print(f"  信号列: {signal_col}")

# 3. 解析心电信号
def parse_signal(signal_str):
    """将字符串格式的信号转换为numpy数组"""
    if isinstance(signal_str, str):
        # 去除可能的中文字符或多余空格
        signal_str = signal_str.strip()
        # 分割字符串并转换为浮点数
        values = signal_str.split(',')
        return np.array([float(x) for x in values if x.strip()])
    else:
        return np.array([])

train_df['signal_array'] = train_df[signal_col].apply(parse_signal)

# 显示信号长度信息
train_df['signal_length'] = train_df['signal_array'].apply(len)
print(f"\n信号长度统计:")
print(f"  最小长度: {train_df['signal_length'].min()}")
print(f"  最大长度: {train_df['signal_length'].max()}")
print(f"  平均长度: {train_df['signal_length'].mean():.1f}")

# 4. 按标签分组，每个标签选择第一个样本
categories = sorted(train_df[label_col].unique())
print(f"\n标签类别分布: {categories}")

# 显示每个类别的样本数量
print("\n每个类别的样本数量:")
for cat in categories:
    count = len(train_df[train_df[label_col] == cat])
    print(f"  类别 {int(cat)}: {count} 个样本")

selected_samples = {}
for cat in categories:
    cat_data = train_df[train_df[label_col] == cat]
    # 选择该类别的第一个样本
    first_sample = cat_data.iloc[0]
    selected_samples[cat] = first_sample
    print(f"\n类别 {int(cat)}: 选择 索引={first_sample.name}, 信号长度={first_sample['signal_length']}")

# 5. 创建可视化 - 每个类别选1个样本，显示前5个数据点
fig, ax = plt.subplots(figsize=(12, 7))

# 定义颜色和标记
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
n_points = 5

print("\n" + "="*60)
print("每个类别前5个数据点详情:")
print("="*60)

for idx, cat in enumerate(categories):
    sample = selected_samples[cat]
    signal = sample['signal_array']
    
    # 取前5个数据点
    signal_first_5 = signal[:n_points]
    
    # 创建x轴坐标 (0, 1, 2, 3, 4)
    x_values = np.arange(len(signal_first_5))
    
    # 绘制折线图和散点图
    color_idx = idx % len(colors)
    marker_idx = idx % len(markers)
    
    ax.plot(x_values, signal_first_5, 
            color=colors[color_idx],
            marker=markers[marker_idx],
            markersize=12,
            linewidth=2.5,
            label=f'类别 {int(cat)}',
            alpha=0.8)
    
    # 在每个数据点上显示数值
    for i, (x, y) in enumerate(zip(x_values, signal_first_5)):
        ax.annotate(f'{y:.3f}', 
                   (x, y),
                   textcoords="offset points",
                   xytext=(0, 12),
                   ha='center',
                   fontsize=9,
                   color=colors[color_idx],
                   fontweight='bold')
    
    # 打印详细信息
    print(f"\n类别 {int(cat)} (索引:{sample.name}) 前5个数据点:")
    for i, val in enumerate(signal_first_5):
        print(f"  第{i+1}个点: {val:.6f}")

# 设置图表属性
ax.set_title('训练集 - 心电图(ECG)信号 (每类选取1个样本，显示前5个数据点)', 
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
plt.savefig('train_first_5_points_by_class.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n✓ 图表已保存为: train_first_5_points_by_class.png")

# ==================== 额外：同时显示完整信号对比 ====================
print("\n" + "="*60)
print("生成完整信号对比图 (作为补充)")
print("="*60)

fig2, ax2 = plt.subplots(figsize=(14, 6))

for idx, cat in enumerate(categories):
    sample = selected_samples[cat]
    signal = sample['signal_array']
    
    color_idx = idx % len(colors)
    marker_idx = idx % len(markers)
    
    # 绘制完整信号
    x_values_full = np.arange(len(signal))
    ax2.plot(x_values_full, signal, 
             color=colors[color_idx],
             linewidth=1.5,
             label=f'类别 {int(cat)}',
             alpha=0.7)
    
    # 标记前5个点
    ax2.scatter(x_values_full[:5], signal[:5], 
               color=colors[color_idx],
               s=80,
               zorder=5,
               marker=markers[marker_idx],
               edgecolors='black',
               linewidth=0.5)

ax2.set_title('训练集 - 心电图(ECG)完整信号对比 (标记前5个数据点)', 
             fontsize=16, fontweight='bold')
ax2.set_xlabel('采样点索引 (Sample Point Index)', fontsize=13)
ax2.set_ylabel('信号幅值 (Amplitude)', fontsize=13)
ax2.legend(loc='best', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('train_full_signals_by_class.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ 完整信号对比图已保存为: train_full_signals_by_class.png")

# ==================== 统计摘要 ====================
print("\n" + "="*60)
print("统计摘要")
print("="*60)
print(f"训练集总样本数: {len(train_df)}")
print(f"类别数量: {len(categories)}")
print(f"\n类别分布:")
for cat in categories:
    count = len(train_df[train_df[label_col] == cat])
    print(f"  类别 {int(cat)}: {count} 个样本 ({count/len(train_df)*100:.1f}%)")

print("\n选取的样本:")
for cat in categories:
    sample = selected_samples[cat]
    print(f"  类别 {int(cat)}: 索引={sample.name}, 信号长度={sample['signal_length']}")

print("\n显示的是每个样本的前5个数据点")
print("\n✅ 可视化完成！生成的文件:")
print("  📊 train_first_5_points_by_class.png - 每类1个样本，显示前5个数据点")
print("  📊 train_full_signals_by_class.png - 完整信号对比 (标记前5个点)")

# ==================== 导出选中的数据 ====================
selected_data = []
for cat in categories:
    sample = selected_samples[cat]
    signal_first_5 = sample['signal_array'][:5]
    row_data = {
        '类别': int(cat),
        '索引': sample.name,
        '点1': signal_first_5[0] if len(signal_first_5) > 0 else None,
        '点2': signal_first_5[1] if len(signal_first_5) > 1 else None,
        '点3': signal_first_5[2] if len(signal_first_5) > 2 else None,
        '点4': signal_first_5[3] if len(signal_first_5) > 3 else None,
        '点5': signal_first_5[4] if len(signal_first_5) > 4 else None,
    }
    selected_data.append(row_data)

selected_df = pd.DataFrame(selected_data)
selected_df.to_csv('train_selected_samples_first_5_points.csv', index=False)
print("  📄 train_selected_samples_first_5_points.csv - 选中的样本数据")