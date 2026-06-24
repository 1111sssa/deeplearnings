import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取数据（注意分隔符是 \t）
file_path = "train_set.csv"  # 修改为你的实际路径
df = pd.read_csv(file_path, sep='\t')

# 2. 查看数据基本情况
print(f"原始数据总量: {len(df)}")
print(df.head())

# 3. 划分训练集和测试集 (8:2)
train_df, test_df = train_test_split(
    df, 
    test_size=0.2,          # 20% 作为测试集
    random_state=42,        # 固定随机种子，保证可重复
    stratify=df['label']    # 按标签分层采样（推荐，保持类别分布一致）
)

# 4. 保存文件
train_df.to_csv("TrainSet.csv", sep='\t', index=False)
test_df.to_csv("TestSet.csv", sep='\t', index=False)

# 5. 输出结果
print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")