import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import datetime

# ==================== 1. 定义模型结构（必须和训练时一样）====================
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout

class Net1(Model):
    """模型1：基础CNN模型（必须和训练时定义完全一致）"""
    def __init__(self, input_dim=205):
        super(Net1, self).__init__()
        self.conv1 = Conv1D(16, 3, padding='same', activation='relu', input_shape=(input_dim, 1))
        self.conv2 = Conv1D(32, 3, dilation_rate=2, padding='same', activation='relu')
        self.conv3 = Conv1D(64, 3, dilation_rate=2, padding='same', activation='relu')
        self.conv4 = Conv1D(64, 5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool1 = MaxPool1D(3, strides=2, padding='same')
        self.conv5 = Conv1D(128, 5, dilation_rate=2, padding='same', activation='relu')
        self.conv6 = Conv1D(128, 5, dilation_rate=2, padding='same', activation='relu')
        self.max_pool2 = MaxPool1D(3, strides=2, padding='same')
        self.dropout = Dropout(0.5)
        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc21 = Dense(16, activation='relu')
        self.fc22 = Dense(256, activation='sigmoid')
        self.fc3 = Dense(4, activation='softmax')
            
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = self.conv5(x)
        x = self.conv6(x) 
        x = self.max_pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x1 = self.fc1(x)
        x2 = self.fc22(self.fc21(x))
        return self.fc3(x1 + x2)

# ==================== 2. 加载测试数据 ====================
print("加载测试数据...")
test = pd.read_csv('./TestSet.csv')

# 保存原始id
original_ids = test['id'].values

# 解析测试集
test_list = []
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])
test_df = pd.DataFrame(np.array(test_list))
feature_dim = len(test_list[0]) - 1
test_df.columns = ['id'] + [f's_{i}' for i in range(feature_dim)]

print(f"测试集形状: {test_df.shape}")
print(f"特征维度: {feature_dim}")

# ==================== 3. 数据预处理 ====================
print("数据预处理...")
X_test = test_df.drop(['id'], axis=1)

# 标准化（需要和训练时一样的scaler）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# 重塑为CNN输入格式
X_test_cnn = X_test_scaled.reshape(-1, feature_dim, 1).astype(np.float32)
print(f"测试集输入形状: {X_test_cnn.shape}")

# ==================== 4. 加载模型权重 ====================
print("加载Net1权重...")
model = Net1(input_dim=feature_dim)

# 构建模型
dummy = tf.random.normal((1, feature_dim, 1))
_ = model(dummy)

# 加载权重
model.load_weights('./Net1_weights.h5')
print("权重加载成功！")

# ==================== 5. 预测 ====================
print("开始预测...")
predictions = model.predict(X_test_cnn, verbose=1)
print(f"预测完成，结果形状: {predictions.shape}")

# ==================== 6. 生成提交文件 ====================
print("生成提交文件...")
submit = pd.DataFrame()
submit['id'] = original_ids  # 使用原始id，不改
submit['label_0'] = predictions[:, 0]
submit['label_1'] = predictions[:, 1]
submit['label_2'] = predictions[:, 2]
submit['label_3'] = predictions[:, 3]

# 后处理：将概率转换为0/1标签（可选）
threshold = 0.5
for idx, row in submit.iterrows():
    max_prob = max(row[1:])
    if max_prob > threshold:
        for i in range(1, 5):
            submit.iloc[idx, i] = 1 if row[i] > threshold else 0

# 保存结果
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'./Net1_predict_{timestamp}.csv'
submit.to_csv(output_file, index=False)

print(f"\n✅ 预测完成！结果保存到: {output_file}")
print(f"预测结果前5行:")
print(submit.head())
print(f"\nid范围: {submit['id'].min()} - {submit['id'].max()}")