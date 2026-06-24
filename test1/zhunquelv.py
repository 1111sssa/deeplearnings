import pandas as pd
import numpy as np

# 加载数据
true_labels = pd.read_csv('./111.csv')
predictions = pd.read_csv('./Net1_predict_20260602_165518.csv')

# 合并
merged = predictions.merge(true_labels, on='id')

# 计算预测类别
pred_classes = merged[['label_0', 'label_1', 'label_2', 'label_3']].values.argmax(axis=1)
true_classes = merged['label'].values

# 准确率
accuracy = (pred_classes == true_classes).mean()
print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")