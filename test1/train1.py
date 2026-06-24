"""
深度学习时间序列分类项目 - 简化版（仅训练Net1）
包含数据预处理、EDA可视化、单CNN模型训练、预测和后处理
"""

# ==================== 1. 导入必要的库 ====================
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

# 忽略TensorFlow提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==================== 2. 数据预处理函数 ====================

def reduce_mem_usage(df):
    """数据精度量化压缩，减少内存占用"""
    start_mem = df.memory_usage().sum() / 1024**2 
    print(f'优化前内存: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2 
    print(f'优化后内存: {end_mem:.2f} MB (减少 {100*(start_mem-end_mem)/start_mem:.1f}%)')
    return df


def handle_outliers(df, columns, method='iqr'):
    """处理异常值"""
    df_clean = df.copy()
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            df_clean.loc[z_scores > 3, col] = df_clean[col].median()
    return df_clean


# ==================== 3. 可视化函数 ====================
def plot_class_distribution(y_train, y_resampled=None):
    """绘制类别分布图"""
    # 判断需要几个子图
    n_plots = 2 if y_resampled is not None else 1
    
    fig, axes = plt.subplots(1, n_plots, figsize=(14, 5))
    
    # 重要：如果只有一个子图，将axes转换为列表，便于统一索引
    if n_plots == 1:
        axes = [axes]
    
    # 原始分布
    class_counts = y_train.value_counts().sort_index()
    axes[0].bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='black')
    axes[0].set_title('原始训练集类别分布', fontsize=14)
    axes[0].set_xlabel('类别', fontsize=12)
    axes[0].set_ylabel('样本数量', fontsize=12)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontsize=10)
    
    # SMOTE后分布
    if y_resampled is not None:
        resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
        axes[1].bar(resampled_counts.index, resampled_counts.values, color='lightcoral', edgecolor='black')
        axes[1].set_title('SMOTE处理后类别分布', fontsize=14)
        axes[1].set_xlabel('类别', fontsize=12)
        axes[1].set_ylabel('样本数量', fontsize=12)
        for i, v in enumerate(resampled_counts.values):
            axes[1].text(i, v + 5, str(v), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_distribution(X, n_features=20):
    """绘制特征分布图"""
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i in range(min(n_features, len(X.columns))):
        axes[i].hist(X.iloc[:, i], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[i].set_title(f'特征 {X.columns[i]}', fontsize=10)
        axes[i].set_xlabel('值', fontsize=8)
        axes[i].set_ylabel('频数', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('部分特征分布图', fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(X, top_k=30):
    """绘制特征相关性热力图"""
    # 计算相关性矩阵
    corr_matrix = X.corr()
    
    # 选择与目标相关性最高的top_k特征
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 完整热力图（采样）
    sampled_cols = np.random.choice(X.columns, min(50, len(X.columns)), replace=False)
    sns.heatmap(X[sampled_cols].corr(), ax=axes[0], cmap='RdYlBu_r', center=0, 
                annot=False, cbar_kws={'shrink': 0.8})
    axes[0].set_title('特征相关性热力图（部分特征）', fontsize=14)
    
    # Top k特征热力图
    if 'label' in X.columns:
        corr_with_target = corr_matrix['label'].abs().sort_values(ascending=False)
        top_features = corr_with_target.head(top_k).index
        sns.heatmap(X[top_features].corr(), ax=axes[1], cmap='RdYlBu_r', center=0,
                    annot=True, fmt='.2f', annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
        axes[1].set_title(f'Top{top_k}特征相关性热力图', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_sample_sequences(X, y, n_samples=5):
    """绘制样本序列图"""
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        axes[i].plot(X.iloc[idx, :50], 'b-', alpha=0.7, linewidth=1)  # 只显示前50个特征
        axes[i].set_title(f'样本 {idx} - 类别: {y.iloc[idx] if hasattr(y, "iloc") else y[idx]}', fontsize=12)
        axes[i].set_xlabel('特征索引', fontsize=10)
        axes[i].set_ylabel('特征值', fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('样本序列示例（前50个特征）', fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_sequences.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_boxplots(X, n_features=12):
    """绘制箱线图检测异常值"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    selected_features = np.random.choice(X.columns, min(n_features, len(X.columns)), replace=False)
    
    for i, col in enumerate(selected_features):
        axes[i].boxplot(X[col], patch_artist=True, 
                        boxprops=dict(facecolor='lightblue', color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))
        axes[i].set_title(f'特征 {col}', fontsize=10)
        axes[i].set_ylabel('值', fontsize=8)
        axes[i].grid(True, alpha=0.3, axis='y')
    
    for i in range(len(selected_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('特征箱线图（异常值检测）', fontsize=14)
    plt.tight_layout()
    plt.savefig('boxplots.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_visualization(X, y):
    """PCA降维可视化"""
    from sklearn.decomposition import PCA
    
    # PCA降维到2D和3D
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)
    
    X_pca_2d = pca_2d.fit_transform(X)
    X_pca_3d = pca_3d.fit_transform(X)
    
    print(f'PCA 2D解释方差比: {pca_2d.explained_variance_ratio_}')
    print(f'PCA 3D解释方差比: {pca_3d.explained_variance_ratio_}')
    
    # 2D可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter = axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', alpha=0.6, s=10)
    axes[0].set_title('PCA 2D可视化', fontsize=14)
    axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.colorbar(scatter, ax=axes[0], label='类别')
    
    # 3D可视化
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
                         c=y, cmap='viridis', alpha=0.6, s=10)
    ax.set_title('PCA 3D可视化', fontsize=14)
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})', fontsize=10)
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})', fontsize=10)
    plt.colorbar(scatter, ax=ax, label='类别')
    
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves_single(history, model_name):
    """单模型专用训练曲线绘图（2行1列布局）"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 损失曲线
    axes[0].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[0].set_title(f'{model_name} - 损失曲线', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[1].set_title(f'{model_name} - 准确率曲线', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} 模型训练曲线', fontsize=16)
    plt.tight_layout()
    plt.savefig('training_curves_single.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix_single(pred, y_true, model_name):
    """单模型混淆矩阵绘图"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    y_pred = np.argmax(pred, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'类别{j}' for j in range(4)],
                yticklabels=[f'类别{j}' for j in range(4)])
    ax.set_title(f'{model_name} - 混淆矩阵', fontsize=14)
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('真实类别', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_single.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_prediction_distribution(predictions):
    """绘制预测概率分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(4):
        axes[i].hist(predictions[:, i], bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
        axes[i].set_title(f'类别 {i} 预测概率分布', fontsize=12)
        axes[i].set_xlabel('预测概率', fontsize=10)
        axes[i].set_ylabel('样本数量', fontsize=10)
        axes[i].axvline(x=0.5, color='red', linestyle='--', label='阈值0.5')
        axes[i].legend()
    
    plt.suptitle('Net1模型预测概率分布', fontsize=14)
    plt.tight_layout()
    plt.savefig('prediction_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


# ==================== 4. 仅保留Net1模型 ====================
class Net1(Model):
    """模型1：基础CNN模型"""
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


# ==================== 5. 训练函数 ====================
def train_model(model, model_name, x_train, y_train, x_val, y_val, epochs=50, batch_size=64):
    """训练单个模型"""
    print(f"\n{'='*50}")
    print(f"开始训练 {model_name}")
    print(f"{'='*50}")
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
    ]
    
    # 训练
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存模型
    model.save_weights(f'./{model_name}_weights.h5')
    print(f"{model_name} 训练完成，权重已保存")
    
    return history


# ==================== 6. 主程序 ====================
def main():
    print("="*60)
    print("深度学习时间序列分类项目 - 仅Net1训练简化版")
    print("="*60)
    
    # ---------- 1. 加载数据 ----------
    print("\n[1/9] 加载数据...")
    train = pd.read_csv('./TrainSet.csv')
    test = pd.read_csv('./TestSet.csv')
    print(f"训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")
    print(f"\n训练集前5行:")
    print(train.head())
    
    # ---------- 2. 解析特征 ----------
    print("\n[2/9] 解析特征...")
    train_list = []
    for items in train.values:
        train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])
    train_df = pd.DataFrame(np.array(train_list))
    feature_dim = len(train_list[0]) - 2
    train_df.columns = ['id'] + [f's_{i}' for i in range(feature_dim)] + ['label']
    
    test_list = []
    for items in test.values:
        test_list.append([items[0]] + [float(i) for i in items[1].split(',')])
    test_df = pd.DataFrame(np.array(test_list))
    test_df.columns = ['id'] + [f's_{i}' for i in range(len(test_list[0])-1)]
    
    print(f"特征维度: {feature_dim}")
    
    # ---------- 3. EDA可视化分析 ----------
    print("\n[3/9] EDA可视化分析...")
    
    # 分离特征和标签用于可视化
    y_train_vis = train_df['label']
    X_train_vis = train_df.drop(['id', 'label'], axis=1)
    
    # 3.1 类别分布图
    plot_class_distribution(y_train_vis)
    
    # 3.2 特征分布图
    plot_feature_distribution(X_train_vis, n_features=20)
    
    # 3.3 箱线图（异常值检测）
    plot_boxplots(X_train_vis, n_features=12)
    
    # 3.4 样本序列图
    plot_sample_sequences(X_train_vis, y_train_vis, n_samples=5)
    
    # 3.5 PCA可视化
    # 采样部分数据以加快PCA计算
    sample_size = min(5000, len(X_train_vis))
    X_sample = X_train_vis.sample(n=sample_size, random_state=42)
    y_sample = y_train_vis.loc[X_sample.index]
    plot_pca_visualization(X_sample, y_sample)
    
    # ---------- 4. 数据预处理 ----------
    print("\n[4/9] 数据预处理...")
    
    # 4.1 内存优化
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    # 4.2 分离特征和标签
    y_train = train_df['label']
    x_train = train_df.drop(['id', 'label'], axis=1)
    X_test = test_df.drop(['id'], axis=1)
    
    # 4.3 异常值处理
    print("处理异常值...")
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns
    x_train = handle_outliers(x_train, numeric_cols, method='iqr')
    
    # 4.4 数据标准化
    print("数据标准化...")
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(x_train),
        columns=x_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # 4.5 相关性热力图
    train_with_label = x_train_scaled.copy()
    train_with_label['label'] = y_train.values
    plot_correlation_heatmap(train_with_label, top_k=30)
    
    # ---------- 5. SMOTE上采样 ----------
    print("\n[5/9] SMOTE上采样处理类别不平衡...")
    smote = SMOTE(random_state=42, n_jobs=-1)
    X_resampled, y_resampled = smote.fit_resample(x_train_scaled, y_train)
    print(f"上采样前: {x_train_scaled.shape}")
    print(f"上采样后: {X_resampled.shape}")
    print(f"上采样后类别分布: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # 绘制SMOTE后的类别分布
    plot_class_distribution(y_train, y_resampled)
    
    # ---------- 6. 划分训练集和验证集 ----------
    print("\n[6/9] 划分训练集和验证集...")
    X_train, X_val, y_train_enc, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.1, random_state=42, stratify=y_resampled
    )
    
    # 重塑为CNN输入格式
    X_train_cnn = X_train.values.reshape(-1, X_train.shape[1], 1).astype(np.float32)
    X_val_cnn = X_val.values.reshape(-1, X_val.shape[1], 1).astype(np.float32)
    X_test_cnn = X_test_scaled.values.reshape(-1, X_test_scaled.shape[1], 1).astype(np.float32)
    
    print(f"训练集形状: {X_train_cnn.shape}")
    print(f"验证集形状: {X_val_cnn.shape}")
    print(f"测试集形状: {X_test_cnn.shape}")
    
    # ---------- 7. 创建并训练Net1单模型 ----------
    print("\n[7/9] 创建并训练Net1模型...")
    model1 = Net1(input_dim=feature_dim)
    # 构建模型
    dummy = tf.random.normal((1, feature_dim, 1))
    _ = model1(dummy)
    # 训练
    history1 = train_model(model1, 'Net1', X_train_cnn, y_train_enc, X_val_cnn, y_val, epochs=50)
    # 绘制单模型训练曲线
    plot_training_curves_single(history1, 'Net1')
    
    # ---------- 8. 模型评估 ----------
    print("\n[8/9] 模型评估...")
    # 验证集预测
    val_pred1 = model1.predict(X_val_cnn, verbose=0)
    # 绘制单混淆矩阵
    plot_confusion_matrix_single(val_pred1, y_val, 'Net1')
    # 打印分类报告
    y_pred_class = np.argmax(val_pred1, axis=1)
    print(f"\nNet1 分类报告:")
    print(classification_report(y_val, y_pred_class, target_names=[f'类别{i}' for i in range(4)]))
    
    # ---------- 9. 预测（无多模型融合，直接Net1输出） ----------
    print("\n[9/9] Net1模型预测...")
    pred1 = model1.predict(X_test_cnn, verbose=0)
    predictions = pred1
    print(f"预测完成，结果形状: {predictions.shape}")
    # 绘制预测概率分布
    plot_prediction_distribution(predictions)
    
    # ---------- 生成提交文件 ----------
    print("\n生成提交文件...")
    submit = pd.DataFrame()
    submit['id'] = range(100000, 100000 + len(predictions))
    submit['label_0'] = predictions[:, 0]
    submit['label_1'] = predictions[:, 1]
    submit['label_2'] = predictions[:, 2]
    submit['label_3'] = predictions[:, 3]
    
    # 后处理
    threshold = 0.5
    for idx, row in submit.iterrows():
        max_prob = max(row[1:])
        if max_prob > threshold:
            for i in range(1, 5):
                submit.iloc[idx, i] = 1 if row[i] > threshold else 0
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'./submit_{timestamp}.csv'
    submit.to_csv(output_file, index=False)
    print(f"\n预测结果已保存: {output_file}")
    
    # 输出文件清单
    print("\n" + "="*60)
    print("生成可视化文件清单:")
    print("="*60)
    print("1. class_distribution.png - 类别分布图")
    print("2. feature_distributions.png - 特征分布图")
    print("3. boxplots.png - 箱线图（异常值检测）")
    print("4. sample_sequences.png - 样本序列图")
    print("5. pca_visualization.png - PCA降维可视化")
    print("6. correlation_heatmap.png - 特征相关性热力图")
    print("7. training_curves_single.png - Net1单模型训练曲线图")
    print("8. confusion_matrix_single.png - Net1混淆矩阵")
    print("9. prediction_distribution.png - 预测概率分布")
    print(f"10. {output_file} - 预测结果提交文件")
    print("="*60)
    print("程序执行完成！")


if __name__ == "__main__":
    main()