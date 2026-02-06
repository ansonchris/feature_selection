# 1. 导入核心库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')  # 忽略无关警告

# 设置可视化样式，让图更清晰
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

# 2. 定义KS计算函数（风控建模核心评估指标，比AUC更贴合）
def calculate_ks(y_true, y_pred, n_bins=10):
    """
    计算KS值
    :param y_true: 真实标签
    :param y_pred: 模型预测概率
    :param n_bins: 分箱数
    :return: ks值、ks对应的分箱
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # 等频分箱
    df['bin'] = pd.qcut(df['y_pred'], n_bins, duplicates='drop')
    # 按分箱统计
    bin_stats = df.groupby('bin')['y_true'].agg(['count', 'sum']).reset_index()
    bin_stats.columns = ['bin', 'total', 'bad']
    bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
    # 计算累计坏样本率和累计好样本率
    bin_stats['cum_bad'] = bin_stats['bad'].cumsum() / bin_stats['bad'].sum()
    bin_stats['cum_good'] = bin_stats['good'].cumsum() / bin_stats['good'].sum()
    # 计算KS
    bin_stats['ks'] = abs(bin_stats['cum_bad'] - bin_stats['cum_good'])
    ks_value = bin_stats['ks'].max()
    return ks_value, bin_stats

# 3. 加载并预处理数据（演示用信用卡欺诈数据集，可替换为你的业务数据）
# 数据集说明：特征V1-V28为预处理后的连续特征，Amount为交易金额，Class为标签（0=正常，1=欺诈，不平衡数据）
data = pd.read_csv('https://www.kaggle.com/mlg-ulb/creditcardfraud/download?datasetVersionNumber=1')
# 简单预处理（XGBoost可容忍少量缺失，此处仅做金额标准化，适配不平衡数据）
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop('Time', axis=1)  # 剔除无业务意义的时间特征

# 划分特征和标签
X = data.drop('Class', axis=1)
y = data['Class']
feature_names = X.columns.tolist()  # 保存原始特征名

# 关键：划分训练集/测试集，**仅用训练集做特征筛选**，测试集仅用于验证筛选效果
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify=y：分层抽样，保证训练/测试集标签分布一致
)
print(f"原始特征数量：{len(feature_names)}")
print(f"训练集样本数：{X_train.shape[0]}, 测试集样本数：{X_test.shape[0]}")

# 4. 定义并训练XGBoost模型（二分类，风控/分类建模通用参数）
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # 二分类目标函数
    eval_metric='auc',            # 评估指标AUC
    learning_rate=0.1,            # 学习率
    max_depth=5,                  # 树深度
    n_estimators=100,             # 树的数量
    subsample=0.8,                # 采样样本比例
    colsample_bytree=0.8,         # 采样特征比例
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # 处理不平衡数据，正样本权重
)
# 训练模型（仅用训练集）
xgb_model.fit(X_train, y_train)

# 5. 提取XGBoost特征重要性（核心：Gain增益，远优于默认的Weight）
# 方式：通过get_booster()获取原生模型，提取importance_type='gain'的重要性
feature_importance = xgb_model.get_booster().get_score(
    importance_type='gain'  # 可选：gain(增益)、weight(分裂次数)、cover(覆盖样本数)
)
# 转换为DataFrame，方便排序和筛选
fi_df = pd.DataFrame({
    'feature': list(feature_importance.keys()),
    'gain': list(feature_importance.values())
})
# 按gain降序排序
fi_df = fi_df.sort_values(by='gain', ascending=False).reset_index(drop=True)
# 计算相对增益（占总增益的比例，更易设置阈值）
fi_df['relative_gain'] = fi_df['gain'] / fi_df['gain'].sum()
print("\n前10个高重要性特征（按Gain）：")
print(fi_df.head(10))

# 6. 特征重要性可视化（横向柱状图，直观查看）
plt.figure(figsize=(12, 8))
# 绘制前20个特征（避免特征过多图过密）
top_n_vis = 20
sns_data = fi_df.head(top_n_vis)
plt.barh(range(len(sns_data)), sns_data['gain'], color='#1f77b4')
plt.yticks(range(len(sns_data)), sns_data['feature'])
plt.xlabel('XGBoost Gain（特征增益）')
plt.ylabel('特征名')
plt.title(f'XGBoost特征重要性TOP{top_n_vis}（按Gain排序）')
plt.gca().invert_yaxis()  # 从上到下按Gain降序
plt.tight_layout()
plt.show()

# 7. XGBoost特征筛选（两种常用方式，二选一即可，也可结合使用）
## 方式1：按**前N名**筛选（推荐，适合明确想要的特征数量，如风控选10-20个）
select_top_n = 15  # 自定义：保留前15个高Gain特征
selected_features_topn = fi_df.head(select_top_n)['feature'].tolist()

## 方式2：按**相对增益阈值**筛选（适合按预测能力筛选，如保留相对增益>0.01的特征）
gain_threshold = 0.01  # 自定义：相对增益阈值，可根据业务调整
selected_features_thresh = fi_df[fi_df['relative_gain'] > gain_threshold]['feature'].tolist()

# 输出筛选结果
print(f"\n方式1-按前{select_top_n}名筛选，保留特征数：{len(selected_features_topn)}")
print(f"筛选后特征：{selected_features_topn}")
print(f"\n方式2-按相对增益>{gain_threshold}筛选，保留特征数：{len(selected_features_thresh)}")
print(f"筛选后特征：{selected_features_thresh}")

# 8. 验证筛选后特征的模型性能（对比原始特征，确保筛选后性能无显著下降）
def evaluate_model(X_tr, X_te, y_tr, y_te, features, model):
    """
    评估模型性能（AUC+KS）
    :param features: 待评估的特征列表
    :return: auc, ks
    """
    # 用指定特征训练模型
    model.fit(X_tr[features], y_tr)
    # 预测测试集概率
    y_pred = model.predict_proba(X_te[features])[:, 1]
    # 计算AUC和KS
    auc = roc_auc_score(y_te, y_pred)
    ks, _ = calculate_ks(y_te, y_pred)
    return auc, ks

# 评估原始特征模型性能
auc_original, ks_original = evaluate_model(X_train, X_test, y_train, y_test, feature_names, xgb_model)
# 评估方式1筛选后特征性能
auc_topn, ks_topn = evaluate_model(X_train, X_test, y_train, y_test, selected_features_topn, xgb_model)
# 评估方式2筛选后特征性能
auc_thresh, ks_thresh = evaluate_model(X_train, X_test, y_train, y_test, selected_features_thresh, xgb_model)

# 输出性能对比
print("\n========== 筛选前后模型性能对比（测试集）==========")
print(f"原始特征 | AUC：{auc_original:.4f}，KS：{ks_original:.4f}")
print(f"前{select_top_n}名特征 | AUC：{auc_topn:.4f}，KS：{ks_topn:.4f}")
print(f"相对增益>{gain_threshold}特征 | AUC：{auc_thresh:.4f}，KS：{ks_thresh:.4f}")

# 9. 输出最终筛选后的特征（可选择任意一种方式的结果）
final_selected_features = selected_features_topn  # 选择前N名的结果，可替换为selected_features_thresh
print(f"\n最终筛选后的特征列表：{final_selected_features}")


#%%

# 1. Import core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')  # Ignore irrelevant warnings

# Set visualization style for clarity
plt.rcParams['font.sans-serif'] = ['SimHei']  # Fix Chinese display issue
plt.rcParams['axes.unicode_minus'] = False

# 2. Define KS calculation function (core evaluation metric for risk control modeling, more relevant than AUC)
def calculate_ks(y_true, y_pred, n_bins=10):
    """
    Calculate KS value
    :param y_true: True labels
    :param y_pred: Predicted probabilities
    :param n_bins: Number of bins for stratification
    :return: KS value, KS-related bin statistics
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # Equal-frequency binning
    df['bin'] = pd.qcut(df['y_pred'], n_bins, duplicates='drop')
    # Statistics by bin
    bin_stats = df.groupby('bin')['y_true'].agg(['count', 'sum']).reset_index()
    bin_stats.columns = ['bin', 'total', 'bad']
    bin_stats['good'] = bin_stats['total'] - bin_stats['bad']
    # Calculate cumulative bad rate and cumulative good rate
    bin_stats['cum_bad'] = bin_stats['bad'].cumsum() / bin_stats['bad'].sum()
    bin_stats['cum_good'] = bin_stats['good'].cumsum() / bin_stats['good'].sum()
    # Calculate KS for each bin
    bin_stats['ks'] = abs(bin_stats['cum_bad'] - bin_stats['cum_good'])
    ks_value = bin_stats['ks'].max()
    return ks_value, bin_stats

# 3. Load and preprocess data (demonstration using credit card fraud dataset; replace with your business data)
# Dataset description: Features V1-V28 are preprocessed continuous features, Amount is transaction amount, 
# Class is label (0=normal, 1=fraud; imbalanced data)
data = pd.read_csv('https://www.kaggle.com/mlg-ulb/creditcardfraud/download?datasetVersionNumber=1')

# Simple preprocessing (XGBoost tolerates minor missing values; only standardize Amount here to adapt to imbalanced data)
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop('Time', axis=1)  # Remove irrelevant Time feature

# Split features and labels
X = data.drop('Class', axis=1)
y = data['Class']
feature_names = X.columns.tolist()  # Save original feature names

# Key step: Split into training/test sets - only use training set for feature selection; test set for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # Stratified sampling to maintain label distribution
)
print(f"Number of original features: {len(feature_names)}")
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# 4. Define and train XGBoost model (binary classification; general parameters for risk control/classification)
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Binary classification objective function
    eval_metric='auc',            # Evaluation metric: AUC
    learning_rate=0.1,            # Learning rate
    max_depth=5,                  # Tree depth
    n_estimators=100,             # Number of trees
    subsample=0.8,                # Sample proportion for training
    colsample_bytree=0.8,         # Feature proportion per tree
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])  # Weight for positive samples (imbalanced data)
)
# Train model (using only training set)
xgb_model.fit(X_train, y_train)

# 5. Extract XGBoost feature importance (Core: Gain - far superior to default Weight)
# Method: Get native model via get_booster(), extract importance with type 'gain'
feature_importance = xgb_model.get_booster().get_score(
    importance_type='gain'  # Options: gain (feature contribution), weight (split count), cover (sample coverage)
)
# Convert to DataFrame for sorting and selection
fi_df = pd.DataFrame({
    'feature': list(feature_importance.keys()),
    'gain': list(feature_importance.values())
})
# Sort by gain in descending order
fi_df = fi_df.sort_values(by='gain', ascending=False).reset_index(drop=True)
# Calculate relative gain (proportion of total gain for easier threshold setting)
fi_df['relative_gain'] = fi_df['gain'] / fi_df['gain'].sum()

print("\nTop 10 high-importance features (sorted by Gain):")
print(fi_df.head(10))

# 6. Feature importance visualization (horizontal bar chart for intuition)
plt.figure(figsize=(12, 8))
# Plot top 20 features to avoid overcrowding
top_n_vis = 20
sns_data = fi_df.head(top_n_vis)
plt.barh(range(len(sns_data)), sns_data['gain'], color='#1f77b4')
plt.yticks(range(len(sns_data)), sns_data['feature'])
plt.xlabel('XGBoost Gain (Feature Contribution)')
plt.ylabel('Feature Name')
plt.title(f'XGBoost Feature Importance TOP{top_n_vis} (Sorted by Gain)')
plt.gca().invert_yaxis()  # Invert y-axis to show top features first
plt.tight_layout()
plt.show()

# 7. XGBoost feature selection (two common methods; choose one or combine)
## Method 1: Select top N features (Recommended for scenarios with clear desired feature count, e.g., 10-20 for risk control)
select_top_n = 15  # Customize: retain top 15 features with highest Gain
selected_features_topn = fi_df.head(select_top_n)['feature'].tolist()

## Method 2: Select by relative gain threshold (Suitable for selecting by predictive power, e.g., retain features with relative_gain > 0.01)
gain_threshold = 0.01  # Customize: relative gain threshold (adjust based on business needs)
selected_features_thresh = fi_df[fi_df['relative_gain'] > gain_threshold]['feature'].tolist()

# Print selection results
print(f"\nMethod 1 - Top {select_top_n} features: {len(selected_features_topn)} features retained")
print(f"Selected features: {selected_features_topn}")
print(f"\nMethod 2 - Features with relative_gain > {gain_threshold}: {len(selected_features_thresh)} features retained")
print(f"Selected features: {selected_features_thresh}")

# 8. Validate model performance after selection (compare with original features to ensure no significant degradation)
def evaluate_model(X_tr, X_te, y_tr, y_te, features, model):
    """
    Evaluate model performance (AUC + KS)
    :param features: List of features to evaluate
    :return: AUC score, KS score
    """
    # Train model with specified features
    model.fit(X_tr[features], y_tr)
    # Predict probabilities on test set
    y_pred = model.predict_proba(X_te[features])[:, 1]
    # Calculate AUC and KS
    auc = roc_auc_score(y_te, y_pred)
    ks, _ = calculate_ks(y_te, y_pred)
    return auc, ks

# Evaluate performance with original features
auc_original, ks_original = evaluate_model(X_train, X_test, y_train, y_test, feature_names, xgb_model)
# Evaluate performance with top N features (Method 1)
auc_topn, ks_topn = evaluate_model(X_train, X_test, y_train, y_test, selected_features_topn, xgb_model)
# Evaluate performance with threshold-based features (Method 2)
auc_thresh, ks_thresh = evaluate_model(X_train, X_test, y_train, y_test, selected_features_thresh, xgb_model)

# Print performance comparison
print("\n========== Model Performance Comparison Before and After Selection (Test Set) ==========")
print(f"Original features | AUC: {auc_original:.4f}, KS: {ks_original:.4f}")
print(f"Top {select_top_n} features | AUC: {auc_topn:.4f}, KS: {ks_topn:.4f}")
print(f"Features with relative_gain > {gain_threshold} | AUC: {auc_thresh:.4f}, KS: {ks_thresh:.4f}")

# 9. Output final selected features (choose result from either method)
final_selected_features = selected_features_topn  # Use top N features; replace with selected_features_thresh if needed
print(f"\nFinal selected features: {final_selected_features}")


#%%

