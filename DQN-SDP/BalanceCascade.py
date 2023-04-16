from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, resample
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class BalanceCascade(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=50, n_max_subset=10, random_state=None):
        self.n_estimators = n_estimators
        self.n_max_subset = n_max_subset
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.estimators_ = []
        self.subsets_ = []

        # 创建并拟合基分类器
        for i in range(self.n_estimators):
            est = RandomForestClassifier(n_estimators=10, random_state=self.random_state + i)
            est.fit(X, y)
            self.estimators_.append(est)

            # 在最多n_max_subset个子集中选择具有最佳分类性能的子集
            X_train = X
            y_train = y
            best_subset = None
            best_score = 0
            for j in range(self.n_max_subset):
                X_resampled, y_resampled = resample(X_train, y_train, random_state=self.random_state + j)
                score = est.score(X_resampled, y_resampled)
                if score > best_score:
                    best_score = score
                    best_subset = (X_resampled, y_resampled)

            self.subsets_.append(best_subset)

            # 移除最好的子集的样本
            X_train = X_train[np.logical_not(np.isin(X_train, best_subset[0]).all(axis=1))]
            y_train = y_train[np.logical_not(np.isin(y_train, best_subset[1]))]

        return self

    def predict(self, X):
        check_is_fitted(self)
        predictions = []
        for est, subset in zip(self.estimators_, self.subsets_):
            X_subset, y_subset = subset
            predictions.append(est.predict(X[np.isin(X, X_subset).all(axis=1)]))
        return np.concatenate(predictions)

    def fit_resample(self, X, y):
        self.fit(X, y)
        return self.subsets_[0][0], self.subsets_[0][1]


# 读取数据集
data = pd.read_csv('D:\DQN-SDP\Dataset\pc1.csv')

# 分离特征和标签
X = data.drop('defects', axis=1)
y = data['defects']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建采样器并进行采样
sampler = BalanceCascade(n_estimators=50, n_max_subset=10, random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

# 保存采样结果
resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)


resampled_data.to_csv('D:\DQN-SDP\Dataset\PC1_BalanceCascade.csv', index=False)

