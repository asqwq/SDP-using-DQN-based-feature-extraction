import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


# 读取数据
data = pd.read_csv('D:\DQN-SDP\Dataset\\ar1_smote.csv')

# 将数据分为特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征选择
def feature_selection(X_train, y_train, X_test):
    features = []
    n_features = X_train.shape[1]
    for i in range(n_features):
        p = np.sum(X_train[:, i] == 1) / len(X_train[:, i]) # 计算第i个特征在训练集中出现的概率
        if p > 0:
            q = np.sum(y_train[X_train[:, i] == 1] == 1) / np.sum(X_train[:, i] == 1) # 计算第i个特征在训练集中标签为1的概率
            if q == 0 or q == 1:
                e = 0
            else:
                e = - (p * q * np.log(q) + p * (1 - q) * np.log(1 - q)) / np.log(2) # 计算第i个特征的期望交叉熵
            features.append(e)
        else:
            features.append(0)
    print(features);

    # 排序并选择前n_features/n个特征
    idx = np.argsort(features)[::-1][:int(n_features/1.6)]
    print(idx)
    X_train_selected = X_train[:, idx]
    X_test_selected = X_test[:, idx]

    return X_train_selected, X_test_selected

# 特征选择后的训练集和测试集
X_train_selected, X_test_selected = feature_selection(np.array(X_train), np.array(y_train), np.array(X_test))

# 训练模型
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1)  #, random_state=42
clf.fit(X_train_selected, y_train)

# 预测测试集并计算准确率
y_pred = clf.predict(X_test_selected)

# 计算特征选择结果
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}%'.format(accuracy * 100))

# 计算AUC值
auc = roc_auc_score(y_test, y_pred)

print("SVM模型的AUC值为", auc)

# 计算精度值
pre = precision_score(y_test, y_pred)

print("SVM模型的precision值为", pre)

print("SVM模型的f1值为", f1_score(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef

print("SVM模型的MCC值为", matthews_corrcoef(y_test, y_pred))

