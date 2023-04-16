import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# 读取数据集
data = pd.read_csv("D:\DQN-SDP\Dataset\PC1_smote.csv")

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 进行特征选择
chi_selector = SelectKBest(chi2, k=10)  # 选择前10个最相关的特征
X_kbest = chi_selector.fit_transform(X, y)

# 将选择的特征转换为dataframe格式
mask = chi_selector.get_support()  # 获取掩码
selected_features = X.columns[mask]  # 获取选择的特征
X_new = pd.DataFrame(X_kbest, columns=selected_features)
print(X_new)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# 训练SVM模型
clf = SVC(kernel='poly')
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 输出准确率
print("Accuracy:", accuracy_score(y_test, y_pred))
# 计算AUC值
auc = roc_auc_score(y_test, y_pred)

print("SVM模型的AUC值为", auc)

# 计算精度值
pre = precision_score(y_test, y_pred)

print("SVM模型的precision值为", pre)

print("SVM模型的f1值为", f1_score(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef

print("SVM模型的MCC值为", matthews_corrcoef(y_test, y_pred))