import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier


# 读取数据集
data = pd.read_csv('D:\DQN-SDP\Dataset\\ar1_smote.csv')

# 提取特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用信息增益特征选择技术选择前10个特征
selector = SelectKBest(mutual_info_classif, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

# 使用KNN分类器进行缺陷预测
knn = KNeighborsClassifier()
knn.fit(X_train_new, y_train)
y_pred = knn.predict(X_test_new)

# 计算precision值
precision = precision_score(y_test, y_pred, average='binary')
print('Precision:', precision)

auc = roc_auc_score(y_test, y_pred)

print("KNN模型的AUC值为", auc)




print("KNN模型的F-度量值为", f1_score(y_test, y_pred))

print("KNN模型的MCC值为", matthews_corrcoef(y_test, y_pred))

# 定义SVM模型
clf = DecisionTreeClassifier(max_depth=4)

# 在训练集上拟合模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)




# 计算AUC值
auc = roc_auc_score(y_test, y_pred)

print("DT模型的AUC值为", auc)

# 计算精度值
pre = precision_score(y_test, y_pred)

print("DT模型的precision值为", pre)

print("DT模型的f1值为", f1_score(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef

print("DT模型的MCC值为", matthews_corrcoef(y_test, y_pred))