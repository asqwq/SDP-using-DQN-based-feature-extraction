import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


# 读取数据集
data = pd.read_csv('D:\DQN-SDP\Dataset\PC1_smote.csv')

# 将数据集拆分为特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 创建KNN分类器模型
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)


print("KNN模型的准确率为：{:.2f}%".format(accuracy*100))



# 计算AUC值
auc = roc_auc_score(y_test, y_pred)

print("KNN模型的AUC值为", auc)

# 计算精度值

pre = precision_score(y_test, y_pred)

print("KNN模型的precision值为", pre)


print("KNN模型的F-度量值为", f1_score(y_test, y_pred))

print("KNN模型的MCC值为", matthews_corrcoef(y_test, y_pred))