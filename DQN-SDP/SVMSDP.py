import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# 读取数据集
df = pd.read_csv("D:\DQN-SDP\Dataset\kc2.csv")

# 分离自变量和因变量
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义SVM模型
svm = SVC(kernel='linear', C=1)  # c为惩罚参数，kernel为核函数 可以为linear、poly、rbf、sigmoid、precomputed

# 在训练集上拟合模型
svm.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 打印准确率
print("Accuracy:", accuracy)


# 计算AUC值
auc = roc_auc_score(y_test, y_pred)

print("SVM模型的AUC值为", auc)

# 计算精度值
pre = precision_score(y_test, y_pred)

print("SVM模型的precision值为", pre)

print("SVM模型的f1值为", f1_score(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef

print("SVM模型的MCC值为", matthews_corrcoef(y_test, y_pred))