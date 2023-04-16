import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

start_time = time.time()
# 读取数据集
data = pd.read_csv('D:\DQN-SDP\Dataset\PC1_smote.csv')

# 提取特征列
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 实例化PCA模型，指定要提取的特征数量
pca = PCA(n_components=2)

# 对特征数据进行PCA降维
X_pca = pca.fit_transform(X)

# 将PCA降维后的数据转换为DataFrame格式
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df_pca, y, test_size=0.2, random_state=42)

# 实例化逻辑回归模型
model = LogisticRegression()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行缺陷预测
y_pred = model.predict(X_test)

# 计算预测性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 输出预测性能指标
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1-score: {:.2f}'.format(f1))

end_time = time.time()
run_time = end_time - start_time
print(run_time)