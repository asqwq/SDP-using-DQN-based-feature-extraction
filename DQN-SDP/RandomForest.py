# 导入必要的库
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# 加载数据集
data = pd.read_csv('D:\DQN-SDP\Dataset\pc1.csv')

# 数据预处理
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 拟合模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 输出模型性能评估指标
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef

print("随机森林模型的MCC值为", matthews_corrcoef(y_test, y_pred))