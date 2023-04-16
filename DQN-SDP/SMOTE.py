import pandas as pd
from imblearn.over_sampling import SMOTE

# 加载数据集
data = pd.read_csv('D:\DQN-SDP\Dataset\\ar1.csv')

# 分离特征和标签
X = data.drop(['defects'], axis=1)
y = data['defects']

# print(y)
# input()
# 使用SMOTE算法生成新的样本
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)

# 处理数据
smote_data = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1)

# 将处理后的数据保存到CSV文件中
smote_data.to_csv('D:\DQN-SDP\Dataset\\ar1_smote.csv', index=False)