import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
plt.figure(figsize=(20, 20)) # 画面大小
print("\nRMT关系矩阵实验\n组件求解器的斯皮尔曼相关系数 表格图\n")
# analysis_data = pd.read_csv(args.label_file_path + '/' + args.TestName + '.csv').values
# del_col = []
# for i in range(0, args.NumberSolver*2, 2):
#     del_col.append(i)
# analysis_data = np.delete(analysis_data, del_col, axis = 1)
# analysis_data = analysis_data.astype(int)
# analysis_data = np.array()

orig_data = []
with open("D:\DQN-SDP\Dataset\pc1.csv") as f:
    for line in f.readlines():
        clean_data = line.split(",")[:-1]
        orig_data.append(clean_data)

head_line = orig_data[0]
analysis_data = np.array(orig_data[1:]).astype(float)
print(analysis_data.shape)
# input()

# analysis_data = np.random.randint(0, 100, size=(500, 8))
col_name = head_line
analysis_data = pd.DataFrame(analysis_data, columns=col_name)
corr = analysis_data.corr('spearman') # 去掉就是perason
fig = sns.heatmap(corr, annot=True)
scatter_fig = fig.get_figure()
scatter_fig.savefig("D:\DQN-SDP\\fig\PC1Spearman.svg", format='svg', bbox_inches='tight', pad_inches=0.0)