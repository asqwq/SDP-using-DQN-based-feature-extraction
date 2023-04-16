import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, t, f, chi2
import itertools


# 读取数据
data = pd.read_csv("D:\DQN-SDP\Dataset\\AUCbase.csv")
# 绘制箱线图
plt.figure(figsize=(10, 8))
boxplot = data.boxplot(column='value', by="group")
boxplot.set_title('')
# 计算Scott-Knott多重比较方法的p值
def scott_knott(data, alpha=0.05):
    """
    使用Scott-Knott多重比较方法计算p值
    """
    n = len(data)
    means = data.mean()
    sorted_means = sorted(means, reverse=True)
    treatments = {idx: [mean] for idx, mean in enumerate(means)}
    last_p = 0
    max_p = 0
    for k in range(1, n):
        p_value = f.ppf(1 - alpha, k, n - k) if k > 2 else norm.ppf(1 - alpha / 2)
        groups = itertools.combinations(range(n), k)
        for group in groups:
            new_p = max(treatments[t][0] for t in group) - min(treatments[t][0] for t in group)
            if new_p > max_p:
                max_p = new_p
                max_group = group
        if max_p > p_value and last_p > max_p:
            break
        treatments[n + k - 1] = [sorted_means[idx] for idx in max_group]
        last_p = max_p
        max_p = 0
    return treatments
# 显示图形
plt.ylabel('AUC')
plt.xlabel(None)
plt.suptitle(None)
plt.savefig('D:\DQN-SDP\\fig\\AUCbase.pdf', format='pdf', bbox_inches='tight')
plt.show()
