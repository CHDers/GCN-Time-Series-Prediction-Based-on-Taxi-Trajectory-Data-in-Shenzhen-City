# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:38
# @Author  : Yanjun Hao
# @Site    : 
# @File    : visualization.py
# @Software: PyCharm 
# @Comment :


x = [[], [], [], [], []]
y = [[], [], [], [], []]
for i in range(result.shape[0]):
    x[0].append(result[i][4][0])
    y[0].append(result_original[i][4][0])
    x[1].append(result[i][18][0])
    y[1].append(result_original[i][18][0])
    x[2].append(result[i][30][0])
    y[2].append(result_original[i][30][0])
    x[3].append(result[i][60][0])
    y[3].append(result_original[i][60][0])
    x[4].append(result[i][90][0])
    y[4].append(result_original[i][90][0])

x = x[1]
y = y[1]
plt.figure(figsize=(20, 8), dpi=150)
plt.xlabel("Time granularity=15min")
plt.ylabel("Speed")
L1, = plt.plot(x, color="r")
L2, = plt.plot(y, color="y")
plt.legend([L1, L2], ["pre", "actual"], loc='best')
plt.show()