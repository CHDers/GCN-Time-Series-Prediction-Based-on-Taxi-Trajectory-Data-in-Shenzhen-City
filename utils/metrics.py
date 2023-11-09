# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:32
# @Author  : Yanjun Hao
# @Site    : 
# @File    : metrics.py
# @Software: PyCharm 
# @Comment :


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import numpy as np
import torch

"""
class Metrics
func :  define metrics for 2-d array
parameter
Y_true : grand truth  (n, 156)
Y_pred : prediction   (n, 156)
"""


class Metrics:  ##二维
    def __init__(self, Y_true, Y_pred):
        self.Y_true = Y_true
        self.Y_pred = Y_pred

    def weighted_mean_absolute_percentage_error(self):
        total_sum = np.sum(self.Y_true)
        average = []
        for i in range(len(self.Y_true)):
            for j in range(len(self.Y_true[0])):
                if self.Y_true[i][j] > 0:
                    # 加权   (y_true[i][j]/np.sum(y_true[i]))*
                    temp = (self.Y_true[i][j] / total_sum) * np.abs(
                        (self.Y_true[i][j] - self.Y_pred[i][j]) / self.Y_true[i][j])
                    average.append(temp)
        return np.sum(average)

    def evaluate_performance(self):
        RMSE = sqrt(mean_squared_error(self.Y_true, self.Y_pred))
        R2 = r2_score(self.Y_true, self.Y_pred)
        MAE = mean_absolute_error(self.Y_true, self.Y_pred)
        WMAPE = self.weighted_mean_absolute_percentage_error()
        return RMSE, R2, MAE, WMAPE


class Metrics_1d:  ## 一维
    def __init__(self, Y_true, Y_pred):
        self.Y_true = Y_true
        self.Y_pred = Y_pred

    def weighted_mean_absolute_percentage_error(self):
        total_sum = np.sum(self.Y_true)
        average = []
        for i in range(len(self.Y_true)):
            if self.Y_true[i] > 0:
                # 加权   (y_true[i][j]/np.sum(y_true[i]))*
                temp = (self.Y_true[i] / total_sum) * np.abs((self.Y_true[i] - self.Y_pred[i]) / self.Y_true[i])
                average.append(temp)
        return np.sum(average)

    def evaluate_performance(self):
        RMSE = sqrt(mean_squared_error(self.Y_true, self.Y_pred))
        R2 = r2_score(self.Y_true, self.Y_pred)
        MAE = mean_absolute_error(self.Y_true, self.Y_pred)
        WMAPE = self.weighted_mean_absolute_percentage_error()
        return RMSE, R2, MAE, WMAPE

