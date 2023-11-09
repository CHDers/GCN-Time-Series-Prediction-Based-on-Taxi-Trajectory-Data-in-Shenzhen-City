# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:31
# @Author  : Yanjun Hao
# @Site    : 
# @File    : utils.py.py
# @Software: PyCharm 
# @Comment :

import numpy as np
from math import sqrt


class GetLaplacian:
    def __init__(self, adjacency):
        self.adjacency = adjacency

    def get_normalized_adj(self, station_num):
        I = np.matrix(np.eye(station_num))
        A_hat = self.adjacency + I
        D_hat = np.array(np.sum(A_hat, axis=0))[0]
        D_hat_sqrt = [sqrt(x) for x in D_hat]
        D_hat_sqrt = np.array(np.diag(D_hat_sqrt))
        D_hat_sqrtm_inv = np.linalg.inv(D_hat_sqrt)  # 开方后求逆即为矩阵的-1/2次方
        # D_A_final=D_hat**-1/2 * A_hat *D_hat**-1/2
        D_A_final = np.dot(D_hat_sqrtm_inv, A_hat)
        D_A_final = np.dot(D_A_final, D_hat_sqrtm_inv)
        # print(D_A_final.shape)
        return np.array(D_A_final, dtype="float32")
