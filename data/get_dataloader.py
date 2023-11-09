# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:28
# @Author  : Yanjun Hao
# @Site    : 
# @File    : get_dataloader.py
# @Software: PyCharm 
# @Comment :


import torch
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data import DataLoader

"""
Parameter:
time_interval, time_lag, tg_in_one_day, forecast_day_number, is_train=True, is_val=False, val_rate=0.1, pre_len
"""


class Traffic_speed(Dataset):
    def __init__(self, time_interval, time_lag, tg_in_one_day, forecast_day_number, speed_data, pre_len, is_train=True,
                 is_val=False, val_rate=0.1):
        super().__init__()
        # 此部分的作用是将数据集划分为训练集、验证集、测试集。
        # 完成后X的维度为 num*156*10，10代表10个时间步,Y的维度为 num*156*1
        # X为临近同一时段的10个时间步
        # Y为156条主干道未来1个时间步
        self.time_interval = time_interval
        self.time_lag = time_lag
        self.tg_in_one_day = tg_in_one_day
        self.forecast_day_number = forecast_day_number
        self.tg_in_one_week = self.tg_in_one_day * self.forecast_day_number
        self.speed_data = np.loadtxt(speed_data, delimiter=",").T  # 对数据进行转置
        self.max_speed = np.max(self.speed_data)
        self.min_speed = np.min(self.speed_data)
        self.is_train = is_train
        self.is_val = is_val
        self.val_rate = val_rate
        self.pre_len = pre_len

        # Normalization
        self.speed_data_norm = np.zeros((self.speed_data.shape[0], self.speed_data.shape[1]))
        for i in range(len(self.speed_data)):
            for j in range(len(self.speed_data[0])):
                self.speed_data_norm[i, j] = round(
                    (self.speed_data[i, j] - self.min_speed) / (self.max_speed - self.min_speed), 5)
        if self.is_train:
            self.start_index = self.tg_in_one_week + time_lag
            self.end_index = len(self.speed_data[0]) - self.tg_in_one_day * self.forecast_day_number - self.pre_len
        else:
            self.start_index = len(self.speed_data[0]) - self.tg_in_one_day * self.forecast_day_number
            self.end_index = len(self.speed_data[0]) - self.pre_len

        self.X = [[] for index in range(self.start_index, self.end_index)]
        self.Y = []
        self.Y_original = []
        # print(self.start_index, self.end_index)
        for index in range(self.start_index, self.end_index):
            temp = self.speed_data_norm[:, index - self.time_lag: index]  # 邻近几个时间段的进站量
            temp = temp.tolist()
            self.X[index - self.start_index] = temp
            self.Y.append(self.speed_data_norm[:, index:index + self.pre_len])
        self.X, self.Y = torch.from_numpy(np.array(self.X)), torch.from_numpy(np.array(self.Y))  # (num, 276, time_lag)

        # if val is not zero
        if self.val_rate * len(self.X) != 0:
            val_len = int(self.val_rate * len(self.X))
            train_len = len(self.X) - val_len
            if self.is_val:
                self.X = self.X[-val_len:]
                self.Y = self.Y[-val_len:]
            else:
                self.X = self.X[:train_len]
                self.Y = self.Y[:train_len]
        print("X.shape", self.X.shape, "Y.shape", self.Y.shape)

        if not self.is_train:
            for index in range(self.start_index, self.end_index):
                self.Y_original.append(
                    self.speed_data[:, index:index + self.pre_len])  # the predicted speed before normalization
            self.Y_original = torch.from_numpy(np.array(self.Y_original))

    def get_max_min_speed(self):
        return self.max_speed, self.min_speed

    def __getitem__(self, item):
        if self.is_train:
            return self.X[item], self.Y[item]  ##返回训练集
        else:
            return self.X[item], self.Y[item], self.Y_original[item]  ## 返回测试集及真实数据值

    def __len__(self):
        return len(self.X)


def get_speed_dataloader(time_interval=15, time_lag=10, tg_in_one_day=96, forecast_day_number=5, pre_len=1,
                         batch_size=32):
    # train speed data loader
    print("train speed")
    speed_train = Traffic_speed(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                                forecast_day_number=forecast_day_number,
                                pre_len=pre_len, speed_data=speed_data, is_train=True, is_val=False, val_rate=0.1)
    max_speed, min_speed = speed_train.get_max_min_speed()
    speed_data_loader_train = DataLoader(speed_train, batch_size=batch_size, shuffle=False)

    # validation speed data loader
    print("val speed")
    speed_val = Traffic_speed(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                              forecast_day_number=forecast_day_number,
                              pre_len=pre_len, speed_data=speed_data, is_train=True, is_val=True, val_rate=0.1)
    speed_data_loader_val = DataLoader(speed_val, batch_size=batch_size, shuffle=False)

    # test speed data loader
    print("test speed")
    speed_test = Traffic_speed(time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
                               forecast_day_number=forecast_day_number,
                               pre_len=pre_len, speed_data=speed_data, is_train=False, is_val=False, val_rate=0)
    speed_data_loader_test = DataLoader(speed_test, batch_size=batch_size, shuffle=False)

    return speed_data_loader_train, speed_data_loader_val, speed_data_loader_test, max_speed, min_speed


speed_data = "./sz_speed.csv"
