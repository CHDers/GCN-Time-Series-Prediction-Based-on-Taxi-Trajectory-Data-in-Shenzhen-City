# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:35
# @Author  : Yanjun Hao
# @Site    : 
# @File    : predict.py
# @Software: PyCharm 
# @Comment :


import numpy as np
import time, torch
from torch import nn
from model.main_model import Model
import matplotlib.pyplot as plt
from utils.metrics import Metrics, Metrics_1d
from data.get_dataloader import get_speed_dataloader
from utils.utils import GetLaplacian

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 设置相关模型参数取值
epoch_num = 2000
lr = 0.001
time_interval = 15
time_lag = 10
tg_in_one_day = 96
forecast_day_number = 5
pre_len = 1
batch_size = 32
station_num = 156
model_type = 'ours'
TIMESTAMP = str(time.strftime("%Y_%m_%d_%H_%M_%S"))
save_dir = './save_model/' + model_type + '_' + TIMESTAMP  # 模型保存地址

speed_data_loader_train, speed_data_loader_val, speed_data_loader_test, max_speed, min_speed = get_speed_dataloader(
    time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
    forecast_day_number=forecast_day_number,
    pre_len=pre_len, batch_size=batch_size)

# get normalized adj
adjacency = np.loadtxt('./data/sz_adj1.csv', delimiter=",")
adjacency = torch.tensor(GetLaplacian(adjacency).get_normalized_adj(station_num)).type(torch.float32).to(device)

global_start_time = time.time()

model = Model(time_lag, pre_len, station_num, device)

if torch.cuda.is_available():
    model.cuda()

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

path = './bestmodel.pth'
checkpoint = torch.load(path)
model.load_state_dict(checkpoint, strict=True)  ##加载已经训练好的模型
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# test
result = []
result_original = []

with torch.no_grad():
    model.eval()
    test_loss = 0
    for speed_te in enumerate(speed_data_loader_test):
        i_batch, (test_speed_X, test_speed_Y, test_speed_Y_original) = speed_te
        test_speed_X, test_speed_Y = test_speed_X.type(torch.float32).to(device), test_speed_Y.type(torch.float32).to(
            device)
        target = model(test_speed_X, adjacency)
        loss = mse(input=test_speed_Y, target=target)
        test_loss += loss.item()
        # evaluate on original scale
        # 获取result (batch, 276, pre_len)
        clone_prediction = target.cpu().detach().numpy().copy() * (
                max_speed - min_speed) + min_speed  # clone(): Copy the tensor and allocate the new memory
        #         print(clone_prediction.shape)  # (16, 276, 1)
        for i in range(clone_prediction.shape[0]):
            result.append(clone_prediction[i])

        # 获取result_original
        test_speed_Y_original = test_speed_Y_original.cpu().detach().numpy()
        # print(test_OD_Y_original.shape)  # (16, 276, 1)
        for i in range(test_speed_Y_original.shape[0]):
            result_original.append(test_speed_Y_original[i])

    print(np.array(result).shape, np.array(result_original).shape)  # (num, 276, 1)
    # 取整&非负取0
    result = np.array(result).astype('int')
    result[result < 0] = 0
    result_original = np.array(result_original).astype('int')
    result_original[result_original < 0] = 0

    result = np.array(result).reshape(station_num, -1)
    result_original = result_original.reshape(station_num, -1)

    RMSE, R2, MAE, WMAPE = Metrics(result_original, result).evaluate_performance()

    avg_test_loss = test_loss / len(speed_data_loader_test)
    print('test Loss:', avg_test_loss)
ALL = [RMSE, MAE, WMAPE]
print("ALL:", ALL)
print("end")
