# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:34
# @Author  : Yanjun Hao
# @Site    : 
# @File    : train.py
# @Software: PyCharm 
# @Comment :

import numpy as np
import os, time, torch
from rich import print
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from utils.utils import GetLaplacian
from model.main_model import Model
from utils.earlystopping import EarlyStopping
from data.get_dataloader import get_speed_dataloader

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
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

speed_data_loader_train, speed_data_loader_val, speed_data_loader_test, max_speed, min_speed = get_speed_dataloader(
    time_interval=time_interval, time_lag=time_lag, tg_in_one_day=tg_in_one_day,
    forecast_day_number=forecast_day_number, pre_len=pre_len, batch_size=batch_size)

# get normalized adj
adjacency = np.loadtxt('./data/sz_adj1.csv', delimiter=",")  # 加载邻接矩阵
adjacency = torch.tensor(GetLaplacian(adjacency).get_normalized_adj(station_num)).type(torch.float32).to(device)

global_start_time = time.time()
# writer = SummaryWriter()
model = Model(time_lag, pre_len, station_num, device)
print(model)
if torch.cuda.is_available():
    model.cuda()

# 确定优化器和损失函数
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = torch.nn.MSELoss().to(device)

temp_time = time.time()
early_stopping = EarlyStopping(patience=100, verbose=True)

train_loss_list = []
val_loss_list = []

for epoch in range(0, epoch_num):
    # model train
    train_loss = 0
    ## 便利每个batch,模型训练
    model.train()
    for speed_tr in enumerate(speed_data_loader_train):
        i_batch, (train_speed_X, train_speed_Y) = speed_tr
        train_speed_X, train_speed_Y = train_speed_X.type(torch.float32).to(device), train_speed_Y.type(
            torch.float32).to(device)
        target = model(train_speed_X, adjacency)
        loss = mse(input=train_speed_Y, target=target)

        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        # model validation
        model.eval()
        val_loss = 0
        for speed_val in enumerate(speed_data_loader_val):
            i_batch, (val_speed_X, val_speed_Y) = speed_tr
            val_speed_X, val_speed_Y = val_speed_X.type(torch.float32).to(device), val_speed_Y.type(torch.float32).to(
                device)
            target = model(val_speed_X, adjacency)
            loss = mse(input=val_speed_Y, target=target)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(speed_data_loader_train)
    avg_val_loss = val_loss / len(speed_data_loader_val)
    #     writer.add_scalar("loss_train", avg_train_loss, epoch)
    #     writer.add_scalar("loss_eval", avg_val_loss, epoch)

    print('epoch:', epoch, 'train Loss', avg_train_loss, 'val Loss:', avg_val_loss)

    if epoch > 0:
        # early stopping
        model_dict = model.state_dict()
        early_stopping(avg_val_loss, model_dict, model, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early Stopping")
            break
    # 每10个epoch打印一次训练时间
    if epoch % 10 == 0:
        print("time for 10 epoches:", round(time.time() - temp_time, 2))
        temp_time = time.time()

global_end_time = time.time() - global_start_time
print("global end time:", global_end_time)

Train_time_ALL = []
Train_time_ALL.append(global_end_time)
# np.savetxt('result/lr_' + str(lr) + '_batch_size_' + str(batch_size) + '_Train_time_ALL.txt', Train_time_ALL)
print("end")
