# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:53
# @Author  : Yanjun Hao
# @Site    : 
# @File    : earlystopping.py
# @Software: PyCharm 
# @Comment :


import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=100, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model_dict, model, epoch, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, model, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}', self.val_loss_min
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_dict, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_dict, model, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...'
            )
        torch.save(model_dict, save_path + "/" + "model_dict_checkpoint_{}_{:.8f}.pth".format(epoch, val_loss))
        # torch.save(model, save_path + "/" + "model_checkpoint_{}_{:.8f}.pth".format(epoch, val_loss))
        self.val_loss_min = val_loss
