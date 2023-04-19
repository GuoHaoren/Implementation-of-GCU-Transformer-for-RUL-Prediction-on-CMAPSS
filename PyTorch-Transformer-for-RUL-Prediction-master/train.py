from sys import set_asyncgen_hooks
import torch

torch.manual_seed(1)
from model import *
from data_process.data_processing2 import *
from visualize import *
from data_process.loader import *
from torch.utils.data import DataLoader
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.logger import init_logger
import argparse
import time
import logging
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import pdb
import numpy as np
import random
from torch.utils.data.sampler import SubsetRandomSampler
from itertools import cycle
from model import GCU_Transformer



class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def Training(opt):
    PATH = opt.path + "-" + opt.dataset
    logger = init_logger(opt.save_path, opt, True)

    WRITER = SummaryWriter(log_dir=opt.save_path)

    ##------load parameters--------##
    dataset=opt.dataset
    num_epochs = opt.epoch  # Number of training epochs
    d_model = opt.dim_en  # dimension in encoder
    heads = opt.head  # number of heads in multi-head attention
    N = opt.num_enc_layers  # number of encoder layers
    m = opt.num_features  # number of features
    dropout = opt.drop_out
    batch_size = opt.batch_size
    train_seq_len = opt.train_seq_len
    test_seq_len = opt.test_seq_len
    split_slice = opt.slices
    patch_size = opt.patch_size
    LR = opt.LR
    smooth_param = opt.smooth_param
    ##------Model to CUDA------##

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ##------load dataset--------##
    group_train, y_test, group_test, X_test = data_processing(dataset, smooth_param)

    print("data processed")
    train_dataset = SequenceDataset(mode='train', group=group_train, sequence_train=train_seq_len,
                                    patch_size=train_seq_len)
    test_dataset = SequenceDataset(mode='test', group=group_test, y_label=y_test, sequence_train=train_seq_len,
                                   patch_size=train_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("train loaded")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False)
    print("test loaded")

    ##------SAVE PATH--------##
    if opt.path == '':
        PATH = "train-model-" + time.strftime("%m-%d-%H:%M:%S", time.localtime()) + ".pth"
    else:
        PATH = PATH

    logger.cprint("------Train-------")
    logger.cprint("------" + PATH + "-------")
    # result.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # result.cprint("Path Name:%s" % (PATH))

    ##------model define--------##
    # pretrain_model = PretrainTransformer(m, d_model, N, heads, dropout, batch_size, patch_size)
    train_model = GCU_Transformer(seq_size=train_seq_len, patch_size=patch_size, in_chans=m,
                                   embed_dim=d_model, depth=N, num_heads=heads,
                                   decoder_embed_dim=d_model, decoder_depth=N, decoder_num_heads=heads,
                                   norm_layer=nn.LayerNorm, batch_size=batch_size)

    print(train_model)

    # ------put model to GPU------#
    if torch.cuda.is_available():
        train_model = train_model.to(device)

    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = torch.nn.MSELoss(reduction="mean")
    optimization = torch.optim.Adam(filter(lambda p: p.requires_grad, train_model.parameters()), lr=LR,
                                    weight_decay=opt.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimization, step_size=opt.decay_step, gamma=opt.decay_ratio)

    for p in train_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # 服从均匀分布的glorot初始化器

    loss1 = torch.tensor(0)

    loss2 = torch.tensor(0)
    loss3 = torch.tensor(0)
    best_test_rmse1 = 10000
    final_best_test_rmse = 10000
    for epoch in range(num_epochs):
        train_model.train()
        train_epoch_loss = 0
        train_epoch_loss2 = 0
        # train_epoch_loss3 = 0
        # train_epoch_loss4 = 0
        iter_num = 0

        for X, y in train_loader:
            iter_num += 1

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            y_pred = train_model.forward(X)

            loss = criterion(y_pred.reshape(y_pred.shape[0]), y)  # mse loss
            optimization.zero_grad()
            loss.backward()
            # lr_scheduler.step()
            optimization.step()

            train_epoch_loss = train_epoch_loss + loss.item()

            if iter_num % 10 == 0:
                train_model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    res = 0
                    for X, y in test_loader:
                        if torch.cuda.is_available():
                            X = X.cuda()
                            y = y.cuda()

                        y_hat_recons = train_model.forward(X)
                        y_hat_unscale = y_hat_recons * 125

                        subs = y_hat_unscale.reshape(y_hat_recons.shape[0]) - y
                        subs = subs.cpu().detach().numpy()

                        if subs[0] < 0:
                            res = res + np.exp(-subs / 13)[0] - 1
                        else:
                            res = res + np.exp(subs / 10)[0] - 1

                        loss = criterion(y_hat_unscale.reshape(y_hat_recons.shape[0]), y)
                        test_epoch_loss = test_epoch_loss + loss
                    test_loss = torch.sqrt(test_epoch_loss / len(test_loader))
                    WRITER.add_scalar('Test loss', test_loss, epoch)
                    if epoch >= 10 and test_loss < best_test_rmse1:
                        best_test_rmse1 = test_loss
                        best_score = res
                        cur_best = train_model.state_dict()
                        best_model_path = PATH + "_new_best" + ".pth"
                        torch.save(cur_best, best_model_path)
                        logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                        logger.cprint(
                            "========New Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (test_loss, res))
                        logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                train_model.train()

        train_epoch_loss = np.sqrt(train_epoch_loss / len(train_loader))

        WRITER.add_scalar('Train RMSE', train_epoch_loss, epoch)

        train_model.eval()
        with torch.no_grad():
            test_epoch_loss = 0
            res = 0
            for X, y in test_loader:
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                y_hat_recons = train_model.forward(X)
                y_hat_unscale = y_hat_recons * 125

                subs = y_hat_unscale.reshape(y_hat_recons.shape[0]) - y
                subs = subs.cpu().detach().numpy()

                if subs[0] < 0:
                    res = res + np.exp(-subs / 13)[0] - 1
                else:
                    res = res + np.exp(subs / 10)[0] - 1

                loss = criterion(y_hat_unscale.reshape(y_hat_recons.shape[0]), y)
                test_epoch_loss = test_epoch_loss + loss

            test_loss = torch.sqrt(test_epoch_loss / len(test_loader))
            WRITER.add_scalar('Test loss', test_loss, epoch)
            if epoch >= 10 and test_loss < final_best_test_rmse:
                final_best_test_rmse = test_loss
                if test_loss < best_test_rmse1:
                    best_test_rmse1 = test_loss
                    best_score = res
                    cur_best = train_model.state_dict()
                    best_model_path = PATH + "_new_best" + ".pth"
                    torch.save(cur_best, PATH + "_new_best" + ".pth")
                    logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    logger.cprint(
                        "========New Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (test_loss, res))
                    logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                torch.save(cur_best, PATH + "final_new_best" + ".pth")
                final_best_score = res
                cur_best = train_model.state_dict()
                best_model_path = PATH + "final_new_best" + ".pth"
                torch.save(cur_best, best_model_path)
                logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                logger.cprint(
                    "========New Final Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (test_loss, res))
                logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        logger.cprint("Epoch Loss: %d, training loss: %1.5f, testing rmse: %1.5f, score: %1.5f" % (
        epoch, train_epoch_loss, test_loss, res))
        logger.cprint("------------------------------------------------------------")

    logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.cprint("========New Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (best_test_rmse1, best_score))
    logger.cprint("========New Final Best Test Loss Updata: %1.5f Best Score: %1.5f========" % (
    final_best_test_rmse, final_best_score))
    logger.cprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return
