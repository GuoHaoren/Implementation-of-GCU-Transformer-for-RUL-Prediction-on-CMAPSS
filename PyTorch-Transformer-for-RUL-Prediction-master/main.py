import torch
torch.manual_seed(1)
from data_process.data_processing2 import data_processing as data_processing2
from visualize import *
from data_process.loader import *
from torch.utils.data import DataLoader
from train import *
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
from itertools import cycle
from model import *

if __name__ == "__main__":
   
    #------PARAMETER DEFINITION------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD001', help='which dataset to run')
    parser.add_argument('--modes', type=str, default='pretrain', help='pretrain or train or test')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--epoch', type=int, default=20, help='epoch to train')
    parser.add_argument('--dim_en', type=int, default=128, help='dimension in encoder')
    parser.add_argument('--head', type=int, default=4, help='number of heads in multi-head attention')
    parser.add_argument('--num_enc_layers', type=int, default=1, help='number of encoder layers')
    parser.add_argument('--num_features', type=int, default=14, help='number of features')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--train_seq_len', type=int, default=20, help='train_seq_len')
    parser.add_argument('--test_seq_len', type=int, default=20, help='test_seq_len')
    parser.add_argument('--slices', type=int, default=10, help='split_slices')
    parser.add_argument('--LR', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--save_path', type=str, default='', help='log save path')
    parser.add_argument('--patch_size', type=int, default=3, help='length of patch')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='length of patch')
    parser.add_argument('--decay_step', type=float, default=3, help='length of patch')
    parser.add_argument('--decay_ratio', type=float, default=0.5, help='length of patch')
    parser.add_argument('--smooth_param', type=float, default=0.8, help='none or freq')



    opt = parser.parse_args()

    if opt.modes == "Train":
        Training(opt)
    elif opt.modes == "test":
        PATH = opt.path
        print(PATH)
        group_train, y_test, group_test, X_test = data_processing2(opt.dataset,opt.smooth_param)
     
        test_dataset = SequenceDataset(mode='test',group = group_test, y_label=y_test, sequence_train=opt.train_seq_len, patch_size=opt.train_seq_len)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        model = GCU_Transformer(seq_size=opt.train_seq_len, patch_size=opt.patch_size, in_chans=opt.num_features,
                 embed_dim=opt.dim_en, depth=opt.num_enc_layers, num_heads=opt.head,
                 decoder_embed_dim=opt.dim_en, decoder_depth=opt.num_enc_layers, decoder_num_heads=opt.head,
                 norm_layer=nn.LayerNorm,batch_size = opt.batch_size)

        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        model.load_state_dict(torch.load(PATH))
        criterion = torch.nn.MSELoss()
        if torch.cuda.is_available():
            model = model.to(device)

        model.eval()
        result=[]

        with torch.no_grad():
            test_epoch_loss = 0
            for X,y in test_loader:
                if torch.cuda.is_available():
                    X=X.cuda()
                    y=y.cuda()
            
                y_hat_recons = model.forward(X)
           
                y_hat_unscale = y_hat_recons*125
                result.append(y_hat_unscale.item())
         
                loss = criterion(y_hat_unscale.reshape(y_hat_recons.shape[0]),y)
                test_epoch_loss = test_epoch_loss + loss   

            test_loss = torch.sqrt(test_epoch_loss/len(test_loader))

        
        print("testing rmse: %1.5f" % (test_loss))
        
        y_test.index = y_test.index
        result = y_test.join(pd.DataFrame(result))

        result = result.sort_values('RUL', ascending=False)
        result['RUL'].clip(upper=125, inplace=True)
        #visualize the testing result
        visualize(result, test_loss)