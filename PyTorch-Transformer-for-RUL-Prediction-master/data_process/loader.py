from sys import set_asyncgen_hooks
from typing import Set
import torch
from torch.utils.data import Dataset
import pdb
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, mode = 'pretrain', group = None, y_label = None, sequence_train = 50, sequence_test = 20, patch_size=10):
        self.mode = mode
        X_ = []
        X_hp_ = []
        X_lp_ = []
        y_ = []
        time_stamp = []
    
        if mode == 'train':
            self.unit_nr_total = len(group["unit_nr"].value_counts())
            i=1
       
            while i <self.unit_nr_total:
                self.x = group.get_group(i).to_numpy()
   
                # split by sequences
                length_cur_unit_nr = len(self.x)
                for j in range(patch_size,length_cur_unit_nr):
                    
                    X=self.x[j-patch_size:j,2:-1]

                    X = X.astype(float) 
                    X_.append(X)
                    

                    y=self.x[j-1,-1]

                    if y >=125:
                        y_.append(125)
                    else:
                        y_.append(y)

                    time_stamp.append(j)
                i+=1

            self.y = torch.tensor(y_).float()
            self.X = torch.tensor(X_).float()
            self.toggle=0

            self.time_stamp = torch.tensor(time_stamp)


        if mode == 'val':
            self.unit_nr_total = len(group["unit_nr"].value_counts())
            i = 1

            while i <= self.unit_nr_total:
                self.x = group.get_group(i).to_numpy()

                # split by sequences
                length_cur_unit_nr = len(self.x)

    
                for j in range(patch_size,length_cur_unit_nr):   
                    X=self.x[j-patch_size:j,2:-1]
                    X_.append(X)

                    y=self.x[j-1,-1]

                    if y >=125:
                        y_.append(125)
                    else:
                        y_.append(y)

                    x_data_transpose = X.transpose(0,1)

                    time_stamp.append(j)
                i+=1
                
            self.y = torch.tensor(y_).float()
            self.X = torch.tensor(X_).float()
            self.toggle=0

            self.time_stamp = torch.tensor(time_stamp)


        elif mode == 'test':
            self.unit_nr_total = len(group["unit_nr"].value_counts())
            y_label = y_label["RUL"].to_numpy()
            i=1
            while i <= self.unit_nr_total:
                    self.x = group.get_group(i).to_numpy()

                    # split by sequences
                    length_cur_unit_nr = len(self.x)

                    if length_cur_unit_nr < patch_size:
                        data = np.zeros((patch_size, self.x.shape[1]))
                        for j in range(data.shape[1]):
                            x_old = np.linspace(0, len(self.x)-1, len(self.x), dtype=np.float64)
                            params = np.polyfit(x_old, self.x[:, j].flatten(), deg=1)
                            k = params[0]
                            b = params[1]
                            x_new = np.linspace(0, patch_size-1, patch_size, dtype=np.float64)
                            data[:, j] = (x_new * len(self.x) / patch_size * k + b)
                    else:
                        data = self.x
                    X=data[-patch_size:, 2:]
                    X_.append(X)
                    y_cur = y_label[i-1]
                    if y_cur >= 125:
                        y_.append(125)
                    else:
                        y_.append(y_cur)

                    i+=1
                
                    
            self.y = torch.tensor(y_).float()
            self.X = torch.tensor(X_).float()
            self.toggle=0
     

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        
        return self.X[i], self.y[i]