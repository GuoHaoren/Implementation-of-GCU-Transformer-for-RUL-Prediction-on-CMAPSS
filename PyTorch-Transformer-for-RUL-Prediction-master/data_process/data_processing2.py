from typing import Set
import pandas as pd
from .add_remaining_useful_life import *
import torch
import pdb
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def data_processing(data_name="FD001",smooth_param=1,exclude=False):

    # define filepath to read data
    dir_path = './CMAPSSData/'
    
    data_name1="FD001"
    data_name2="FD002"
    data_name3="FD003"
    data_name4="FD004"

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    operating_name = ['op_cond']
    col_names = index_names + setting_names + sensor_names+operating_name

    # read data
    train1 = pd.read_csv((dir_path + 'train_'+data_name1+'.txt'), sep='\s+', header=None, names=col_names)
    test1 = pd.read_csv((dir_path + 'test_'+data_name1+'.txt'), sep='\s+', header=None, names=col_names)
    y_test1 = pd.read_csv((dir_path + 'RUL_'+data_name1+'.txt'), sep='\s+', header=None, names=['RUL'])

    train2 = pd.read_csv((dir_path + 'train_'+data_name2+'.txt'), sep='\s+', header=None, names=col_names)
    test2 = pd.read_csv((dir_path + 'test_'+data_name2+'.txt'), sep='\s+', header=None, names=col_names)
    y_test2 = pd.read_csv((dir_path + 'RUL_'+data_name2+'.txt'), sep='\s+', header=None, names=['RUL'])

    train3 = pd.read_csv((dir_path + 'train_'+data_name3+'.txt'), sep='\s+', header=None, names=col_names)
    test3 = pd.read_csv((dir_path + 'test_'+data_name3+'.txt'), sep='\s+', header=None, names=col_names)
    y_test3 = pd.read_csv((dir_path + 'RUL_'+data_name3+'.txt'), sep='\s+', header=None, names=['RUL'])

    train4 = pd.read_csv((dir_path + 'train_'+data_name4+'.txt'), sep='\s+', header=None, names=col_names)
    test4 = pd.read_csv((dir_path + 'test_'+data_name4+'.txt'), sep='\s+', header=None, names=col_names)
    y_test4 = pd.read_csv((dir_path + 'RUL_'+data_name4+'.txt'), sep='\s+', header=None, names=['RUL'])

    
    print(data_name)
    if data_name=="pretrain_all":
        train = pd.concat([train1,train2,train3,train4])
        test = pd.concat([test1,test2,test3,test4])
        y_test = pd.concat([y_test1,y_test2,y_test3,y_test4])
    elif data_name=="train_other3":
        train = pd.concat([train2,train3,train4])
        test = pd.concat([test2,test3,test4])
        y_test = pd.concat([y_test2,y_test3,y_test4])
    elif data_name == "FD001":
        train = train1
        test = test1
        y_test = y_test1
    elif data_name == "FD002":
        train = train2
        test = test2
        y_test = y_test2
    elif data_name == "FD003":
        train = train3
        test = test3
        y_test = y_test3
    elif data_name == "FD004":
        train = train4
        test = test4
        y_test = y_test4
    elif data_name == "FD001andFD002":
        train = pd.concat([train2,train1])
        test = pd.concat([test2,test1])
        y_test = pd.concat([y_test2,y_test1])
    elif data_name == "FD002andFD003":
        train = pd.concat([train2,train3])
        test = pd.concat([test2,test3])
        y_test = pd.concat([y_test2,y_test3])
    elif data_name == "FD002andFD004":
        train = pd.concat([train2,train4])
        test = pd.concat([test2,test4])
        y_test = pd.concat([y_test2,y_test4])
    elif data_name == "FD003andFD004":
        train = pd.concat([train4,train3])
        test = pd.concat([test4,test3])
        y_test = pd.concat([y_test4,y_test3])
    # drop non-informative features in training set
    #sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
    sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11','s_12', 's_13', 's_14','s_15', 's_17', 's_20','s_21']
    #drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

    alpha = smooth_param
 
    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=125, inplace=True)
    train['RUL'] = train['RUL']/125
    # remove unused sensors
    drop_sensors = [element for element in sensor_names if element not in sensors]
    
    # scale with respect to the operating condition
    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))
    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    
    ori_X_train_pre=X_train_pre

    X_train_pre, X_test_pre = condition_scaler(X_train_pre, X_test_pre, sensors)

    # exponential smoothing
    X_train_pre= exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    X_train_pre.drop(labels=setting_names+operating_name, axis=1, inplace=True)
    X_test_pre.drop(labels=setting_names+operating_name, axis=1, inplace=True)

    group = X_train_pre.groupby(by="unit_nr")
    group_test = X_test_pre.groupby(by="unit_nr")
    Xtest = X_test_pre.groupby(by="unit_nr")

    if exclude==False:
        return group, y_test, group_test, Xtest
    else:

        ori_X_train_pre.drop(labels=setting_names+["op_cond","RUL"], axis=1, inplace=True)

        # separate title information and sensor data

        train_data = ori_X_train_pre.iloc[:, 2:]
        list_train_labels = list(train_data.columns.values)


        #scaler = StandardScaler()
        #scaler = MinMaxScaler()
        scaler = MinMaxScaler(feature_range=(0, 1))
        # min-max normalization of the sensor data
        #data_norm = (data - data.min()) / (data.max() - data.min())

        scaler.fit(train_data[list_train_labels])
  
        return group, y_test, group_test, Xtest,scaler