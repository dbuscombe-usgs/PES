# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2025, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import os
from glob import glob
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from pypots.utils.random import set_random_seed
from pypots.utils.metrics import calc_mae, calc_rmse
from pypots.optim import Adam
from pypots.imputation import SAITS#, Transformer

import warnings
warnings.filterwarnings("ignore", category=Warning) 
import pickle 

def spike_in_generation(data, prop_missing, target=None):
    spike_in = pd.DataFrame(np.zeros_like(data), columns= data.columns)
    if target is not None:
        for column in data.columns:
            if column==target:
                subset = np.random.choice(data[column].index[data[column].notnull()], np.int64(prop_missing*len(data)), replace= False)
                spike_in.loc[subset, column] = 1
    else: #known target
        for column in data.columns:
            subset = np.random.choice(data[column].index[data[column].notnull()], np.int64(prop_missing*len(data)), replace= False)
            spike_in.loc[subset, column] = 1        
    return spike_in


####################################################
predictor_variables = ['u_1205', 'v_1206',  'P_4023'] #'w_1204',
target_variables = ['rBack_978', 'NEP1_56', 'Trb_980', 'rBack1_978']

folder = 'data_in'

files = glob(folder+os.sep+'*.csv')
print(files)

# for file in files:
#     data = pd.read_csv(file) # Dataframe Loading Example

#     dataset = file.split(os.sep)[-1].split('.csv')[0]

#     data.drop(['Unnamed: 0', 'w_1204'], axis=1, inplace=True)
#     target = [t for t in target_variables if t in list(data.columns)][0]

#     dates = mdates.date2num(data['time'])

#     ax = plt.subplot(211)
#     plt.plot(dates, data[target])
#     ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
#     plt.xticks(rotation=45)  # Rotate labels for better readability
#     plt.ylabel(target)
#     plt.xlabel('Date')
#     plt.title(dataset)

#     ax = plt.subplot(212)
#     plt.plot(dates, data['u_1205'])
#     ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
#     plt.xticks(rotation=45)  # Rotate labels for better readability
#     plt.ylabel('u_1205')
#     plt.xlabel('Date')

#     plt.show()

## CHC16M2T03adv-s, >= 2016-05-20

## SPB14N1T02adv-s, 2014-07-18 to 2014-08-07

## CSF20CHT04vec-s , not usable

## ERO20PST03vec-s, until 2020-03-27

## CSF20SC2 , not usable

## ITX11LST02adv-s, not usable

## ITX12LST02adv-s, until 2012-06-14

## ITX12LNT02adv-s, until 2012-06-13

## SPC14N1T02adv-s, until 2014-11-01

## CHC16S1T02adv-s, >1 = np.nan

## SPD15N1T03adv-s, 2015-02-24 to 2015-04-10

## SPA14N1T02adv-s, 2014-02-02 to 2014-04-18

# T=[]
T = ['rBack1_978',
 'rBack1_978',
 'rBack1_978',
 'NEP1_56',
 'rBack1_978',
 'rBack1_978',
 'rBack1_978',
 'NEP1_56',
 'NEP1_56',
 'rBack1_978',
 'NEP1_56',
 'rBack1_978',
 'rBack1_978',
 'NEP1_56',
 'rBack1_978',
 'NEP1_56',
 'NEP1_56',
 'rBack1_978',
 'rBack1_978',
 'rBack1_978']

rbackfiles = np.array(files)[np.where(np.array(T)=='rBack1_978')]
nepfiles = np.array(files)[np.where(np.array(T)=='NEP1_56')]

D=[]
# for file in rbackfiles:
for file in nepfiles:

    data = pd.read_csv(file) # Dataframe Loading Example

    dataset = file.split(os.sep)[-1].split('.csv')[0]

    data.drop(['Unnamed: 0', 'w_1204'], axis=1, inplace=True)
    target = [t for t in target_variables if t in list(data.columns)][0]
    # T.append(target)

    ##### Preprocessing Data
    data = data[data[target].isnull() == False]
    data = data.reset_index(drop = True)

    if 'CHC16M2T03adv' in dataset:
        # >= 2016-05-20
        data = data[(data['time'] > '2016-05-20')]

    elif 'SPB14N1T02adv' in dataset:
        #2014-07-18 to 2014-08-07
        data = data[(data['time'] > '2014-07-18') & (data['time'] < '2014-08-07')]

    elif 'ERO20PST03vec' in dataset:
        #until 2020-03-27
        data = data[(data['time'] < '2020-03-27')]

    elif 'ITX12LST02adv' in dataset:
        #until 2012-06-14
        data = data[(data['time'] < '2012-06-14')]

    elif 'ITX12LNT02adv' in dataset:
        #until until 2012-06-13
        data = data[(data['time'] < '2012-06-13')]

    elif 'SPC14N1T02adv' in dataset:
        #until 2014-11-01
        data = data[(data['time'] < '2014-11-01')]

    elif 'CHC16S1T02adv' in dataset:
        #>1 = np.nan
        data[target][data[target]>1] = np.nan

    elif 'SPD15N1T03adv' in dataset:
        #2015-02-24 to 2015-04-10
        data = data[(data['time'] > '2015-02-24') & (data['time'] < '2015-04-10')]

    elif 'SPA14N1T02adv' in dataset:
        #2014-02-02 to 2014-04-18
        data = data[(data['time'] > '2014-02-02') & (data['time'] < '2014-04-18')]

    D.append(data)


## drop time columns
# data.drop(['time'], axis=1, inplace=True)
# 

# dataset = 'rBack'

dataset = 'NEP'
train_size = 0.8
prop_missing = 0.25
d_model = 128
d_ffn = 128

for timesteps in [50, 100, 300]:
    show_timeseries = timesteps

    concat_data = pd.concat(D, axis=0) # concatenating along columns

    ##### Preprocessing Data
    concat_data = concat_data[concat_data[target].isnull() == False]
    concat_data = concat_data.reset_index(drop = True)

    na_loc = concat_data.isnull()
    concat_data[na_loc] = np.nan

    concat_data_0 = concat_data.copy()
    concat_data_0 = concat_data_0.reset_index(drop=True)

    spike_in = spike_in_generation(concat_data, prop_missing, target)
    concat_data[spike_in == 1] = np.nan #data_scaled

    concat_data.drop_duplicates(subset='time', keep='first', inplace=True)
    time = concat_data.time.values

    concat_data_0.drop_duplicates(subset='time', keep='first', inplace=True)

    ##make train and test sets
    train,test = train_test_split(concat_data, train_size = train_size, random_state = 42) #data
    train,test = train.reset_index(drop = True), test.reset_index(drop = True) 
    train.isna().sum(), test.isna().sum()

    train0,test0 = train_test_split(concat_data_0, train_size = train_size, random_state = 42) #data_0
    train0,test0 = train0.reset_index(drop = True), test0.reset_index(drop = True) 
    train0.isna().sum(), test0.isna().sum()

    train_time = train.time.values
    train.drop(['time'], axis=1, inplace=True)
    train0.drop(['time'], axis=1, inplace=True)

    test_time = test.time.values
    test.drop(['time'], axis=1, inplace=True)
    test0.drop(['time'], axis=1, inplace=True)


    # sc = StandardScaler()
    sc = MinMaxScaler()
    train_scaled = sc.fit_transform(train)
    test_scaled = sc.transform(test)

    concat_data_scaled = sc.transform(concat_data.drop(['time'], axis=1))
    concat_data0_scaled = sc.transform(concat_data_0.drop(['time'], axis=1))

    train0_scaled = sc.transform(train0)
    test0_scaled = sc.transform(test0)

    train0_scaled = pd.DataFrame(train0_scaled, columns = train0.columns)
    test0_scaled = pd.DataFrame(test0_scaled, columns = train0.columns)

    train_scaled = pd.DataFrame(train_scaled, columns = train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns = train.columns)

    concat_data_scaled = pd.DataFrame(concat_data_scaled, columns = data.drop(['time'], axis=1).columns)
    concat_data0_scaled = pd.DataFrame(concat_data0_scaled, columns = data.drop(['time'], axis=1).columns)

    # prep train subset
    num_features = train_scaled.shape[1]
    train_crop = int(np.floor(train_scaled.shape[0]/timesteps)*timesteps)
    train_scaled = train_scaled[:train_crop]
    train_scaled = np.array(train_scaled).reshape(int(train_crop/timesteps), timesteps, num_features)

    # train_scaled_ori = train0_scaled[:train_crop]
    # train_scaled_ori = np.array(train_scaled_ori).reshape(int(train_crop/timesteps), timesteps, num_features)

    # prep val subset
    num_features = test_scaled.shape[1]
    val_crop = int(np.floor(test_scaled.shape[0]/timesteps)*timesteps)

    # y = y[:crop]
    test_scaled = test_scaled[:val_crop]
    test_scaled = np.array(test_scaled).reshape(int(val_crop/timesteps), timesteps, num_features)

    test_scaled_ori = test0_scaled[:val_crop]
    test_scaled_ori = np.array(test_scaled_ori).reshape(int(val_crop/timesteps), timesteps, num_features)

    ##################################################
    n_features = test_scaled.shape[-1]

    dataset_for_training = {"X": train_scaled}  # X for model input
    print(train_scaled.shape)  # (113, 300, 4), 90 samples and each sample has 300 time steps, 5 features

    dataset_for_validating = {"X": test_scaled, "X_ori": test_scaled_ori}  
    # X for model input
    print(test_scaled.shape)  # (28, 300, 4), 22 samples and each sample has 300 time steps, 5 features

    ################### SAITS model

    for d_v in [64,128,256]:
        for d_k in [64,128,256]:
            for batch_size in [128,256]:
                for n_heads in [1,2,4]:#,4]:
                    for n_layers in [2,4,6]:
                        for lr in [1e-3,1e-4,1e-5]: #, 1e-5, 1e-6]:

                                # initialize the model
                                saits = SAITS(
                                    n_steps=timesteps,
                                    n_features=n_features,
                                    n_layers=n_layers,
                                    d_model=d_model,
                                    d_ffn=d_ffn,
                                    n_heads=n_heads,
                                    d_k=d_k, #128, #64,
                                    d_v=d_v, #128, #64,
                                    dropout=0.1,
                                    attn_dropout=0.1,
                                    diagonal_attention_mask=True,  # otherwise the original self-attention mechanism will be applied
                                    ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight
                                    # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
                                    MIT_weight=1,
                                    batch_size=batch_size,
                                    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
                                    epochs=1000,
                                    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
                                    # You can leave it to defualt as None to disable early stopping.
                                    patience=50,
                                    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
                                    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
                                    optimizer=Adam(lr=lr),
                                    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
                                    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
                                    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
                                    num_workers=0,
                                    # just leave it to default as None, PyPOTS will automatically assign the best device for you.
                                    # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
                                    device=None,  
                                    # set the path for saving tensorboard and trained model files 
                                    saving_path="saits",
                                    # only save the best model after training finished.
                                    # You can also set it as "better" to save models performing better ever during training.
                                    model_saving_strategy="best",
                                )

                                saits.fit(train_set=dataset_for_training, val_set=dataset_for_validating)

                                print(saits.num_params)


                                #### skill on test set
                                imputation = saits.impute(dataset_for_validating)  # impute the originally-missing values and artificially-missing values
                                indicating_mask = np.isnan(test_scaled) ^ np.isnan(test_scaled_ori)  # indicating mask for imputation error calculation
                                mae = calc_mae(imputation, np.nan_to_num(test_scaled_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
                                rmse = calc_rmse(imputation, np.nan_to_num(test_scaled_ori), indicating_mask)  # calculate rmse on the ground truth (artificially-missing values)
                                mae = str(mae)[:6]
                                print(rmse)
                                print(mae)


                                est_test = imputation.reshape(val_crop,num_features)
                                obs_test = test_scaled.reshape(val_crop,num_features)
                                mask_test = indicating_mask.reshape(val_crop,num_features)

                                est_test_OGscaling = sc.inverse_transform(est_test)
                                obs_test_OGscaling = sc.inverse_transform(obs_test)

                                r2 = np.min(np.corrcoef(est_test_OGscaling[:,-1], np.nan_to_num(obs_test_OGscaling[:,-1])))**2
                                print(r2)


                                obs_test_ori = test_scaled_ori.reshape(val_crop,num_features)
                                obs_test_ori_OGscaling = sc.inverse_transform(obs_test_ori)

                                ind = np.where(mask_test[:,-1])[0]

                                dates = mdates.date2num(test_time[:val_crop])


                                plt.close('all')
                                plt.figure(figsize=(12,12)) 

                                ax = plt.subplot(111)
                                # plt.plot(obs_test_ori_OGscaling[ind,-1], est_test_OGscaling[ind,-1], 'k.')
                                plt.plot(obs_test_ori[ind,-1], est_test[ind,-1], 'k.')
                                xl = plt.xlim()
                                plt.plot(xl,xl,'r--')
                                plt.text(.8,.8,f'mae = {mae}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                                plt.text(.8,.9,f'num model params = {saits.num_params}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)

                                # plt.show()
                                plt.savefig(f'{dataset}_SAITS_layers{n_layers}_heads{n_heads}_lr{lr}_d{d_model}_ffn{d_ffn}_batch{batch_size}_example_testset_missing{prop_missing}_ts{timesteps}.png',dpi=300,bbox_inches='tight')
                                plt.close()
                                # del saits



