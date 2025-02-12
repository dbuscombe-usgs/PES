

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xarray as xr 

import os
from glob import glob
# import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.utils.metrics import calc_mae, calc_rmse


# from pypots.data.generating import gene_physionet2012
# from pypots.utils.random import set_random_seed
from pypots.utils.metrics import calc_mae
from pypots.optim import Adam
from pypots.imputation import SAITS, Transformer


####################################################
predictor_variables = ['u_1205', 'v_1206', 'w_1204', 'P_4023']
target_variables = ['rBack_978', 'NEP1_56', 'Trb_980', 'rBack1_978']

# folder = '/media/marda/FOURTB/PES/tinyML/data_in'
# files = glob(folder+os.sep+'*.csv')

files = glob('/media/marda/FOURTB/PES/tsfill/*.nc')
print(files)


# file = '/media/marda/FOURTB/PES/tinyML/data_in/NMB15M1T03aqd_timeseries_PES.csv'
# for file in files:
#     data = pd.read_csv(file) # Dataframe Loading Example
#     print(file)
#     print(data.shape)


#data = pd.read_csv(file) 
#data.drop(['Unnamed: 0'], axis=1, inplace=True)

file = files[1] ## this is the biggest file


dataset = file.split(os.sep)[-1].split('.nc')[0]

ds_disk = xr.open_dataset(file)
df = ds_disk.to_dataframe()
df = df.reset_index(drop=False)

dfav = df.groupby('burst').mean()

target = [t for t in target_variables if t in list(df.columns)][0]
df['time'] = pd.to_datetime(df['time'])  

data = df[['time', 'u_1205','v_1206','w_1204','P_4023', target]]

### why do I need to do this?
data.drop_duplicates(subset='time', keep='first', inplace=True)

# y = data[target].values
# sel_cols = [col for col in data.columns if col != target]
# X = data[sel_cols]

train_size = 0.5
# timesteps = 300


#######################################
###################### SAITS

doplot = False

rmse_test_vec =[]
mae_test_vec = []
r2_test_vec = []
rmse_train_vec =[]
mae_train_vec = []
r2_train_vec = []
rmse_val_vec =[]
mae_val_vec = []
r2_val_vec = []
timesteps_vec = []
prop_missing_vec = []
lr_vec = []
n_layers_vec = []
n_heads_vec = []
batch_size_vec = []
num_params_vec = []
d_k_vec = []
d_v_vec = []

for timesteps in [50,100,200]: #,200,300]:
    for prop_missing in [0.1, 0.2, 0.3]:

        show_timeseries = timesteps

        d_model = 64 #128 #256
        d_ffn = 64 #128
        # batch_size = 32

        ## shuffle=false!!!!!
        test,train = train_test_split(data, train_size = train_size, shuffle=False)
        ## split train set to get train and val
        val,train = train_test_split(train, train_size = train_size, shuffle=False)

        print(f"train size: {train.shape}")
        print(f"val size: {val.shape}")
        print(f"test size: {test.shape}")

        ## make a copy of time vector for plotting later
        test_time = test.time.values
        train_time = train.time.values
        val_time = val.time.values

        ## drop time columns
        test.drop(['time'], axis=1, inplace=True)
        train.drop(['time'], axis=1, inplace=True)
        val.drop(['time'], axis=1, inplace=True)


        print(np.nanmax(train[target]),np.nanmin(train[target]))
        print(np.nanmax(test[target]),np.nanmin(test[target]))
        print(np.nanmax(val[target]),np.nanmin(val[target]))


        train_OG = train.copy()
        test_OG = test.copy()
        val_OG = val.copy()

        sc = StandardScaler()
        train = sc.fit_transform(train)
        test = sc.transform(test)
        val = sc.transform(val)

        # prep train subset
        num_features = train.shape[1]
        train_crop = int(np.floor(train.shape[0]/timesteps)*timesteps)

        # y = y[:crop]
        train = train[:train_crop,:]
        train = train.reshape(int(train_crop/timesteps), timesteps, num_features)
        train_ori = train  # keep X_ori for validation
        train = mcar(train, prop_missing)  # randomly hold out 10% observed values as ground truth


        # prep val subset
        num_features = val.shape[1]
        val_crop = int(np.floor(val.shape[0]/timesteps)*timesteps)

        # y = y[:crop]
        val = val[:val_crop,:]
        val = val.reshape(int(val_crop/timesteps), timesteps, num_features)
        val_ori = val  # keep X_ori for validation
        val = mcar(val, prop_missing)  # randomly hold out 10% observed values as ground truth

        ####################
        # prep test subset
        num_features = test.shape[1]
        test_crop = int(np.floor(test.shape[0]/timesteps)*timesteps)

        # y = y[:crop]
        test = test[:test_crop,:]
        test = test.reshape(int(test_crop/timesteps), timesteps, num_features)
        test_ori = test  # keep X_ori for test
        test = mcar(test, prop_missing)  # randomly hold out 10% observed values as ground truth


        ##################################################
        n_features = val.shape[-1]

        dataset_for_training = {"X": train}  # X for model input
        print(train.shape)  # (90, 300, 5), 90 samples and each sample has 300 time steps, 5 features

        dataset_for_validating = {"X": val, "X_ori": val_ori}  
        # X for model input
        print(val.shape)  # (22, 300, 5), 22 samples and each sample has 300 time steps, 5 features

        dataset_for_testing = {"X": test, "X_ori": test_ori}  
        # X for model input
        print(test.shape)  # (22, 300, 5), 28 samples and each sample has 300 time steps, 5 features



        ################### SAITS model
        # n_layers = 2
        # n_heads = 4
        lr = 1e-4

        for d_v in [64,128,256]:
            for d_k in [64,128,256]:
                for batch_size in [128,256]:
                    for n_heads in [1,2,4]:#,4]:
                        for n_layers in [2,4,6]:
                            for lr in [1e-3,1e-4,1e-5]: #, 1e-5, 1e-6]:

                                ## the inner layer
                                timesteps_vec.append(timesteps)
                                prop_missing_vec.append(prop_missing)
                                lr_vec.append(lr)
                                n_layers_vec.append(n_layers)
                                n_heads_vec.append(n_heads)
                                batch_size_vec.append(batch_size)
                                d_k_vec.append(d_k)
                                d_v_vec.append(d_v)

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
                                num_params_vec.append(saits.num_params)


                                #### skill on test set
                                imputation = saits.impute(dataset_for_testing)  # impute the originally-missing values and artificially-missing values
                                indicating_mask = np.isnan(test) ^ np.isnan(test_ori)  # indicating mask for imputation error calculation
                                mae = calc_mae(imputation, np.nan_to_num(test_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
                                rmse = calc_rmse(imputation, np.nan_to_num(test_ori), indicating_mask)  # calculate rmse on the ground truth (artificially-missing values)
                                mae = str(mae)[:6]


                                est_test = imputation.reshape(test_crop,num_features)
                                obs_test = test.reshape(test_crop,num_features)
                                mask_test = indicating_mask.reshape(test_crop,num_features)

                                est_test_OGscaling = sc.inverse_transform(est_test)
                                obs_test_OGscaling = sc.inverse_transform(obs_test)

                                r2 = np.min(np.corrcoef(est_test_OGscaling[:,-1], np.nan_to_num(obs_test_OGscaling[:,-1])))**2
                                print(r2)

                                ####################
                                rmse_test_vec.append(rmse)
                                mae_test_vec.append(mae)
                                r2_test_vec.append(r2)
                                ####################


                                obs_test_ori = test_ori.reshape(test_crop,num_features)
                                obs_test_ori_OGscaling = sc.inverse_transform(obs_test_ori)

                                ind = np.where(mask_test[:,-1])[0]

                                dates = mdates.date2num(test_time[:test_crop])

                                if doplot:

                                    plt.close('all')
                                    plt.figure(figsize=(24,12))
                                    ax = plt.subplot(331)
                                    plt.plot(dates, test_OG[target].values[:test_crop],'k',lw=2,alpha=0.5)
                                    # plt.plot(dates[ind], obs_test_ori_OGscaling[ind,-1],'ko')
                                    plt.plot(dates, est_test_OGscaling[:,-1],'r.-')

                                    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
                                    plt.xticks(rotation=45)  # Rotate labels for better readability
                                    plt.tight_layout()  # Adjust layout 
                                    plt.title('Test subset')

                                    # test_OG_df_vals = sc.inverse_transform( test_OG_df.values[:test_crop][:show_timeseries])[:,-1]
                                    # tmp = np.isnan(test_OG_df_vals)

                                    ax = plt.subplot(332)
                                    plt.plot(dates[:show_timeseries], test_OG[target].values[:test_crop][:show_timeseries],'k',lw=6,alpha=0.5)
                                    # plt.plot(dates[:show_timeseries], obs_test_ori_OGscaling[:show_timeseries,-1],'b',lw=2)
                                    plt.plot(dates[:show_timeseries][ind[ind<show_timeseries]], obs_test_ori_OGscaling[:show_timeseries,-1][ind[ind<show_timeseries]],'ko')
                                    plt.plot(dates[:show_timeseries], est_test_OGscaling[:show_timeseries,-1],'r-o')

                                    ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H'))  # Format the date display
                                    plt.xticks(rotation=45)  # Rotate labels for better readability
                                    plt.tight_layout()  # Adjust layout 

                                    ax = plt.subplot(333)
                                    plt.plot(obs_test_ori_OGscaling[ind,-1], est_test_OGscaling[ind,-1], 'k.')
                                    xl = plt.xlim()
                                    plt.plot(xl,xl,'r--')
                                    plt.text(.8,.8,f'mae = {mae}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)



                                #### skill on validation set
                                imputation = saits.impute(dataset_for_validating)  # impute the originally-missing values and artificially-missing values
                                indicating_mask = np.isnan(val) ^ np.isnan(val_ori)  # indicating mask for imputation error calculation
                                mae = calc_mae(imputation, np.nan_to_num(val_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
                                rmse = calc_rmse(imputation, np.nan_to_num(val_ori), indicating_mask)  # calculate rmse on the ground truth (artificially-missing values)
                                mae = str(mae)[:6]
                                print(mae)

                                est_val = imputation.reshape(val_crop,num_features)
                                obs_val = val.reshape(val_crop,num_features)
                                mask_val = indicating_mask.reshape(val_crop,num_features)

                                est_val_OGscaling = sc.inverse_transform(est_val)
                                obs_val_OGscaling = sc.inverse_transform(obs_val)

                                r2 = np.min(np.corrcoef(est_test_OGscaling[:,-1], np.nan_to_num(obs_test_OGscaling[:,-1])))**2

                                ####################
                                rmse_val_vec.append(rmse)
                                mae_val_vec.append(mae)
                                r2_val_vec.append(r2)
                                ####################

                                obs_val_ori = val_ori.reshape(val_crop,num_features)
                                obs_val_ori_OGscaling = sc.inverse_transform(obs_val_ori)
                                ind = np.where(mask_val[:,-1])[0]

                                dates = mdates.date2num(val_time[:val_crop])

                                if doplot:
                                    # plt.figure(figsize=(18,12))
                                    ax = plt.subplot(334)

                                    plt.plot(dates, val_OG[target].values[:val_crop],'k',lw=2,alpha=0.5)
                                    plt.plot(dates, est_val_OGscaling[:,-1],'r.-')

                                    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
                                    plt.xticks(rotation=45)  # Rotate labels for better readability
                                    plt.tight_layout()  # Adjust layout 
                                    plt.title('Validation subset')

                                    ax = plt.subplot(335)
                                    plt.plot(dates[:show_timeseries], val_OG[target].values[:val_crop][:show_timeseries],'k',lw=6,alpha=0.5)
                                    plt.plot(dates[:show_timeseries], est_val_OGscaling[:show_timeseries,-1],'r.-')
                                    plt.plot(dates[:show_timeseries][ind[ind<show_timeseries]], obs_val_ori_OGscaling[:show_timeseries,-1][ind[ind<show_timeseries]],'ko')

                                    ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H'))  # Format the date display
                                    plt.xticks(rotation=45)  # Rotate labels for better readability
                                    plt.tight_layout()  # Adjust layout 

                                    ax = plt.subplot(336)

                                    plt.plot(obs_val_ori_OGscaling[ind,-1], est_val_OGscaling[ind,-1], 'k.')
                                    xl = plt.xlim()
                                    plt.plot(xl,xl,'r--')
                                    plt.text(.9,.7,f'mae = {mae}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)

                                    # plt.show()
                                    # plt.savefig(f'SAITS_example_valset_missing{prop_missing}.png',dpi=300,bbox_inches='tight')


                                #### skill on train set
                                imputation = saits.impute(dataset_for_training)  # impute the originally-missing values and artificially-missing values
                                indicating_mask = np.isnan(train) ^ np.isnan(train_ori)  # indicating mask for imputation error calculation
                                mae = calc_mae(imputation, np.nan_to_num(train_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing trainues)
                                rmse = calc_rmse(imputation, np.nan_to_num(train_ori), indicating_mask)  # calculate rmse on the ground truth (artificially-missing trainues)
                                mae = str(mae)[:6]
                                print(mae)

                                est_train = imputation.reshape(train_crop,num_features)
                                obs_train = train.reshape(train_crop,num_features)
                                mask_train = indicating_mask.reshape(train_crop,num_features)

                                est_train_OGscaling = sc.inverse_transform(est_train)
                                obs_train_OGscaling = sc.inverse_transform(obs_train)

                                r2 = np.min(np.corrcoef(est_test_OGscaling[:,-1], np.nan_to_num(obs_test_OGscaling[:,-1])))**2

                                ####################
                                rmse_train_vec.append(rmse)
                                mae_train_vec.append(mae)
                                r2_train_vec.append(r2)
                                ####################

                                obs_train_ori = train_ori.reshape(train_crop,num_features)
                                obs_train_ori_OGscaling = sc.inverse_transform(obs_train_ori)
                                ind = np.where(mask_train[:,-1])[0]

                                dates = mdates.date2num(train_time[:train_crop])

                                if doplot:

                                    # plt.figure(figsize=(18,12))
                                    ax = plt.subplot(337)
                                    plt.plot(dates, train_OG[target].values[:train_crop],'k',lw=2,alpha=0.5)
                                    plt.plot(dates, est_train_OGscaling[:,-1],'r.-')

                                    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
                                    plt.xticks(rotation=45)  # Rotate labels for better readability
                                    plt.tight_layout()  # Adjust layout 
                                    plt.title('Train subset')


                                    ax = plt.subplot(338)
                                    plt.plot(dates[:show_timeseries], train_OG[target].values[:train_crop][:show_timeseries],'k',lw=6,alpha=0.5)
                                    plt.plot(dates[:show_timeseries], est_train_OGscaling[:show_timeseries,-1],'r.-')
                                    plt.plot(dates[:show_timeseries][ind[ind<show_timeseries]], obs_train_ori_OGscaling[:show_timeseries,-1][ind[ind<show_timeseries]],'ko')

                                    ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
                                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H'))  # Format the date display
                                    plt.xticks(rotation=45)  # Rotate labels for better readability
                                    plt.tight_layout()  # Adjust layout 

                                    ax = plt.subplot(339)
                                    plt.plot(obs_train_ori_OGscaling[ind,-1], est_train_OGscaling[ind,-1], 'k.')
                                    xl = plt.xlim()
                                    plt.plot(xl,xl,'r--')
                                    plt.text(.9,.7,f'mae = {mae}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)

                                    # plt.show()
                                    plt.savefig(f'{dataset}_SAITS_NEP_layers{n_layers}_heads{n_heads}_lr{lr}_d{d_model}_ffn{d_ffn}_batch{batch_size}_example_testset_missing{prop_missing}_ts{timesteps}.png',dpi=300,bbox_inches='tight')
                                    plt.close()
                                    # del saits



colu_names = [
'rmse_test_vec',
'mae_test_vec',
'r2_test_vec',
'rmse_train_vec',
'mae_train_vec',
'r2_train_vec',
'rmse_val_vec',
'mae_val_vec',
'r2_val_vec',
'timesteps_vec',
'prop_missing_vec',
'lr_vec',
'n_layers_vec',
'n_heads_vec',
'batch_size_vec',
'num_params_vec',
'd_k_vec',
'd_v_vec'
]

dict_df = {}
dict_df['rmse_test_vec'] = rmse_test_vec
dict_df['mae_test_vec'] = mae_test_vec
dict_df['r2_test_vec'] = r2_test_vec
dict_df['rmse_train_vec'] = rmse_train_vec
dict_df['mae_train_vec'] = mae_train_vec
dict_df['r2_train_vec'] = r2_train_vec
dict_df['rmse_val_vec'] = rmse_val_vec
dict_df['mae_val_vec'] = mae_val_vec
dict_df['r2_val_vec'] = r2_val_vec
dict_df['timesteps_vec'] = timesteps_vec
dict_df['prop_missing_vec'] = prop_missing_vec
dict_df['lr_vec'] = lr_vec
dict_df['n_layers_vec'] = n_layers_vec
dict_df['n_heads_vec'] = n_heads_vec
dict_df['batch_size_vec'] = batch_size_vec
dict_df['num_params_vec'] = num_params_vec

dict_df['d_k_vec'] = d_k_vec
dict_df['d_v_vec'] = d_v_vec


df = pd.DataFrame.from_dict(dict_df)

df.to_csv(f'{dataset}_SAITS_NEP_layers_modelresults.csv')


best_settings = df.groupby('r2_test_vec').max().tail(n=1).values
########################################### best saits


prop_missing = best_settings[0][9]
timesteps, lr, n_layers, n_heads, batch_size, d_k, d_v = best_settings[0][8], best_settings[0][10], best_settings[0][11], best_settings[0][12], best_settings[0][13],  best_settings[0][15],  best_settings[0][16] 

print(f"timesteps: {timesteps}")
print(f"lr: {lr}")
print(f"n_layers: {n_layers}")
print(f"n_heads: {n_heads}")
print(f"batch_size: {batch_size}")
print(f"d_k: {d_k}")
print(f"d_v: {d_v}")



best_settings = df.groupby('mae_test_vec').min().tail(n=1).values
########################################### best saits


prop_missing = best_settings[0][9]
timesteps, lr, n_layers, n_heads, batch_size, d_k, d_v = best_settings[0][8], best_settings[0][10], best_settings[0][11], best_settings[0][12], best_settings[0][13],  best_settings[0][15],  best_settings[0][16] 

print(f"timesteps: {timesteps}")
print(f"lr: {lr}")
print(f"n_layers: {n_layers}")
print(f"n_heads: {n_heads}")
print(f"batch_size: {batch_size}")
print(f"d_k: {d_k}")
print(f"d_v: {d_v}")









## shuffle=false!!!!!
test,train = train_test_split(data, train_size = train_size, shuffle=False)
## split train set to get train and val
val,train = train_test_split(train, train_size = train_size, shuffle=False)

print(f"train size: {train.shape}")
print(f"val size: {val.shape}")
print(f"test size: {test.shape}")


## make a copy of time vector for plotting later
test_time = test.time.values
train_time = train.time.values
val_time = val.time.values

## drop time columns
test.drop(['time'], axis=1, inplace=True)
train.drop(['time'], axis=1, inplace=True)
val.drop(['time'], axis=1, inplace=True)


train_OG = train.copy()
test_OG = test.copy()
val_OG = val.copy()

sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)
val = sc.transform(val)

# prep train subset
num_features = train.shape[1]
train_crop = int(np.floor(train.shape[0]/timesteps)*timesteps)

# y = y[:crop]
train = train[:train_crop,:]
train = train.reshape(int(train_crop/timesteps), timesteps, num_features)
train_ori = train  # keep X_ori for validation
train = mcar(train, prop_missing)  # randomly hold out 10% observed values as ground truth


# prep val subset
num_features = val.shape[1]
val_crop = int(np.floor(val.shape[0]/timesteps)*timesteps)

# y = y[:crop]
val = val[:val_crop,:]
val = val.reshape(int(val_crop/timesteps), timesteps, num_features)
val_ori = val  # keep X_ori for validation
val = mcar(val, prop_missing)  # randomly hold out 10% observed values as ground truth

####################
# prep test subset
num_features = test.shape[1]
test_crop = int(np.floor(test.shape[0]/timesteps)*timesteps)

# y = y[:crop]
test = test[:test_crop,:]
test = test.reshape(int(test_crop/timesteps), timesteps, num_features)
test_ori = test  # keep X_ori for test
test = mcar(test, prop_missing)  # randomly hold out 10% observed values as ground truth


##################################################
n_features = val.shape[-1]

dataset_for_training = {"X": train}  # X for model input
print(train.shape)  # (90, 300, 5), 90 samples and each sample has 300 time steps, 5 features

dataset_for_validating = {"X": val, "X_ori": val_ori}  
# X for model input
print(val.shape)  # (22, 300, 5), 22 samples and each sample has 300 time steps, 5 features

dataset_for_testing = {"X": test, "X_ori": test_ori}  
# X for model input
print(test.shape)  # (22, 300, 5), 28 samples and each sample has 300 time steps, 5 features




# initialize the model
saits = SAITS(
    n_steps=timesteps,
    n_features=n_features,
    n_layers=n_layers,
    d_model=d_model,
    d_ffn=d_ffn,
    n_heads=n_heads,
    d_k=d_k, #64,
    d_v=d_v,
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
    patience=15,
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


#### skill on test set
imputation = saits.impute(dataset_for_testing)  # impute the originally-missing values and artificially-missing values
indicating_mask = np.isnan(test) ^ np.isnan(test_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(test_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
rmse = calc_rmse(imputation, np.nan_to_num(test_ori), indicating_mask)  # calculate rmse on the ground truth (artificially-missing values)
mae = str(mae)[:6]
print(mae)

est_test = imputation.reshape(test_crop,num_features)
obs_test = test.reshape(test_crop,num_features)
mask_test = indicating_mask.reshape(test_crop,num_features)

est_test_OGscaling = sc.inverse_transform(est_test)
obs_test_OGscaling = sc.inverse_transform(obs_test)

r2 = np.min(np.corrcoef(est_test_OGscaling[:,-1], np.nan_to_num(obs_test_OGscaling[:,-1])))**2

obs_test_ori = test_ori.reshape(test_crop,num_features)
obs_test_ori_OGscaling = sc.inverse_transform(obs_test_ori)

ind = np.where(mask_test[:,-1])[0]

dates = mdates.date2num(test_time[:test_crop])

plt.close('all')
plt.figure(figsize=(24,12))
ax = plt.subplot(331)
plt.plot(dates, test_OG[target].values[:test_crop],'k',lw=2,alpha=0.5)
plt.plot(dates, est_test_OGscaling[:,-1],'r.-',alpha=0.5)

ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout 
plt.title('Test subset')

# test_OG_df_vals = sc.inverse_transform( test_OG_df.values[:test_crop][:show_timeseries])[:,-1]
# tmp = np.isnan(test_OG_df_vals)

ax = plt.subplot(332)
plt.plot(dates[:show_timeseries], test_OG[target].values[:test_crop][:show_timeseries],'k',lw=6,alpha=0.5)
plt.plot(dates[:show_timeseries][ind[ind<=show_timeseries]], obs_test_ori_OGscaling[:show_timeseries,-1][ind[ind<=show_timeseries]],'ko')
# plt.plot(dates[:show_timeseries], obs_test_ori_OGscaling[:show_timeseries,-1],'b',lw=2)
# plt.plot(dates[:show_timeseries], obs_test_ori_OGscaling[:show_timeseries,-1],'k')
plt.plot(dates[:show_timeseries], est_test_OGscaling[:show_timeseries,-1],'r.-',alpha=0.5)

ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H'))  # Format the date display
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout 

ax = plt.subplot(333)
plt.plot(obs_test_ori_OGscaling[ind,-1], est_test_OGscaling[ind,-1], 'k.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.text(.8,.8,f'mae = {mae}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)



#### skill on validation set
imputation = saits.impute(dataset_for_validating)  # impute the originally-missing values and artificially-missing values
indicating_mask = np.isnan(val) ^ np.isnan(val_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(val_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
rmse = calc_rmse(imputation, np.nan_to_num(val_ori), indicating_mask)  # calculate rmse on the ground truth (artificially-missing values)
mae = str(mae)[:6]
print(mae)

est_val = imputation.reshape(val_crop,num_features)
obs_val = val.reshape(val_crop,num_features)
mask_val = indicating_mask.reshape(val_crop,num_features)

est_val_OGscaling = sc.inverse_transform(est_val)
obs_val_OGscaling = sc.inverse_transform(obs_val)

r2 = np.min(np.corrcoef(est_test_OGscaling[:,-1], np.nan_to_num(obs_test_OGscaling[:,-1])))**2


obs_val_ori = val_ori.reshape(val_crop,num_features)
obs_val_ori_OGscaling = sc.inverse_transform(obs_val_ori)
ind = np.where(mask_val[:,-1])[0]

dates = mdates.date2num(val_time[:val_crop])

# plt.figure(figsize=(18,12))
ax = plt.subplot(334)

plt.plot(dates, val_OG[target].values[:val_crop],'k',lw=2,alpha=0.5)
plt.plot(dates, est_val_OGscaling[:,-1],'r.-',alpha=0.5)

ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout 
plt.title('Validation subset')

ax = plt.subplot(335)
plt.plot(dates[:show_timeseries], val_OG[target].values[:val_crop][:show_timeseries],'k',lw=6,alpha=0.5)
plt.plot(dates[:show_timeseries][ind[ind<=show_timeseries]], obs_val_ori_OGscaling[:show_timeseries,-1][ind[ind<=show_timeseries]],'ko')
plt.plot(dates[:show_timeseries], est_val_OGscaling[:show_timeseries,-1],'r.-',alpha=0.5)

ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H'))  # Format the date display
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout 

ax = plt.subplot(336)

plt.plot(obs_val_ori_OGscaling[ind,-1], est_val_OGscaling[ind,-1], 'k.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.text(.9,.7,f'mae = {mae}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)

# plt.show()
# plt.savefig(f'SAITS_example_valset_missing{prop_missing}.png',dpi=300,bbox_inches='tight')


#### skill on train set
imputation = saits.impute(dataset_for_training)  # impute the originally-missing values and artificially-missing values
indicating_mask = np.isnan(train) ^ np.isnan(train_ori)  # indicating mask for imputation error calculation
mae = calc_mae(imputation, np.nan_to_num(train_ori), indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing trainues)
rmse = calc_rmse(imputation, np.nan_to_num(train_ori), indicating_mask)  # calculate rmse on the ground truth (artificially-missing trainues)
mae = str(mae)[:6]
print(mae)

est_train = imputation.reshape(train_crop,num_features)
obs_train = train.reshape(train_crop,num_features)
mask_train = indicating_mask.reshape(train_crop,num_features)

est_train_OGscaling = sc.inverse_transform(est_train)
obs_train_OGscaling = sc.inverse_transform(obs_train)

r2 = np.min(np.corrcoef(est_test_OGscaling[:,-1], np.nan_to_num(obs_test_OGscaling[:,-1])))**2


obs_train_ori = train_ori.reshape(train_crop,num_features)
obs_train_ori_OGscaling = sc.inverse_transform(obs_train_ori)
ind = np.where(mask_train[:,-1])[0]

dates = mdates.date2num(train_time[:train_crop])

# plt.figure(figsize=(18,12))
ax = plt.subplot(337)
plt.plot(dates, train_OG[target].values[:train_crop],'k',lw=2,alpha=0.5)
plt.plot(dates, est_train_OGscaling[:,-1],'r.-',alpha=0.5)

ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout 
plt.title('Train subset')


ax = plt.subplot(338)
plt.plot(dates[:show_timeseries], train_OG[target].values[:train_crop][:show_timeseries],'k',lw=6,alpha=0.5)
plt.plot(dates[:show_timeseries][ind[ind<=show_timeseries]], obs_train_ori_OGscaling[:show_timeseries,-1][ind[ind<=show_timeseries]],'ko')
plt.plot(dates[:show_timeseries], est_train_OGscaling[:show_timeseries,-1],'r.-',alpha=0.5)

ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d-%H'))  # Format the date display
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.tight_layout()  # Adjust layout 

ax = plt.subplot(339)
plt.plot(obs_train_ori_OGscaling[ind,-1], est_train_OGscaling[ind,-1], 'k.')
xl = plt.xlim()
plt.plot(xl,xl,'r--')
plt.text(.9,.7,f'mae = {mae}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)

# plt.show()
plt.savefig(f'{dataset}_SAITS_bestmodel_NEP_layers{n_layers}_heads{n_heads}_lr{lr}_d{d_model}_ffn{d_ffn}_batch{batch_size}_example_testset_missing{prop_missing}_ts{timesteps}.png',dpi=300,bbox_inches='tight')
plt.close()
# del saits
