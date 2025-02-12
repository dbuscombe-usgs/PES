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

## https://github.com/TsLu1s/MLimputer
from mlimputer.imputation import MLimputer
import mlimputer.model_selection as ms
from mlimputer.parameters import imputer_parameters

import warnings
warnings.filterwarnings("ignore", category=Warning) #-> For a clean console
import pickle 
# import xarray as xr 

import mlimputer.evaluation as Evaluator                   
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor


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
for file in rbackfiles:

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
# prop_missing = 0.25

dataset = 'rBack'

rmse_test_vec =[]
r2_test_vec = []
prop_missing_vec = []
mod_vec = []
train_size_vec = []

for train_size in [0.5,0.6,0.8]:

    for prop_missing in [0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4,0.5]:

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

        # dates = mdates.date2num(time)

        # plt.figure(figsize=(24,18))
        # ax = plt.subplot(211)
        # plt.plot(dates, concat_data_0['rBack1_978'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('rBack1_978')
        # plt.xlabel('Date')
        # plt.title('rBack data')

        # ax = plt.subplot(212)
        # plt.plot(dates, concat_data_0['u_1205'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('u_1205')
        # plt.xlabel('Date')
        # plt.tight_layout()  # Adjust layout 

        # # plt.show()
        # plt.savefig(f'rBack_datasets.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # train_size = 0.8

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

        np.sum(np.isnan(train))/len(train)
        np.sum(np.isnan(test))/len(test)


        # plt.figure(figsize=(24,18))
        # ax = plt.subplot(211)
        # dates = mdates.date2num(train_time)
        # plt.plot(dates, train_scaled['rBack1_978'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('rBack1_978')
        # plt.xlabel('Date')
        # plt.title('Train')

        # ax = plt.subplot(212)
        # dates = mdates.date2num(test_time)
        # plt.plot(dates, test_scaled['rBack1_978'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('rBack1_978')
        # plt.xlabel('Date')
        # plt.title('Test')
        # plt.tight_layout()  # Adjust layout 

        # # plt.show()
        # plt.savefig(f'rBack_datasets_train_test.png', dpi=300, bbox_inches='tight')
        # plt.close()


        # All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"
        # Customizing Hyperparameters Example
        hparameters = imputer_parameters()
        print(hparameters)
        hparameters["KNN"]["n_neighbors"] = 15 #5
        hparameters["RandomForest"]["n_estimators"] = 30
        hparameters["ExtraTrees"]["n_estimators"] = 30 #15
        hparameters["GBR"]["n_estimators"] = 30 #15
        hparameters["Lightgbm"]["learning_rate"] = 0.001 #0.01
        hparameters["Catboost"]["loss_function"] = "MAE"

        # Imputation 1 : KNN
        mli_knn = MLimputer(imput_model = "KNN", imputer_configs = hparameters)
        mli_knn.fit_imput(X = train_scaled)
        train_knn = mli_knn.transform_imput(X = train_scaled)
        test_knn = mli_knn.transform_imput(X = test_scaled)

        # Imputation 2 : RandomForest
        mli_rf = MLimputer(imput_model = "RandomForest", imputer_configs = hparameters)
        mli_rf.fit_imput(X = train_scaled)
        train_rf = mli_rf.transform_imput(X = train_scaled)
        test_rf = mli_rf.transform_imput(X = test_scaled)

        # Imputation 3 : Extratrees
        mli_et = MLimputer(imput_model = "ExtraTrees", imputer_configs = hparameters)
        mli_et.fit_imput(X = train_scaled)
        train_et = mli_et.transform_imput(X = train_scaled)
        test_et = mli_et.transform_imput(X = test_scaled)

        # Imputation 4 : GBR
        mli_gbr = MLimputer(imput_model = "GBR", imputer_configs = hparameters)
        mli_gbr.fit_imput(X = train_scaled)
        train_gbr = mli_gbr.transform_imput(X = train_scaled)
        test_gbr = mli_gbr.transform_imput(X = test_scaled)

        # Imputation 5 : lightGBR
        mli_lgbr = MLimputer(imput_model = "Lightgbm", imputer_configs = hparameters)
        mli_lgbr.fit_imput(X = train_scaled)
        train_lgbr = mli_lgbr.transform_imput(X = train_scaled)
        test_lgbr = mli_lgbr.transform_imput(X = test_scaled)

        # Imputation 6 : catboost
        mli_cb = MLimputer(imput_model = "Catboost", imputer_configs = hparameters)
        mli_cb.fit_imput(X = train_scaled)
        train_cb = mli_cb.transform_imput(X = train_scaled)
        test_cb = mli_cb.transform_imput(X = test_scaled)

        ## use all models on all data
        tmp = concat_data_scaled.reset_index(drop = True) #data

        all_knn = mli_knn.transform_imput(tmp)
        all_rf = mli_rf.transform_imput(tmp)
        all_et = mli_et.transform_imput(tmp)
        all_gbr = mli_gbr.transform_imput(tmp)
        all_lgbr = mli_lgbr.transform_imput(tmp)
        all_cb = mli_cb.transform_imput(tmp)


        ###=============================================
        ind = np.where(np.isnan(concat_data_scaled[target]))[0]
        dates = mdates.date2num(time)

        sstart = 500
        send = 750
        R = []

        for mod in ['rf','knn','cb','lgbr','gbr','et']:
            print(mod)
            rmse = np.nan
            r2 = np.nan
            mod_vec.append(mod)
            prop_missing_vec.append(prop_missing)
            train_size_vec.append(train_size)
            
            plt.figure(figsize=(18,12))
            ax = plt.subplot(231)
            plt.plot(dates, concat_data0_scaled[target],'k.', alpha=0.5, lw=1, label='Missing') #data_0
            plt.plot(dates, concat_data_scaled[target],'m.',  lw=3, label='Observed') #data
            if mod=='rf':
                plt.plot(dates, all_rf[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'knn':
                plt.plot(dates, all_knn[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'cb':
                plt.plot(dates, all_cb[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'et':
                plt.plot(dates, all_et[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'gbr':
                plt.plot(dates, all_gbr[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'lgbr':
                plt.plot(dates, all_lgbr[target],'g.', alpha=0.5, lw=1, label='Imputed')

            plt.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
            plt.xticks(rotation=45)  # Rotate labels for better readability
            plt.ylabel(target)
            plt.xlabel('Date')

            ax = plt.subplot(232)
            plt.plot(dates[sstart:send], concat_data0_scaled[target][sstart:send],'k', lw=1, alpha=0.5, label='Missing') #data_0
            plt.plot(dates[sstart:send], concat_data_scaled[target][sstart:send],'m', lw=3, label='Observed') #data
            if mod=='rf':
                plt.plot(dates[sstart:send], all_rf[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'knn':
                plt.plot(dates[sstart:send], all_knn[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'cb':
                plt.plot(dates[sstart:send], all_cb[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'et':
                plt.plot(dates[sstart:send], all_et[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'gbr':
                plt.plot(dates[sstart:send], all_gbr[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'lgbr':
                plt.plot(dates[sstart:send], all_lgbr[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')

            plt.legend()
            ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Format the date display
            plt.xticks(rotation=45)  # Rotate labels for better readability
            plt.ylabel(target)
            plt.xlabel('Day')

            ind = np.where(np.isnan(test_scaled[target]))[0]
            # tind = np.where(np.isnan(train_scaled[target]))[0]

            y_target = test0_scaled[target][ind]

            ax=plt.subplot(233)
            if mod=='rf':
                plt.plot( test0_scaled[target][ind],  test_rf[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_rf[target][tind], 'r.', label='Est (train)')
                y_est = test_rf[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'cb':
                plt.plot( test0_scaled[target][ind],  test_cb[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_cb[target][tind], 'r.', label='Est (train)')
                y_est = test_cb[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'et':
                plt.plot( test0_scaled[target][ind],  test_et[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_et[target][tind], 'r.', label='Est (train)')
                y_est = test_et[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'gbr':
                plt.plot( test0_scaled[target][ind],  test_gbr[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_gbr[target][tind], 'r.', label='Est (train)')
                y_est = test_gbr[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'lgbr':
                plt.plot( test0_scaled[target][ind],  test_lgbr[target][ind], 'k.',label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_lgbr[target][tind], 'r.', label='Est (train)')
                y_est = test_lgbr[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'knn':
                plt.plot( test0_scaled[target][ind],  test_knn[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_knn[target][tind], 'r.', label='Est (train)')
                y_est = test_knn[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)


            plt.ylabel(f'Estimated {target}')
            plt.xlabel(f'Observed {target}')

            plt.legend()

            rmse_test_vec.append(rmse)
            r2_test_vec.append(r2)
            
            xlim = plt.xlim()
            plt.plot(xlim,xlim,'r--')
            plt.tight_layout()  # Adjust layout 
            # plt.show()
            plt.savefig(f'{dataset}_{train_size}_impute_{mod}_{prop_missing}_{rmstr}.png', dpi=300, bbox_inches='tight')
            plt.close()


    dict_df = {}
    dict_df['rmse_test_vec'] = rmse_test_vec
    dict_df['prop_missing_vec'] = prop_missing_vec
    dict_df['r2_test_vec'] = r2_test_vec
    dict_df['mod_vec'] = mod_vec
    dict_df['train_size_vec'] = train_size_vec

    df = pd.DataFrame.from_dict(dict_df)

    df.to_csv(f'{dataset}_results.csv')





D=[]
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
# prop_missing = 0.25

dataset = 'NEP'

rmse_test_vec =[]
r2_test_vec = []
prop_missing_vec = []
mod_vec = []
train_size_vec = []

for train_size in [0.5,0.6,0.8]:

    for prop_missing in [0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4,0.5]:

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

        # dates = mdates.date2num(time)

        # plt.figure(figsize=(24,18))
        # ax = plt.subplot(211)
        # plt.plot(dates, concat_data_0['NEP1_56'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('NEP1_56')
        # plt.xlabel('Date')
        # plt.title('rBack data')

        # ax = plt.subplot(212)
        # plt.plot(dates, concat_data_0['u_1205'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('u_1205')
        # plt.xlabel('Date')
        # plt.tight_layout()  # Adjust layout 

        # # plt.show()
        # plt.savefig(f'NEP_datasets.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # train_size = 0.8

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

        np.sum(np.isnan(train))/len(train)
        np.sum(np.isnan(test))/len(test)


        # plt.figure(figsize=(24,18))
        # ax = plt.subplot(211)
        # dates = mdates.date2num(train_time)
        # plt.plot(dates, train_scaled['NEP1_56'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('NEP1_56')
        # plt.xlabel('Date')
        # plt.title('Train')

        # ax = plt.subplot(212)
        # dates = mdates.date2num(test_time)
        # plt.plot(dates, test_scaled['NEP1_56'],'.')
        # ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
        # plt.xticks(rotation=45)  # Rotate labels for better readability
        # plt.ylabel('NEP1_56')
        # plt.xlabel('Date')
        # plt.title('Test')
        # plt.tight_layout()  # Adjust layout 

        # # plt.show()
        # plt.savefig(f'NEP_datasets_train_test.png', dpi=300, bbox_inches='tight')
        # plt.close()


        # All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"
        # Customizing Hyperparameters Example
        hparameters = imputer_parameters()
        print(hparameters)
        hparameters["KNN"]["n_neighbors"] = 15 #5
        hparameters["RandomForest"]["n_estimators"] = 30
        hparameters["ExtraTrees"]["n_estimators"] = 30 #15
        hparameters["GBR"]["n_estimators"] = 30 #15
        hparameters["Lightgbm"]["learning_rate"] = 0.001 #0.01
        hparameters["Catboost"]["loss_function"] = "MAE"

        # Imputation 1 : KNN
        mli_knn = MLimputer(imput_model = "KNN", imputer_configs = hparameters)
        mli_knn.fit_imput(X = train_scaled)
        train_knn = mli_knn.transform_imput(X = train_scaled)
        test_knn = mli_knn.transform_imput(X = test_scaled)

        # Imputation 2 : RandomForest
        mli_rf = MLimputer(imput_model = "RandomForest", imputer_configs = hparameters)
        mli_rf.fit_imput(X = train_scaled)
        train_rf = mli_rf.transform_imput(X = train_scaled)
        test_rf = mli_rf.transform_imput(X = test_scaled)

        # Imputation 3 : Extratrees
        mli_et = MLimputer(imput_model = "ExtraTrees", imputer_configs = hparameters)
        mli_et.fit_imput(X = train_scaled)
        train_et = mli_et.transform_imput(X = train_scaled)
        test_et = mli_et.transform_imput(X = test_scaled)

        # Imputation 4 : GBR
        mli_gbr = MLimputer(imput_model = "GBR", imputer_configs = hparameters)
        mli_gbr.fit_imput(X = train_scaled)
        train_gbr = mli_gbr.transform_imput(X = train_scaled)
        test_gbr = mli_gbr.transform_imput(X = test_scaled)

        # Imputation 5 : lightGBR
        mli_lgbr = MLimputer(imput_model = "Lightgbm", imputer_configs = hparameters)
        mli_lgbr.fit_imput(X = train_scaled)
        train_lgbr = mli_lgbr.transform_imput(X = train_scaled)
        test_lgbr = mli_lgbr.transform_imput(X = test_scaled)

        # Imputation 6 : catboost
        mli_cb = MLimputer(imput_model = "Catboost", imputer_configs = hparameters)
        mli_cb.fit_imput(X = train_scaled)
        train_cb = mli_cb.transform_imput(X = train_scaled)
        test_cb = mli_cb.transform_imput(X = test_scaled)

        ## use all models on all data
        tmp = concat_data_scaled.reset_index(drop = True) #data

        all_knn = mli_knn.transform_imput(tmp)
        all_rf = mli_rf.transform_imput(tmp)
        all_et = mli_et.transform_imput(tmp)
        all_gbr = mli_gbr.transform_imput(tmp)
        all_lgbr = mli_lgbr.transform_imput(tmp)
        all_cb = mli_cb.transform_imput(tmp)


        ###=============================================
        ind = np.where(np.isnan(concat_data_scaled[target]))[0]
        dates = mdates.date2num(time)

        sstart = 500
        send = 750
        R = []

        for mod in ['rf','knn','cb','lgbr','gbr','et']:
            print(mod)
            rmse = np.nan
            r2 = np.nan
            mod_vec.append(mod)
            prop_missing_vec.append(prop_missing)
            train_size_vec.append(train_size)
            
            plt.figure(figsize=(18,12))
            ax = plt.subplot(231)
            plt.plot(dates, concat_data0_scaled[target],'k.', alpha=0.5, lw=1, label='Missing') #data_0
            plt.plot(dates, concat_data_scaled[target],'m.',  lw=3, label='Observed') #data
            if mod=='rf':
                plt.plot(dates, all_rf[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'knn':
                plt.plot(dates, all_knn[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'cb':
                plt.plot(dates, all_cb[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'et':
                plt.plot(dates, all_et[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'gbr':
                plt.plot(dates, all_gbr[target],'g.', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'lgbr':
                plt.plot(dates, all_lgbr[target],'g.', alpha=0.5, lw=1, label='Imputed')

            plt.legend()
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
            plt.xticks(rotation=45)  # Rotate labels for better readability
            plt.ylabel(target)
            plt.xlabel('Date')

            ax = plt.subplot(232)
            plt.plot(dates[sstart:send], concat_data0_scaled[target][sstart:send],'k', lw=1, alpha=0.5, label='Missing') #data_0
            plt.plot(dates[sstart:send], concat_data_scaled[target][sstart:send],'m', lw=3, label='Observed') #data
            if mod=='rf':
                plt.plot(dates[sstart:send], all_rf[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'knn':
                plt.plot(dates[sstart:send], all_knn[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'cb':
                plt.plot(dates[sstart:send], all_cb[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'et':
                plt.plot(dates[sstart:send], all_et[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'gbr':
                plt.plot(dates[sstart:send], all_gbr[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
            elif mod == 'lgbr':
                plt.plot(dates[sstart:send], all_lgbr[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')

            plt.legend()
            ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Format the date display
            plt.xticks(rotation=45)  # Rotate labels for better readability
            plt.ylabel(target)
            plt.xlabel('Day')

            ind = np.where(np.isnan(test_scaled[target]))[0]
            # tind = np.where(np.isnan(train_scaled[target]))[0]

            y_target = test0_scaled[target][ind]

            ax=plt.subplot(233)
            if mod=='rf':
                plt.plot( test0_scaled[target][ind],  test_rf[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_rf[target][tind], 'r.', label='Est (train)')
                y_est = test_rf[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'cb':
                plt.plot( test0_scaled[target][ind],  test_cb[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_cb[target][tind], 'r.', label='Est (train)')
                y_est = test_cb[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'et':
                plt.plot( test0_scaled[target][ind],  test_et[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_et[target][tind], 'r.', label='Est (train)')
                y_est = test_et[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'gbr':
                plt.plot( test0_scaled[target][ind],  test_gbr[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_gbr[target][tind], 'r.', label='Est (train)')
                y_est = test_gbr[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'lgbr':
                plt.plot( test0_scaled[target][ind],  test_lgbr[target][ind], 'k.',label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_lgbr[target][tind], 'r.', label='Est (train)')
                y_est = test_lgbr[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
            elif mod == 'knn':
                plt.plot( test0_scaled[target][ind],  test_knn[target][ind], 'k.', label='Test set')
                # plt.loglog( train0_scaled[target][tind],  train_knn[target][tind], 'r.', label='Est (train)')
                y_est = test_knn[target][ind]
                rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                rmstr = str(rmse)[:5]
                plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                r2str = str(r2)[:5]
                plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)


            plt.ylabel(f'Estimated {target}')
            plt.xlabel(f'Observed {target}')

            plt.legend()

            rmse_test_vec.append(rmse)
            r2_test_vec.append(r2)
            
            xlim = plt.xlim()
            plt.plot(xlim,xlim,'r--')
            plt.tight_layout()  # Adjust layout 
            # plt.show()
            plt.savefig(f'{dataset}_{train_size}_impute_{mod}_{prop_missing}_{rmstr}.png', dpi=300, bbox_inches='tight')
            plt.close()


    dict_df = {}
    dict_df['rmse_test_vec'] = rmse_test_vec
    dict_df['prop_missing_vec'] = prop_missing_vec
    dict_df['r2_test_vec'] = r2_test_vec
    dict_df['mod_vec'] = mod_vec
    dict_df['train_size_vec'] = train_size_vec

    df = pd.DataFrame.from_dict(dict_df)

    df.to_csv(f'{dataset}_results.csv')








########################################

for file in files:

    rmse_test_vec =[]
    r2_test_vec = []
    prop_missing_vec = []
    mod_vec = []
    train_size_vec = []

    for train_size in [0.5,0.6,0.8]:

        for prop_missing in [0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35,0.4,0.5]:

            data = pd.read_csv(file) # Dataframe Loading Example

            dataset = file.split(os.sep)[-1].split('.csv')[0]
            target = [t for t in target_variables if t in list(data.columns)][0]

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


            data.drop(['Unnamed: 0', 'w_1204'], axis=1, inplace=True)
            target = [t for t in target_variables if t in list(data.columns)][0]

            ##### Preprocessing Data
            data = data[data[target].isnull() == False]
            data = data.reset_index(drop = True)

            na_loc = data.isnull()
            data[na_loc] = np.nan

            data_0 = data.copy()
            data_0 = data_0.reset_index(drop=True)

            ## drop time columns
            # data.drop(['time'], axis=1, inplace=True)

            spike_in = spike_in_generation(data, prop_missing, target)
            data[spike_in == 1] = np.nan #data_scaled

            data.drop_duplicates(subset='time', keep='first', inplace=True)
            time = data.time.values

            data_0.drop_duplicates(subset='time', keep='first', inplace=True)

            ##make train and test sets
            train,test = train_test_split(data, train_size = train_size, random_state = 42) #data
            train,test = train.reset_index(drop = True), test.reset_index(drop = True) 
            train.isna().sum(), test.isna().sum()

            train0,test0 = train_test_split(data_0, train_size = train_size, random_state = 42) #data_0
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

            data_scaled = sc.transform(data.drop(['time'], axis=1))
            data0_scaled = sc.transform(data_0.drop(['time'], axis=1))

            train0_scaled = sc.transform(train0)
            test0_scaled = sc.transform(test0)

            train0_scaled = pd.DataFrame(train0_scaled, columns = train0.columns)
            test0_scaled = pd.DataFrame(test0_scaled, columns = train0.columns)

            train_scaled = pd.DataFrame(train_scaled, columns = train.columns)
            test_scaled = pd.DataFrame(test_scaled, columns = train.columns)

            data_scaled = pd.DataFrame(data_scaled, columns = data.drop(['time'], axis=1).columns)
            data0_scaled = pd.DataFrame(data0_scaled, columns = data.drop(['time'], axis=1).columns)

            np.sum(np.isnan(train))/len(train)
            np.sum(np.isnan(test))/len(test)



            # # Define evaluation parameters
            # imputation_models = ["RandomForest", "ExtraTrees", "GBR", "KNN",]
            #                     #"XGBoost", "Lightgbm", "Catboost"]   # List of imputation models to evaluate
            # n_splits = 3  # Number of splits for cross-validation

            # # Selected models for classification and regression
            # if train[target].dtypes == "object":                                      
            #             models = [RandomForestClassifier(), DecisionTreeClassifier()]
            # else:
            #     models = [XGBRegressor(), RandomForestRegressor()]

            # # Initialize the evaluator
            # evaluator = Evaluator(
            #     imputation_models = imputation_models,  
            #     train = train,
            #     target = target,
            #     n_splits = n_splits,     
            #     hparameters = hparameters)

            # # Perform evaluations
            # cv_results = evaluator.evaluate_imputation_models(
            #     models = models)

            # best_imputer = evaluator.get_best_imputer()  # Get best-performing imputation model

            # test_results = evaluator.evaluate_test_set(
            #     test = test,
            #     imput_model = best_imputer,
            #     models = models)

            # All model imputation options ->  "RandomForest","ExtraTrees","GBR","KNN","XGBoost","Lightgbm","Catboost"
            # Customizing Hyperparameters Example
            hparameters = imputer_parameters()
            print(hparameters)
            hparameters["KNN"]["n_neighbors"] = 15 #5
            hparameters["RandomForest"]["n_estimators"] = 30
            hparameters["ExtraTrees"]["n_estimators"] = 30 #15
            hparameters["GBR"]["n_estimators"] = 30 #15
            hparameters["Lightgbm"]["learning_rate"] = 0.001 #0.01
            hparameters["Catboost"]["loss_function"] = "MAE"

            # Imputation 1 : KNN
            mli_knn = MLimputer(imput_model = "KNN", imputer_configs = hparameters)
            mli_knn.fit_imput(X = train_scaled)
            train_knn = mli_knn.transform_imput(X = train_scaled)
            test_knn = mli_knn.transform_imput(X = test_scaled)

            # Imputation 2 : RandomForest
            mli_rf = MLimputer(imput_model = "RandomForest", imputer_configs = hparameters)
            mli_rf.fit_imput(X = train_scaled)
            train_rf = mli_rf.transform_imput(X = train_scaled)
            test_rf = mli_rf.transform_imput(X = test_scaled)

            # Imputation 3 : Extratrees
            mli_et = MLimputer(imput_model = "ExtraTrees", imputer_configs = hparameters)
            mli_et.fit_imput(X = train_scaled)
            train_et = mli_et.transform_imput(X = train_scaled)
            test_et = mli_et.transform_imput(X = test_scaled)

            # Imputation 4 : GBR
            mli_gbr = MLimputer(imput_model = "GBR", imputer_configs = hparameters)
            mli_gbr.fit_imput(X = train_scaled)
            train_gbr = mli_gbr.transform_imput(X = train_scaled)
            test_gbr = mli_gbr.transform_imput(X = test_scaled)

            # Imputation 5 : lightGBR
            mli_lgbr = MLimputer(imput_model = "Lightgbm", imputer_configs = hparameters)
            mli_lgbr.fit_imput(X = train_scaled)
            train_lgbr = mli_lgbr.transform_imput(X = train_scaled)
            test_lgbr = mli_lgbr.transform_imput(X = test_scaled)

            # Imputation 6 : catboost
            mli_cb = MLimputer(imput_model = "Catboost", imputer_configs = hparameters)
            mli_cb.fit_imput(X = train_scaled)
            train_cb = mli_cb.transform_imput(X = train_scaled)
            test_cb = mli_cb.transform_imput(X = test_scaled)

            ## use all models on all data
            tmp = data_scaled.reset_index(drop = True) #data

            all_knn = mli_knn.transform_imput(tmp)
            all_rf = mli_rf.transform_imput(tmp)
            all_et = mli_et.transform_imput(tmp)
            all_gbr = mli_gbr.transform_imput(tmp)
            all_lgbr = mli_lgbr.transform_imput(tmp)
            all_cb = mli_cb.transform_imput(tmp)


            ###=============================================
            ind = np.where(np.isnan(data_scaled[target]))[0]
            dates = mdates.date2num(time)

            sstart = 500
            send = 750
            R = []

            for mod in ['rf','knn','cb','lgbr','gbr','et']:
                print(mod)
                rmse = np.nan
                mod_vec.append(mod)
                prop_missing_vec.append(prop_missing)
                train_size_vec.append(train_size)

                plt.figure(figsize=(18,12))
                ax = plt.subplot(231)
                plt.plot(dates, data0_scaled[target],'k', alpha=0.5, lw=1, label='Missing') #data_0
                plt.plot(dates, data_scaled[target],'m',  lw=3, label='Observed') #data
                if mod=='rf':
                    plt.plot(dates, all_rf[target],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'knn':
                    plt.plot(dates, all_knn[target],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'cb':
                    plt.plot(dates, all_cb[target],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'et':
                    plt.plot(dates, all_et[target],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'gbr':
                    plt.plot(dates, all_gbr[target],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'lgbr':
                    plt.plot(dates, all_lgbr[target],'g', alpha=0.5, lw=1, label='Imputed')
            
                plt.legend()
                ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to daily
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the date display
                plt.xticks(rotation=45)  # Rotate labels for better readability
                plt.ylabel(target)
                plt.xlabel('Date')

                ax = plt.subplot(232)
                plt.plot(dates[sstart:send], data0_scaled[target][sstart:send],'k', lw=1, alpha=0.5, label='Missing') #data_0
                plt.plot(dates[sstart:send], data_scaled[target][sstart:send],'m', lw=3, label='Observed') #data
                if mod=='rf':
                    plt.plot(dates[sstart:send], all_rf[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'knn':
                    plt.plot(dates[sstart:send], all_knn[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'cb':
                    plt.plot(dates[sstart:send], all_cb[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'et':
                    plt.plot(dates[sstart:send], all_et[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'gbr':
                    plt.plot(dates[sstart:send], all_gbr[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')
                elif mod == 'lgbr':
                    plt.plot(dates[sstart:send], all_lgbr[target][sstart:send],'g', alpha=0.5, lw=1, label='Imputed')

                plt.legend()
                ax.xaxis.set_major_locator(mdates.DayLocator())  # Set major ticks to daily
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  # Format the date display
                plt.xticks(rotation=45)  # Rotate labels for better readability
                plt.ylabel(target)
                plt.xlabel('Day')

                ind = np.where(np.isnan(test_scaled[target]))[0]
                # tind = np.where(np.isnan(train_scaled[target]))[0]

                y_target = test0_scaled[target][ind]

                ax=plt.subplot(233)
                if mod=='rf':
                    plt.plot( test0_scaled[target][ind],  test_rf[target][ind], 'k.', label='Test set')
                    # plt.loglog( train0_scaled[target][tind],  train_rf[target][tind], 'r.', label='Est (train)')
                    y_est = test_rf[target][ind]
                    rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                    rmstr = str(rmse)[:5]
                    plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                    r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                    r2str = str(r2)[:5]
                    plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                elif mod == 'cb':
                    plt.plot( test0_scaled[target][ind],  test_cb[target][ind], 'k.', label='Test set')
                    # plt.loglog( train0_scaled[target][tind],  train_cb[target][tind], 'r.', label='Est (train)')
                    y_est = test_cb[target][ind]
                    rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                    rmstr = str(rmse)[:5]
                    plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                    r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                    r2str = str(r2)[:5]
                    plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                elif mod == 'et':
                    plt.plot( test0_scaled[target][ind],  test_et[target][ind], 'k.', label='Test set')
                    # plt.loglog( train0_scaled[target][tind],  train_et[target][tind], 'r.', label='Est (train)')
                    y_est = test_et[target][ind]
                    rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                    rmstr = str(rmse)[:5]
                    plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                    r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                    r2str = str(r2)[:5]
                    plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                elif mod == 'gbr':
                    plt.plot( test0_scaled[target][ind],  test_gbr[target][ind], 'k.', label='Test set')
                    # plt.loglog( train0_scaled[target][tind],  train_gbr[target][tind], 'r.', label='Est (train)')
                    y_est = test_gbr[target][ind]
                    rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                    rmstr = str(rmse)[:5]
                    plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                    r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                    r2str = str(r2)[:5]
                    plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                elif mod == 'lgbr':
                    plt.plot( test0_scaled[target][ind],  test_lgbr[target][ind], 'k.',label='Test set')
                    # plt.loglog( train0_scaled[target][tind],  train_lgbr[target][tind], 'r.', label='Est (train)')
                    y_est = test_lgbr[target][ind]
                    rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                    rmstr = str(rmse)[:5]
                    plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                    r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                    r2str = str(r2)[:5]
                    plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                elif mod == 'knn':
                    plt.plot( test0_scaled[target][ind],  test_knn[target][ind], 'k.', label='Test set')
                    # plt.loglog( train0_scaled[target][tind],  train_knn[target][tind], 'r.', label='Est (train)')
                    y_est = test_knn[target][ind]
                    rmse = np.sqrt(np.mean(np.square(y_target.values, y_est.values)))
                    rmstr = str(rmse)[:5]
                    plt.text(.8,.85,f'rmse = {rmstr}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)
                    r2 = np.min(np.corrcoef(y_target.values, y_est.values))**2
                    r2str = str(r2)[:5]
                    plt.text(.8,.9,f'R$^2$ = {r2str}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='r', fontsize=14)


                rmse_test_vec.append(rmse)
                r2_test_vec.append(r2)

                plt.ylabel(f'Estimated {target}')
                plt.xlabel(f'Observed {target}')

                plt.legend()

                xlim = plt.xlim()
                plt.plot(xlim,xlim,'r--')
                plt.tight_layout()  # Adjust layout 
                # plt.show()
                plt.savefig(f'{dataset}_{train_size}_impute_{mod}_{prop_missing}_{rmstr}.png', dpi=300, bbox_inches='tight')
                plt.close()


        dict_df = {}
        dict_df['rmse_test_vec'] = rmse_test_vec
        dict_df['prop_missing_vec'] = prop_missing_vec
        dict_df['r2_test_vec'] = r2_test_vec
        dict_df['mod_vec'] = mod_vec
        dict_df['train_size_vec'] = train_size_vec

        df = pd.DataFrame.from_dict(dict_df)

        df.to_csv(f'{dataset}_results.csv')



