__author__ = "Puri Phakmongkol"
__author_email__ = "me@puri.in.th"

"""
* HII Operation Script
*
* Created date : 09/07/2022
*
+      o     +              o
    +             o     +       +
o          +
    o  +           +        +
+        o     o       +        o
-_-_-_-_-_-_-_,------,      o
_-_-_-_-_-_-_-|   /\_/\
-_-_-_-_-_-_-~|__( ^ .^)  +     +
_-_-_-_-_-_-_-""  ""
+      o         o   +       o
    +         +
o      o  _-_-_-_- Station Modeling
    o           +
+      +     o        o      +
""" 

import pandas as pd
import numpy as np

# %config InlineBackend.figure_format = 'svg'

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(6, 4)})
sns.set(rc={'figure.dpi': 230})

import yaml
from yaml.loader import SafeLoader

from pathlib import Path
import datetime
import os
import json

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
import pickle
from scipy.stats import pearsonr

config = yaml.load(
    open('config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

num_to_month = {
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'APR',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC'
}

def find_rbias(prediction, groundtruth) :
    return (np.sum(prediction-groundtruth)/np.sum(groundtruth)) * 100

"""
* Station Prediction Model
https://colab.research.google.com/drive/1C_UI_AyyRtdFYSMAxT8vFgzTyBYpRIWl#scrollTo=G-6iy_9ol21N
"""
def build_model() :
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(12, 6)))
    model.add(tf.keras.layers.GRU(64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(12))

    return model

if __name__ == '__main__' :
    """
    * Load Station Data
    """
    station_all_df = pd.read_csv(f'{project_path}/data/static/tmd_station.csv')
    
    """
    * Load SimIDX Plus data
    """
    if config['simidx_last_update'] == '' or config['simidx_last_update'] == None or 'simidx_last_update' not in config :
        print('There are no indices data')
        exit()
    
    s_selected_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV','DEC']
    group_result_filtered = json.loads(open(Path(project_path) / 'output' / config['simidx_last_update'] / 'simidx_plus' / 'result_filter.json', 'r').read())
    group_result_filtered = { int(k): v for k, v in group_result_filtered.items() }
    
    s_df = pd.read_csv(Path(project_path) / 'output' / config['simidx_last_update'] / 'simidx_plus' / 's_df.csv')
    s_df = s_df.dropna()

    """
    * Station Modeling
    """
    if not Path(f'{project_path}output/{date_path}').exists() :
        os.mkdir(Path(f'{project_path}output/{date_path}'))
        
    if not Path(f'{project_path}output/{date_path}/station_model').exists() :
        os.mkdir(Path(f'{project_path}output/{date_path}/station_model'))

    b_selected_columns = ['JANp', 'FEBp',
       'MARp', 'APRp', 'MAYp', 'JUNp', 'JULp', 'AUGp', 'SEPp', 'OCTp', 'NOVp',
       'DECp']

    basin_result = []
    validate_to = -7

    for basin in [ _ for _ in station_all_df.columns if _ not in ['Unnamed: 0', 'datetime', 'year', 'month'] ][:1] :
        if not Path(f'{project_path}output/{date_path}/station_model/{basin}').exists() :
            os.mkdir(Path(f'{project_path}output/{date_path}/station_model/{basin}'))

        station_result_path = Path(f'{project_path}output/{date_path}/station_model/{basin}')

        basin_df = station_all_df[['year', 'month', basin]]
        basin_df['month_eng'] = basin_df['month'].apply(lambda _ : num_to_month[_])
        basin_df_piv = basin_df.pivot(index='year', columns='month_eng')[basin]
        basin_df_piv = basin_df_piv[num_to_month.values()]
        basin_df_piv['annual_rainfall'] = basin_df_piv.apply(np.sum, axis=1)
        basin_df_piv = basin_df_piv.reset_index()
        basin_df_piv['th_year'] = basin_df_piv['year'] + 543
        # print(basin_df_piv)
        # basin_model_df_s = basin_df_piv.iloc[12:-2, :]
        basin_model_df_s = basin_df_piv.join(s_df[['f_thyear', 'label_th']].set_index('f_thyear'), on='th_year', how='inner')
        basin_model_df_s = basin_model_df_s.fillna(0)
        print('basin = ',basin)
        # print(basin_model_df_s.info()
        # y_basinrainfall = basin_model_df_s[s_selected_columns]

        ts_df = s_df.copy()
        ts_df['f_thyear_pair'] = ts_df['f_thyear'].apply(lambda _: group_result_filtered[_])
        ts_df = ts_df.join(basin_model_df_s.set_index('th_year'), 'f_thyear_pair', rsuffix='p')
        ts_df = ts_df.dropna()
        ts_df.to_pickle(station_result_path / 'ts_df.bin')

        y_basinrainfall = basin_model_df_s.join(ts_df[['f_thyear', 'annual_rainfall']].set_index('f_thyear'), on='th_year', rsuffix='x').dropna()[s_selected_columns]
        # ts_df.to_pickle(f'{param_project_path}%s_station_%s_tsdf.bin'%(param_training_name, basin))

        X_station_pair = ts_df[b_selected_columns]
        X_allrainfall = ts_df[s_selected_columns]

        train_year = s_df['f_thyear'].iloc[:validate_to]
        validate_year = s_df['f_thyear'].iloc[validate_to:]

        overall_scaler = StandardScaler()
        X_allrainfall_train = X_allrainfall.iloc[:validate_to, :]
        X_allrainfall_validate = X_allrainfall.iloc[validate_to:, :]
        X_allrainfall_i_train = overall_scaler.fit_transform(X_allrainfall_train)
        X_allrainfall_i_validate = overall_scaler.transform(X_allrainfall_validate)

        station_pair_scaler = StandardScaler()
        X_station_pair_train = X_station_pair.iloc[:validate_to, :]
        X_station_pair_validate = X_station_pair.iloc[validate_to:, :]
        X_station_pair_i_train = station_pair_scaler.fit_transform(X_station_pair_train)
        X_station_pair_i_validate = station_pair_scaler.transform(X_station_pair_validate)

        iod_u_df = ts_df[['iod_l01', 'iod_l02', 'iod_l03', 'iod_l04', 'iod_l05',
        'iod_l06', 'iod_l07', 'iod_l08', 'iod_l09', 'iod_l10', 'iod_l11',
        'iod_l12']]
        pdo_u_df = ts_df[['pdo_l01', 'pdo_l02', 'pdo_l03', 'pdo_l04', 'pdo_l05',
        'pdo_l06', 'pdo_l07', 'pdo_l08', 'pdo_l09', 'pdo_l10', 'pdo_l11',
        'pdo_l12']]
        oni_u_df = ts_df[['oni_l01', 'oni_l02', 'oni_l03', 'oni_l04', 'oni_l05',
        'oni_l06', 'oni_l07', 'oni_l08', 'oni_l09', 'oni_l10', 'oni_l11',
        'oni_l12']]
        emi_u_df = ts_df[['emi_l01', 'emi_l02', 'emi_l03', 'emi_l04', 'emi_l05',
        'emi_l06', 'emi_l07', 'emi_l08', 'emi_l09', 'emi_l10', 'emi_l11',
        'emi_l12']]

        X_train = np.stack([iod_u_df.iloc[:validate_to, :].values, 
            pdo_u_df.iloc[:validate_to, :].values,
            oni_u_df.iloc[:validate_to, :].values,
            emi_u_df.iloc[:validate_to, :].values,
            X_allrainfall_i_train,
            X_station_pair_i_train
            ], axis=-1)
        
        X_validate = np.stack([iod_u_df.iloc[validate_to:, :].values, 
            pdo_u_df.iloc[validate_to:, :].values,
            oni_u_df.iloc[validate_to:, :].values,
            emi_u_df.iloc[validate_to:, :].values,
            X_allrainfall_i_validate,
            X_station_pair_i_validate
            ], axis=-1)
        
        y_scaler = StandardScaler()
        y_basinrainfall_train = y_basinrainfall.iloc[:validate_to, :]
        y_basinrainfall_validate = y_basinrainfall.iloc[validate_to:, :]
        
        y_basinrainfall_i_train = y_scaler.fit_transform(y_basinrainfall_train)
        y_basinrainfall_i_validate = y_scaler.transform(y_basinrainfall_validate)

        scaler_list = {
            'overall' : overall_scaler,
            'y' : y_scaler,
            'station_pair' : station_pair_scaler,
        }

        pickle.dump(scaler_list, open(station_result_path / 'station_scaler.bin', 'wb'))

        model = build_model()
        model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))

        print(X_train.shape, y_basinrainfall_i_train.reshape(-1, 12, 1).shape)

        history = model.fit(
            X_train, y_basinrainfall_i_train.reshape(-1, 12),
            epochs=500,
            batch_size=8,
            validation_data=(X_validate, y_basinrainfall_i_validate.reshape(-1, 12)),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(station_result_path / 'station_model.bin', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5),
                tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 60),
                # WandbCallback()
            ],
        )
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()
        plt.title('last loss: ' + str(model.history.history['loss'][-1]))
        plt.savefig(station_result_path / 'station_model_loss.png', format='png')

        print('----------')

    """
    * End Modeling
    """
    config['station_model_last_update'] = date_path
    print(config)
    open('%sconfig.yml'%(config['project_path']), 'w').write(yaml.dump(config))