__author__ = "Puri Phakmongkol"
__author_email__ = "me@puri.in.th"

"""
* HII Operation Script
*
* Created date : 07/07/2022
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
o      o  _-_-_-_- Monsoon & MJO Mid Range Model
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
import functools
import os
import pickle
import argparse
import json

from sklearn.preprocessing import StandardScaler
import tensorflow as tf

config = yaml.load(
    open('config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

"""
* Parameters config
"""
lr = 1e-5
batch_size = 16
n_feature = 11
epochs = 1000
validation_size = 30

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

result_path = Path(project_path) / 'output' / date_path 
result_model_path = result_path / 'mm_model'

month_name_to_number = {
    'JAN' : 1,
    'FEB' : 2,
    'MAR' : 3,
    'APR' : 4,
    'MAY' : 5,
    'JUN' : 6,
    'JUL' : 7,
    'AUG' : 8,
    'SEP' : 9,
    'OCT' : 10,
    'NOV' : 11,
    'DEC' : 12
}

"""
* MJO Monsoon Preprocessing function
"""

def findDayAngle(day_of_year: int) -> float :
    return (2*np.pi*(day_of_year -1))/365

def findPhaseAngle(mjo_phase: int) -> float :
    return (2* np.pi * (mjo_phase -1))/8

"""
* Model
"""
def build_model() :
    input_mm = tf.keras.layers.Input(shape=(84, n_feature))
    input_ocean = tf.keras.layers.Input(shape=(3, 8))

    lstm_mm_1 = tf.keras.layers.LSTM(256)(input_mm)
    lstm_ocean_1 = tf.keras.layers.LSTM(16)(input_ocean)

    ccc = tf.keras.layers.Concatenate()([lstm_mm_1, lstm_ocean_1])
    outp = tf.keras.layers.Dense(3, activation=tf.keras.activations.linear)(ccc)

    return tf.keras.Model(inputs=[input_mm, input_ocean], outputs=outp)

if __name__ == '__main__' :
    print('Load MJO and Monsoon Data')

    if latest_indices_update == '' or latest_indices_update == None :
        print('There are no indices data')
        exit()

    df = pd.read_csv(Path(project_path)/ 'data' / latest_indices_update / 'mjo_monsoon_indices.csv')
    df['datetime'] = pd.to_datetime(df['date'])
    df['datetime'] = df['datetime'].dt.tz_localize('Asia/Bangkok')
    df = df.set_index('datetime')

    mjo_monsoon_df = df[['wp_value', 'im_value', 'RMM1', 'RMM2', 'phase', 'amplitude',]]

    print('Load rainfall data')

    rain_df = pd.read_csv(Path(project_path)/ 'data' / 'static' / config['annual_rainfall_file'])
    rain_month_df = rain_df[['year','JAN', 'FEB', 'MAR', 'APR', 'MAY',
       'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]

    monthly_rain_df = rain_month_df.melt(id_vars=['year'], value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
       'OCT', 'NOV', 'DEC'])
    
    monthly_rain_df.columns = ['year', 'month', 'monthly_rain']
    monthly_rain_df['month'] = monthly_rain_df['month'].apply(lambda _: month_name_to_number[_])
    monthly_rain_df['date'] = monthly_rain_df.apply(
        lambda _: str(int(_['year'])) + '-' +  (str(int(_['month'])) if len(str(int(_['month']))) == 2 else '0' + str(int(_['month']))) + '-01',
        axis = 1
    )
    monthly_rain_df['datetime'] = pd.to_datetime(monthly_rain_df['date'])
    monthly_rain_df['datetime'] = monthly_rain_df['datetime'].dt.tz_localize(
        'Asia/Bangkok'
    )
    monthly_rain_df.drop(columns=['date'], inplace=True)
    monthly_rain_df['datetime'] = pd.to_datetime(monthly_rain_df['datetime'])
    monthly_rain_df.set_index('datetime', inplace=True)
    monthly_rain_df.sort_index(inplace=True)
    monthly_rain_df = monthly_rain_df[['monthly_rain']]

    print('Load Oceanic indices')

    oni = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'oni_df.csv')
    iod = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'dmi_df.csv')
    pdo = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'pdo_df.csv')
    emi = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'emi_df.csv')

    pdo = pdo[['month','year','pdo']]

    oni['merge_my'] = oni.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)
    iod['merge_my'] = iod.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)
    pdo['merge_my'] = pdo.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)
    emi['merge_my'] = emi.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)

    ocean_indices_df = oni.join(iod.drop(columns=['year', 'month']).set_index('merge_my'), how='inner', on='merge_my')
    ocean_indices_df = ocean_indices_df.join(pdo.drop(columns=['year', 'month']).set_index('merge_my'), how='inner', on='merge_my')
    ocean_indices_df = ocean_indices_df.join(emi.drop(columns=['year', 'month']).set_index('merge_my'), how='inner', on='merge_my')

    ocean_indices_df = ocean_indices_df.dropna()
    ocean_indices_df = ocean_indices_df[['year', 'month', 'oni', 'iod', 'pdo', 'emi']]
    ocean_indices_df['datetime'] = ocean_indices_df.apply(lambda _: datetime.datetime(int(_['year']), int(_['month']), 1), axis=1)

    ocean_indices_df = ocean_indices_df.set_index('datetime')
    ocean_indices_df = ocean_indices_df.drop(columns=['year', 'month'])

    ocean_indices_df['oni_s1'] = ocean_indices_df['oni'].shift(1)
    ocean_indices_df['iod_s1'] = ocean_indices_df['iod'].shift(1)
    ocean_indices_df['pdo_s1'] = ocean_indices_df['pdo'].shift(1)
    ocean_indices_df['emi_s1'] = ocean_indices_df['emi'].shift(1)
    ocean_indices_df = ocean_indices_df.dropna()

    ocean_indices_df['oni_diff'] = ocean_indices_df['oni'] - ocean_indices_df['oni_s1']
    ocean_indices_df['iod_diff'] = ocean_indices_df['iod'] - ocean_indices_df['iod_s1']
    ocean_indices_df['pdo_diff'] = ocean_indices_df['pdo'] - ocean_indices_df['pdo_s1']
    ocean_indices_df['emi_diff'] = ocean_indices_df['emi'] - ocean_indices_df['emi_s1']
    ocean_indices_df = ocean_indices_df.drop(columns=['oni_s1', 'iod_s1', 'pdo_s1', 'emi_s1'])

    print('MJO Monsoon and Rainfall Preprocessing')

    mjo_monsoon_df['day_of_year'] = [ _.to_pydatetime().timetuple().tm_yday for _ in mjo_monsoon_df.index ]
    
    mjo_monsoon_df['day_of_year_sin'] = mjo_monsoon_df['day_of_year'].apply(
        lambda _: np.sin(findDayAngle(_))
    )
    mjo_monsoon_df['day_of_year_cos'] = mjo_monsoon_df['day_of_year'].apply(
        lambda _: np.cos(findDayAngle(_))
    )
    mjo_monsoon_df['phase_sin'] = mjo_monsoon_df['phase'].apply(
        lambda _: np.sin(findPhaseAngle(_))
    )
    mjo_monsoon_df['phase_cos'] = mjo_monsoon_df['phase'].apply(
        lambda _: np.cos(findPhaseAngle(_))
    )

    year_list = list(set([ _.to_pydatetime().year for _ in mjo_monsoon_df.index ]))
    month_list = list(set([ _.to_pydatetime().month for _ in mjo_monsoon_df.index ]))
    year_month_list = [ (y, m) for y in year_list for m in month_list]

    """
    * 
    """

    year_month_sample_list = []
    year_month_sample_next_list = []
    for _ in range(len(year_month_list) - 2):
        year_month_sample_list.append(year_month_list[_: _+3])
        year_month_sample_next_list.append(year_month_list[_+3: _+6])

    year_month_sample_next_list = year_month_sample_next_list[ : -3]
    year_month_sample_list = year_month_sample_list[: -3]

    X_df_list = []
    X_ocean_list = []
    y_value_list = []
    y_year_list = []

    for f in range(len(year_month_sample_list)) :
        try : 
            mjo_monsoon_f_df = mjo_monsoon_df[datetime.datetime(year_month_sample_list[f][0][0], year_month_sample_list[f][0][1], 1): datetime.datetime(year_month_sample_next_list[f][0][0], year_month_sample_next_list[f][0][1], 1) - datetime.timedelta(days=1)]
            # y_value = monthly_rain_df[datetime.datetime(year_month_sample_next_list[f][0], year_month_sample_next_list[f][1], 1) : datetime.datetime(year_month_sample_next_list[f][0], year_month_sample_next_list[f][1], 1)]
            ocean_indices_f_df = ocean_indices_df[datetime.datetime(year_month_sample_list[f][0][0], year_month_sample_list[f][0][1], 1)
                                                    : datetime.datetime(year_month_sample_list[f][-1][0], year_month_sample_list[f][-1][1], 1)]

            y_value = []
            # print(ocean_indices_f_df.shape)
            for _ in year_month_sample_next_list[f] :
                y_value.append(monthly_rain_df[datetime.datetime(_[0], _[1], 1) : datetime.datetime(_[0], _[1], 1)].values[0][0])

            # s = functools.reduce(lambda a,b : a or b, [ np.isnan(_) for _ in y_value])
            if functools.reduce(lambda a,b : a or b, [ np.isnan(_) for _ in y_value]) :
                raise ValueError('Have Nan')

            if ocean_indices_f_df.shape[0] != 3 :
                raise ValueError('Oceanic Indices less than 3')
            
            in_month_list = list(set([ _.to_pydatetime().month for _ in mjo_monsoon_f_df.index ]))
            in_df_list = []
            for m in in_month_list :
                in_df = mjo_monsoon_f_df.loc[[ _ for _ in mjo_monsoon_f_df.index if _.to_pydatetime().month == m ]]
                in_df_list.append(in_df.iloc[:28, :])

            sum_df = functools.reduce(lambda a,b: pd.concat([a, b], axis=0), in_df_list)

            if sum_df.shape[0] != 84 :
                raise ValueError('Sum DF less than 87')

            y_value_list.append(y_value)
            X_df_list.append(sum_df.sort_index())
            y_year_list.append(year_month_sample_next_list[f])
            X_ocean_list.append(ocean_indices_f_df)

        except :
            print(year_month_sample_next_list[f], 'error')

    """
    * Train / Validation Split
    """
    y_train_year_list = y_year_list[: -validation_size]
    y_validate_year_list = y_year_list[validation_size: ]

    X_train = np.array([ np.array(_.values) for _ in X_df_list[: -validation_size]])
    X_ocean_train = np.array([ np.array(_.values) for _ in X_ocean_list[: -validation_size]])
    X_validation = np.array([ np.array(_.values) for _ in X_df_list[validation_size: ]])
    X_ocean_validation = np.array([ np.array(_.values) for _ in X_ocean_list[validation_size: ]])

    y_train = np.array(y_value_list[: -validation_size])
    y_validate = np.array(y_value_list[validation_size: ])

    print(len(y_train_year_list))
    print(len(y_validate_year_list))

    """
    * Scaling Y
    """
    y_scaler = StandardScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_validation_s = y_scaler.transform(y_validate)

    if not result_path.exists() :
        os.mkdir(result_path)
        
    if not result_model_path.exists() :
        os.mkdir(result_model_path)

    pickle.dump(y_scaler, open(result_model_path / 'scaler.bin', 'wb'))

    """
    * Modeling 
    """
    model = build_model()
    model.summary()

    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    history = model.fit(
        (X_train, X_ocean_train), y_train_s,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=((X_validation, X_ocean_validation), y_validation_s),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(result_model_path / 'mm_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5),
            tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 30),
        ]
    )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.title('last loss: ' + str(model.history.history['loss'][-1]))
    plt.savefig(result_model_path / 'loss.png')

    config['mm_model_last_update'] = date_path
    print(config)
    open('%sconfig.yml'%(config['project_path']), 'w').write(yaml.dump(config))