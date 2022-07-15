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
o      o  _-_-_-_- SimIDX V2
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
import argparse
import json
import functools

import pickle

import tensorflow as tf

config = yaml.load(
    open('config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

n_feature=11
mm_lr = 1e-5
mm_batch_size = 16
mm_n_feature = 11
mm_epochs = 500

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

s_selected_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV','DEC']

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

"""
* MJO Monsoon Preprocessing function
"""

def findDayAngle(day_of_year: int) -> float :
    return (2*np.pi*(day_of_year -1))/365

def findPhaseAngle(mjo_phase: int) -> float :
    return (2* np.pi * (mjo_phase -1))/8

"""
* MM Model
"""
def build_model() :
    input_mm = tf.keras.layers.Input(shape=(84, n_feature))
    input_ocean = tf.keras.layers.Input(shape=(3, 8))

    lstm_mm_1 = tf.keras.layers.LSTM(256)(input_mm)
    lstm_ocean_1 = tf.keras.layers.LSTM(16)(input_ocean)

    ccc = tf.keras.layers.Concatenate()([lstm_mm_1, lstm_ocean_1])
    outp = tf.keras.layers.Dense(3, activation=tf.keras.activations.linear)(ccc)

    return tf.keras.Model(inputs=[input_mm, input_ocean], outputs=outp)

"""
* Station Prediction Model
"""
def build_station_pred_model() :
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(12, 6)))
    model.add(tf.keras.layers.GRU(64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(12))

    return model

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--select_year', help='selected target_year',
                        type=int)

    args = parser.parse_args()
    print('select_year = ', args.select_year)

    """
    * Load SimIDX Plus data
    """
    if config['simidx_last_update'] == '' or config['simidx_last_update'] == None or 'simidx_last_update' not in config :
        print('There are no indices data')
        exit()

    group_result = json.loads(open(Path(project_path) / 'output' / config['simidx_last_update'] / 'simidx_plus' / 'result_filter.json', 'r').read())
    group_result = { int(k): int(v) for k, v in group_result.items() }
    s_df = pd.read_csv(Path(project_path) / 'output' / config['simidx_last_update'] / 'simidx_plus' / 's_df.csv')
    
    test_year = [args.select_year]
    test_year_list = [ (_ , group_result[_]) for _ in test_year ]

    """
    * Prepare indices features from SimIDX Plus
    """
    X_pred_allrainfall = []
    X_pred_oni = []
    X_pred_pdo = []
    X_pred_iod = []
    X_pred_emi = []

    for ty in test_year_list :
        X_pred_allrainfall.append(s_df[s_df['f_thyear'] == ty[1]][s_selected_columns].values.reshape(-1))
        X_pred_oni.append(s_df[s_df['f_thyear'] == ty[0]][['oni_l01', 'oni_l02', 'oni_l03', 'oni_l04', 'oni_l05',
        'oni_l06', 'oni_l07', 'oni_l08', 'oni_l09', 'oni_l10', 'oni_l11',
        'oni_l12']].values.reshape(-1))
        X_pred_pdo.append(s_df[s_df['f_thyear'] == ty[0]][['pdo_l01', 'pdo_l02', 'pdo_l03', 'pdo_l04', 'pdo_l05',
        'pdo_l06', 'pdo_l07', 'pdo_l08', 'pdo_l09', 'pdo_l10', 'pdo_l11',
        'pdo_l12']].values.reshape(-1))
        X_pred_iod.append(s_df[s_df['f_thyear'] == ty[0]][['iod_l01', 'iod_l02', 'iod_l03', 'iod_l04', 'iod_l05',
        'iod_l06', 'iod_l07', 'iod_l08', 'iod_l09', 'iod_l10', 'iod_l11',
        'iod_l12']].values.reshape(-1)) 
        X_pred_emi.append(s_df[s_df['f_thyear'] == ty[0]][['emi_l01', 'emi_l02', 'emi_l03', 'emi_l04', 'emi_l05',
        'emi_l06', 'emi_l07', 'emi_l08', 'emi_l09', 'emi_l10', 'emi_l11',
        'emi_l12']].values.reshape(-1)) 

    X_pred_allrainfall = np.array(X_pred_allrainfall)
    X_pred_oni = np.array(X_pred_oni)
    X_pred_pdo = np.array(X_pred_pdo)
    X_pred_iod = np.array(X_pred_iod)
    X_pred_emi = np.array(X_pred_emi)

    """
    * Preparing MJO, Monsoon and Oceanic Indices Data
    """
    if latest_indices_update == '' or latest_indices_update == None :
        print('There are no indices data')
        exit()

    print('Load MJO and Monsoon Indices Data')

    # df = pd.read_csv(Path(project_path)/ 'data' / latest_indices_update / 'mjo_monsoon_indices.csv')
    # df['datetime'] = pd.to_datetime(df['date'])
    # df['datetime'] = df['datetime'].dt.tz_localize('Asia/Bangkok')
    # df = df.set_index('datetime')

    # mjo_monsoon_df = df[['wp_value', 'im_value', 'RMM1', 'RMM2', 'phase', 'amplitude',]]

    mjo_monsoon_df = pd.read_pickle(Path(project_path) / 'output' / config['mm_model_last_update'] / 'mm_model' / 'mjo_monsoon_df.bin')

    # print('Load Oceanic Indices Data')

    # oni = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'oni_df.csv')
    # iod = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'dmi_df.csv')
    # pdo = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'pdo_df.csv')
    # emi = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'emi_df.csv')

    # pdo = pdo[['month','year','pdo']]

    # oni['merge_my'] = oni.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)
    # iod['merge_my'] = iod.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)
    # pdo['merge_my'] = pdo.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)
    # emi['merge_my'] = emi.apply(lambda _: str(int(_['year'])) + str(int(_['month'])), axis=1)

    # ocean_indices_df = oni.join(iod.drop(columns=['year', 'month']).set_index('merge_my'), how='inner', on='merge_my')
    # ocean_indices_df = ocean_indices_df.join(pdo.drop(columns=['year', 'month']).set_index('merge_my'), how='inner', on='merge_my')
    # ocean_indices_df = ocean_indices_df.join(emi.drop(columns=['year', 'month']).set_index('merge_my'), how='inner', on='merge_my')

    # ocean_indices_df = ocean_indices_df.dropna()
    # ocean_indices_df = ocean_indices_df[['year', 'month', 'oni', 'iod', 'pdo', 'emi']]
    # ocean_indices_df['datetime'] = ocean_indices_df.apply(lambda _: datetime.datetime(int(_['year']), int(_['month']), 1), axis=1)

    # ocean_indices_df = ocean_indices_df.set_index('datetime')
    # ocean_indices_df = ocean_indices_df.drop(columns=['year', 'month'])

    # ocean_indices_df['oni_s1'] = ocean_indices_df['oni'].shift(1)
    # ocean_indices_df['iod_s1'] = ocean_indices_df['iod'].shift(1)
    # ocean_indices_df['pdo_s1'] = ocean_indices_df['pdo'].shift(1)
    # ocean_indices_df['emi_s1'] = ocean_indices_df['emi'].shift(1)
    # ocean_indices_df = ocean_indices_df.dropna()

    # ocean_indices_df['oni_diff'] = ocean_indices_df['oni'] - ocean_indices_df['oni_s1']
    # ocean_indices_df['iod_diff'] = ocean_indices_df['iod'] - ocean_indices_df['iod_s1']
    # ocean_indices_df['pdo_diff'] = ocean_indices_df['pdo'] - ocean_indices_df['pdo_s1']
    # ocean_indices_df['emi_diff'] = ocean_indices_df['emi'] - ocean_indices_df['emi_s1']
    # ocean_indices_df = ocean_indices_df.drop(columns=['oni_s1', 'iod_s1', 'pdo_s1', 'emi_s1'])
    ocean_indices_df = pd.read_pickle(Path(project_path) / 'output' / config['mm_model_last_update'] / 'mm_model' / 'oceanic_indices_df.bin')

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

    year_month_selected_list = [ _ for _ in year_month_list if _[0] in [ x[0]-543 for x in test_year_list ]]

    year_month_sample_list = []
    year_month_sample_next_list = []
    for _ in range(len(year_month_list) - 2):
        try :
            if year_month_list[_+3:_+6][0] in year_month_selected_list : 
                year_month_sample_list.append(year_month_list[_:_+3])
                year_month_sample_next_list.append(year_month_list[_+3:_+6])

        except :
            pass

    year_month_sample_next_list = year_month_sample_next_list[ :-5]
    year_month_sample_list = year_month_sample_list[:-5]

    """
    * Prepare data for inference
    """
    X_df_list = []
    X_ocean_list = []
    y_value_list = []
    y_year_list = []

    for f in range(len(year_month_sample_list)) :
        print(year_month_sample_list[f], '->', year_month_sample_next_list[f])

        try : 
            mjo_monsoon_f_df = mjo_monsoon_df[datetime.datetime(year_month_sample_list[f][0][0], year_month_sample_list[f][0][1], 1): datetime.datetime(year_month_sample_next_list[f][0][0], year_month_sample_next_list[f][0][1], 1) - datetime.timedelta(days=1)]
            ocean_indices_f_df = ocean_indices_df[datetime.datetime(year_month_sample_list[f][0][0], year_month_sample_list[f][0][1], 1)
                                                    : datetime.datetime(year_month_sample_list[f][-1][0], year_month_sample_list[f][-1][1], 1)]
            
            if ocean_indices_f_df.shape[0] != 3 :
                raise ValueError('Oceanic Indices less than 3')
            
            in_month_list = list(set([ _.to_pydatetime().month for _ in mjo_monsoon_f_df.index ]))
            in_df_list = []
            for m in in_month_list :
                in_df = mjo_monsoon_f_df.loc[[ _ for _ in mjo_monsoon_f_df.index if _.to_pydatetime().month == m ]]
                in_df_list.append(in_df.iloc[:28, :])

            sum_df = functools.reduce(lambda a,b: pd.concat([a, b], axis=0), in_df_list)
            print(sum_df.shape)

            if sum_df.shape[0] != 84 :
                raise ValueError('Shape of DF less than 92')

            # y_value_list.append(y_value)
            X_df_list.append(sum_df.sort_index())
            y_year_list.append(year_month_sample_next_list[f])
            X_ocean_list.append(ocean_indices_f_df)

        except :
            print(year_month_sample_next_list[f], 'error')

    # print(X_df_list[-1])

    # print(X_df_list)
    X_pred_longrange = np.array([ np.array(_.values) for _ in X_df_list ])
    X_pred_ocean = np.array([ np.array(_.values) for _ in X_ocean_list ])
    # print(X_pred_longrange.shape)

    y_scaler = pickle.load(open(Path(project_path) / 'output' / config['mm_model_last_update'] / 'mm_model' / 'scaler.bin', 'rb'))

    """
    * Inferences
    """
    model = build_model()
    print(model.summary())
    model.compile(loss=tf.keras.losses.MeanSquaredError(), 
              optimizer=tf.keras.optimizers.Adam(learning_rate=mm_lr))

    model.load_weights(Path(project_path) / 'output' / config['mm_model_last_update'] / 'mm_model' / 'mm_model.hdf5')

    y_pred_mm_group = []
    in_pred = []
    for n_xt in range(len(X_pred_longrange)) : 
        in_y_pred = model.predict((X_pred_longrange[n_xt].reshape(1, 84, mm_n_feature),
                                    X_pred_ocean[n_xt].reshape(1, 3, 8)))
        in_pred.append(in_y_pred)

        y_pred_mm_group.append(np.array(in_pred))

    in_pred = np.array(in_pred)
    X_pred_longrange_scale = [ y_scaler.inverse_transform(s)[0] for s in in_pred ]
    # print(X_pred_longrange_scale)

    """
    * Merge to SimIDX Result
    """
    X_pred_allrainfall_copy = X_pred_allrainfall.copy()
    # print(X_pred_allrainfall_copy)
    X_pred_longrange_scale_group = []
    X_pred_subgroup = []

    last_year = 0

    print(y_year_list)

    for n_y_year in range(len(y_year_list)) :
        if last_year != y_year_list[n_y_year][0][0] :
            last_year = y_year_list[n_y_year][0][0]

            if len(X_pred_subgroup) != 0 :
                X_pred_longrange_scale_group.append(X_pred_subgroup)
                X_pred_subgroup = []

            X_pred_subgroup.append(X_pred_longrange_scale[n_y_year])

        else :
            X_pred_subgroup.append(X_pred_longrange_scale[n_y_year])

    if len(X_pred_subgroup) != 0 :
        X_pred_longrange_scale_group.append(X_pred_subgroup)
        X_pred_subgroup = []

    print(X_pred_longrange_scale_group)
    for n_year in range(len(X_pred_allrainfall)) :
        try :
            in_group = X_pred_longrange_scale_group[n_year]
        except :
            print('index out of range -> Skip')
            continue
            
        n_pos = 0

        for m6_prs in in_group :
            # print(m6_pr)
            m6_pr = m6_prs[:3]
            if len(X_pred_allrainfall[n_year][n_pos:]) >= 3 :
                X_pred_allrainfall[n_year][n_pos: n_pos+3] = m6_pr

            else :
                # print(n_pos)
                # print(X_pred_allrainfall[n_year][n_pos: ])
                X_pred_allrainfall[n_year][n_pos: ] = m6_pr[: len(X_pred_allrainfall[n_year]) - n_pos]

            n_pos += 1

    print('Rainfall after merge with MM Model result')
    print(X_pred_allrainfall)

    if not (Path(project_path) / 'output' / date_path).exists() :
        os.mkdir(Path(project_path) / 'output' / date_path)
        
    if not (Path(project_path) / 'output' / date_path / 'simidx_v2').exists() :
        os.mkdir(Path(project_path) / 'output' / date_path / 'simidx_v2')

    pickle.dump(X_pred_allrainfall, open(Path(project_path) / 'output' / date_path / 'simidx_v2' / 'X_pred_allrainfall.bin', 'wb'))

    """
    * Station Prediction Model
    """
    print('Station Prediction Model')
    station_all_df = pd.read_csv(f'{project_path}/data/static/' + config['station_rainfall_file'])

    station_result = []
    all_station_result = {
        'year_list' : test_year_list,
        'station_prediction' : {}
    }
    for station in [ _ for _ in station_all_df.columns if _ not in ['Unnamed: 0', 'datetime', 'year', 'month'] ] :
        print('station = ', station)

        model = build_station_pred_model()
        model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        model.load_weights(Path(project_path) / 'output' / config['station_model_last_update'] / 'station_model' / station / 'station_model.bin')
        sta_scaler_list = pickle.load(open(Path(project_path) / 'output' / config['station_model_last_update'] / 'station_model' / station / 'station_scaler.bin', 'rb'))
        
        # Station DF
        station_df = station_all_df[['year', 'month', station]]
        station_df['month_eng'] = station_df['month'].apply(lambda _ : num_to_month[_])
        basin_df_piv = station_df.pivot(index='year', columns='month_eng')[station]
        basin_df_piv = basin_df_piv[num_to_month.values()]
        basin_df_piv['annual_rainfall'] = basin_df_piv.apply(np.sum, axis=1)
        basin_df_piv = basin_df_piv.reset_index()
        basin_df_piv['th_year'] = basin_df_piv['year'] + 543

        # print(ts_df)
        y_pred_list = []
        for n_ty in range(len(test_year_list)) :
            X_pred_rainfall = sta_scaler_list['overall'].transform(X_pred_allrainfall[n_ty].reshape(1, -1))
            X_station_pair_df = basin_df_piv[basin_df_piv['th_year'] == test_year_list[n_ty][1]][s_selected_columns].values
            X_pred_station_pair = sta_scaler_list['station_pair'].transform(X_station_pair_df)
            X_oni = X_pred_oni[n_ty].reshape(1, -1)
            X_pdo = X_pred_pdo[n_ty].reshape(1, -1)
            X_iod = X_pred_iod[n_ty].reshape(1, -1)
            X_emi = X_pred_emi[n_ty].reshape(1, -1)
            X_pred = np.stack([X_iod, 
                X_pdo,
                X_oni,
                X_emi,
                X_pred_rainfall,
                X_pred_station_pair
                ], axis=-1)
            y_pred = model.predict(X_pred)
            y_pred_re = sta_scaler_list['y'].inverse_transform(y_pred.reshape(1, -1)).reshape(-1)
            y_pred_list.append(y_pred_re)

        all_station_result['station_prediction'][station] = np.array(y_pred_list)
        print(y_pred_re)

    print('Station Prediction Result (All Stations)')
    print(all_station_result)

    # open(
    #     Path(project_path) / 'output' / date_path / 'simidx_v2' / 'station_prediction_result.json',
    #     'w'
    # ).write(json.dumps(all_station_result))

    """
    * Prepare output dataframe
    """
    print('Prepare output dataframe')
    station_location_df = pd.read_excel(Path(project_path) / 'data' / 'static' / config['station_location_file'])
    station_location_df['station_name_en'] = station_location_df['name_tmd_en'].apply(lambda _ : _.replace('\t', ''))
    
    output_df_list = []
    for station_name in all_station_result['station_prediction'] :
        station_info = list(station_location_df[station_location_df['station_name_en'] == station_name].values[0])

        for year_pred in range(len(all_station_result['year_list'])) :
            year_fc = all_station_result['year_list'][year_pred][0]
            month_fc = 1
            for pred in all_station_result['station_prediction'][station_name][year_pred] :
                output_df_list.append([
                    station_info[0],
                    station_info[12],
                    station_info[13],
                    station_name,
                    station_info[7],
                    year_fc,
                    month_fc,
                    pred
                ])
                month_fc += 1

    output_df = pd.DataFrame(output_df_list, columns=['TMDCode', 
    'Lat', 'Long', 'station_name', 
    'province', 'year_fc', 'month_fc', 'SimIDXV2'])
    output_df.to_pickle(Path(project_path) / 'output' / date_path / 'simidx_v2' / 'output_df.bin')
    output_df.to_csv(Path(project_path) / 'output' / date_path / 'simidx_v2' / 'output.csv')

    config['last_prediction'] = date_path
    print(config)
    open('%sconfig.yml'%(config['project_path']), 'w').write(yaml.dump(config))
