__author__ = "Puri Phakmongkol"
__author_email__ = "me@puri.in.th"

"""
* HII Operation Script
*
* Created date : 06/07/2022
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
o      o  _-_-_-_- SimIDX Plus Script
    o           +
+      +     o        o      +
python 2_simidx_plus.py --init_month 10 --target_year 2565
""" 

import pandas as pd
import numpy as np

# %config InlineBackend.figure_format = 'svg'

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(6, 4)})

import yaml
from yaml.loader import SafeLoader

import functools
import requests
import re
import io
from pathlib import Path
import datetime
import os
import argparse

config = yaml.load(
    open('/home/studio-lab-user/sagemaker-studiolab-notebooks/workspace/hii_operation/config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

project_path = config['project_path']
latest_indices_update = config['last_update']

def group_class(_, p) :
    # print(_)
    if _ <= p['10'] :
        r = 'ED'
    elif _ > p['10'] and _ <= p['20'] :
        r = 'D'
    elif _ > p['20'] and _ <= p['33'] :
        r = 'ND'
    elif _ > p['33'] and _ <= p['66'] : 
        r = 'N'
    elif _ > p['66'] and _ <= p['80'] : 
        r = 'NW'
    elif _ > p['80'] and _ <= p['90'] : 
        r = 'W'
    elif _ > p['90'] :
        r = 'EW'
    else : 
        r = np.nan

    return r

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--select_month', help='selected init month',
                        type=int)
    parser.add_argument('--end_year', help='selected target_year',
                        type=int)

    args = parser.parse_args()
    print('select_month = ', args.select_month)
    print('end_year', args.end_year)
    
    print('Loading Rainfall data')
    rain_df = pd.read_csv(Path(f'{project_path}data/static') / 'TMD_idw_annual_rainfall_2019.csv')
    rain_df['sRain'] = rain_df['annual_rainfall']
    climRain = rain_df[ (rain_df['year']>= 1981)&(rain_df['year']<= 2010) ]
    
    medianR = climRain['sRain'].quantile(q=0.5)
    p95R = climRain['sRain'].quantile(q=0.95)
    p90R = climRain['sRain'].quantile(q=0.90)
    p85R = climRain['sRain'].quantile(q=0.85)
    p80R = climRain['sRain'].quantile(q=0.80)
    p75R = climRain['sRain'].quantile(q=0.75)
    p70R = climRain['sRain'].quantile(q=0.70)
    p60R = climRain['sRain'].quantile(q=0.60)
    p50R = climRain['sRain'].quantile(q=0.50)
    p40R = climRain['sRain'].quantile(q=0.40)
    p30R = climRain['sRain'].quantile(q=0.30)
    p20R = climRain['sRain'].quantile(q=0.20)
    p25R = climRain['sRain'].quantile(q=0.25)
    p15R = climRain['sRain'].quantile(q=0.15)
    p10R = climRain['sRain'].quantile(q=0.10)
    p05R = climRain['sRain'].quantile(q=0.05)
    p33R = climRain['sRain'].quantile(q=0.33)
    p66R = climRain['sRain'].quantile(q=0.66)
    
    p = {
         '95' : p95R,
         '90' : p90R,
         '85' : p85R,
         '80' : p80R,
         '75' : p75R,
         '70' : p70R,
         '60' : p60R,
         '50' : p50R,
         '40' : p40R,
         '30' : p30R,
         '20' : p20R,
         '25' : p25R,
         '15' : p15R,
         '10' : p10R,
         '05' : p05R,
         '33' : p33R,
         '66' : p66R
    }
    
    rain_df['group3'] = rain_df['sRain'].apply(lambda _: 'D' if _ <= p['15'] else ( 'W' if _ > p['85'] else ( np.nan if np.isnan(_) else 'N') ))
    rain_df['group'] = rain_df['sRain'].apply(lambda _: group_class(_, p))

    s_rain = rain_df['year']
    
    print('Loading indices data')
    oni = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'oni_df.csv')
    iod = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'dmi_df.csv')
    pdo = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'pdo_df.csv')
    emi = pd.read_csv(Path(f'{project_path}data') / latest_indices_update / 'emi_df.csv')
    
    pdo = pdo[['month','year','pdo']]
    
    pdo['pdo_l01'] = pdo['pdo'].shift(periods=1)
    pdo['pdo_l02'] = pdo['pdo'].shift(periods=2)
    pdo['pdo_l03'] = pdo['pdo'].shift(periods=3)
    pdo['pdo_l04'] = pdo['pdo'].shift(periods=4)
    pdo['pdo_l05'] = pdo['pdo'].shift(periods=5)
    pdo['pdo_l06'] = pdo['pdo'].shift(periods=6)
    pdo['pdo_l07'] = pdo['pdo'].shift(periods=7)
    pdo['pdo_l08'] = pdo['pdo'].shift(periods=8)
    pdo['pdo_l09'] = pdo['pdo'].shift(periods=9)
    pdo['pdo_l10'] = pdo['pdo'].shift(periods=10)
    pdo['pdo_l11'] = pdo['pdo'].shift(periods=11)
    pdo['pdo_l12'] = pdo['pdo'].shift(periods=12)
    
    iod['iod_l01'] = iod['iod'].shift(periods=1)
    iod['iod_l02'] = iod['iod'].shift(periods=2)
    iod['iod_l03'] = iod['iod'].shift(periods=3)
    iod['iod_l04'] = iod['iod'].shift(periods=4)
    iod['iod_l05'] = iod['iod'].shift(periods=5)
    iod['iod_l06'] = iod['iod'].shift(periods=6)
    iod['iod_l07'] = iod['iod'].shift(periods=7)
    iod['iod_l08'] = iod['iod'].shift(periods=8)
    iod['iod_l09'] = iod['iod'].shift(periods=9)
    iod['iod_l10'] = iod['iod'].shift(periods=10)
    iod['iod_l11'] = iod['iod'].shift(periods=11)
    iod['iod_l12'] = iod['iod'].shift(periods=12)
    
    oni['oni_l01'] = oni['oni'].shift(periods=1)
    oni['oni_l02'] = oni['oni'].shift(periods=2)
    oni['oni_l03'] = oni['oni'].shift(periods=3)
    oni['oni_l04'] = oni['oni'].shift(periods=4)
    oni['oni_l05'] = oni['oni'].shift(periods=5)
    oni['oni_l06'] = oni['oni'].shift(periods=6)
    oni['oni_l07'] = oni['oni'].shift(periods=7)
    oni['oni_l08'] = oni['oni'].shift(periods=8)
    oni['oni_l09'] = oni['oni'].shift(periods=9)
    oni['oni_l10'] = oni['oni'].shift(periods=10)
    oni['oni_l11'] = oni['oni'].shift(periods=11)
    oni['oni_l12'] = oni['oni'].shift(periods=12)
    
    emi['emi_l01'] = emi['emi'].shift(periods=1)
    emi['emi_l02'] = emi['emi'].shift(periods=2)
    emi['emi_l03'] = emi['emi'].shift(periods=3)
    emi['emi_l04'] = emi['emi'].shift(periods=4)
    emi['emi_l05'] = emi['emi'].shift(periods=5)
    emi['emi_l06'] = emi['emi'].shift(periods=6)
    emi['emi_l07'] = emi['emi'].shift(periods=7)
    emi['emi_l08'] = emi['emi'].shift(periods=8)
    emi['emi_l09'] = emi['emi'].shift(periods=9)
    emi['emi_l10'] = emi['emi'].shift(periods=10)
    emi['emi_l11'] = emi['emi'].shift(periods=11)
    emi['emi_l12'] = emi['emi'].shift(periods=12)
    
    """
    * Define parameters
    """
    # Lag 3 Month
    selected_month = args.select_month
    # Lag 1 Month
    # selected_month = 12

    start_year = 1981
    end_year = args.end_year - 543
    
    StartYr = 1981
    EndYr = args.end_year - 543
    init_month = selected_month
    i = init_month
    
    """
    * Preprocess Indices dataframe
    """
    
    oni_df = oni[oni['month'] == selected_month]
    oni_df = oni_df[(oni_df['year'] >= start_year)&(oni_df['year'] <= end_year)]
    
    pdo_df = pdo[pdo['month'] == selected_month]
    pdo_df = pdo_df[(pdo_df['year'] >= start_year)&(pdo_df['year'] <= end_year)]
    
    iod_df = iod[iod['month'] == selected_month]
    iod_df = iod_df[(iod_df['year'] >= start_year)&(iod_df['year'] <= end_year)]
    
    emi_df = emi[emi['month'] == selected_month]
    emi_df = emi_df[(emi_df['year'] >= start_year)&(emi_df['year'] <= end_year)]
    
    indices_df = pdo_df.join(oni_df.drop(columns=['month']).set_index('year'), how='inner', on='year')
    indices_df = indices_df.join(iod_df.drop(columns=['month']).set_index('year'), how='inner', on='year')
    indices_df = indices_df.join(emi_df.drop(columns=['month']).set_index('year'), how='inner', on='year')
    indices_df = indices_df.reset_index()
    
    indices_df = indices_df.drop(columns=['index', 'month', 'pdo', 'oni', 'iod', 'emi'])
    
    s_df = indices_df.copy()
    s_df['th_year'] = s_df['year'] + 543
    if i < 8 :
        s_df['f_year'] = s_df['year']
        s_df['f_thyear'] = s_df['th_year']

    else :
        s_df['f_year'] = s_df['year'] + 1
        s_df['f_thyear'] = s_df['th_year'] + 1
    
    s_df = s_df.join(rain_df.drop(columns=['th_year']).set_index('year'), how='left', on='f_year')
    s_df = s_df[(s_df['year'] >= StartYr)&(s_df['year'] <= EndYr)]
    
    sub_df = s_df[['pdo_l01', 'pdo_l02', 'pdo_l03', 'pdo_l04', 'pdo_l05',
       'pdo_l06', 'pdo_l07', 'pdo_l08', 'pdo_l09', 'pdo_l10', 'pdo_l11',
       'pdo_l12', 'oni_l01', 'oni_l02', 'oni_l03', 'oni_l04', 'oni_l05',
       'oni_l06', 'oni_l07', 'oni_l08', 'oni_l09', 'oni_l10', 'oni_l11',
       'oni_l12', 'iod_l01', 'iod_l02', 'iod_l03', 'iod_l04', 'iod_l05',
       'iod_l06', 'iod_l07', 'iod_l08', 'iod_l09', 'iod_l10', 'iod_l11',
       'iod_l12', 'emi_l01', 'emi_l02', 'emi_l03', 'emi_l04', 'emi_l05',
       'emi_l06', 'emi_l07', 'emi_l08', 'emi_l09', 'emi_l10', 'emi_l11',
       'emi_l12']]
    
    """
    * Define Label
    """
    
    print(indices_df)
    