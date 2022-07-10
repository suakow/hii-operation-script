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
import os
import argparse
import json

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
epochs = 500

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

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

    print(rain_df.head())