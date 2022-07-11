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
o      o  _-_-_-_- Update Station Rainfall
    o           +
+      +     o        o      +
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

import geopandas as gpd

from tqdm.auto import tqdm

import rioxarray
import xarray as xr

config = yaml.load(
    open('config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

s_selected_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV','DEC']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

"""
Preprocess *.tiff to CSV
https://colab.research.google.com/drive/1YbYX-qd-EyK0YKQtgzEbaoJPLW6-sKIB
"""

if __name__ == '__main__' :

    """
    * Load Basin Shape files
    """
    print('Load basin shape files...')
    gdf = gpd.read_file(Path(project_path) / 'data' / 'shape_file' / 'MainBasin_ONWR_WGS84_4K_3A_With_Island/MainBasin_ONWR_WGS84.shp')
    gdf["area"] = gdf.area

    """
    * Load Station Location
    """
    print('Load station locations...')
    station_location_df = pd.read_excel(Path(project_path) / 'data' / 'static' / config['station_location_file'])
    station_location_df = station_location_df[['code_tmd', 'name_tmd_en', 'main_basin', 'new_lat', 'new_long']]
    station_location_df['name_tmd_en'] = station_location_df['name_tmd_en'].apply(lambda _: _.replace('\t', ''))

    """
    * Load rainfall observation files (Groundtruth)
    """
    monthly_path = Path(project_path) / 'data' / 'rainfall_map' / 'tiff_monthly'

    year_list = []
    month_list = []
    for f in monthly_path.glob('*.tif') :
        f_name = f.name
        ym = f_name.split('o_th')[1].split('.tif')[0]
        year_list.append(int(ym[:4]))
        month_list.append(int(ym[4:]))

    year_list = list(set(year_list))
    monthlist = list(set(month_list))

    year_list = [ str(_) for _ in year_list ]
    # year_list = [ str(_) for _ in year_list if _ != '2021']
    month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    station_value_list = []
    
    for year in tqdm(year_list) :
        for month in tqdm(month_list) : 
            print(f'Y{year} M{month}')
            in_f = monthly_path / f'o_th{year}{month}.tif'

            if not(in_f.exists()) : 
                continue

            xa_sum = rioxarray.open_rasterio(in_f)
            # xa_sum = xa_sum.drop_vars('band')

            in_station_value = [datetime.datetime(int(year), int(month), 1), int(year), int(month)]

            for station in tqdm(station_location_df['name_tmd_en'].values) :
                # print(station)
                # print(station_location_df[station_location_df['name_tmd_en'] == station]['new_long'][0])
                station_value = xa_sum.sel(y=station_location_df[station_location_df['name_tmd_en'] == station]['new_lat'].values[0], 
                                        x=station_location_df[station_location_df['name_tmd_en'] == station]['new_long'].values[0], 
                                        method="nearest").values[0]
                if station_value < 0.0001 :
                    station_value = 0
                
                in_station_value.append(station_value)

            station_value_list.append(in_station_value)

    """
    * End for Loop
    """
    station_value_df = pd.DataFrame(station_value_list, columns=['datetime', 'year', 'month'] + list(station_location_df['name_tmd_en'].values))
    
    """
    * Filter out stations with no rainfall
    """
    column_zero = []
    station_value_describe_df = station_value_df.describe()
    for _ in list(station_value_describe_df.loc['max'].index)[3 :] :
        if station_value_describe_df.loc['max', _] == 0.0 :
            column_zero.append(_)

    station_value_df = station_value_df.drop(columns=column_zero)
    station_value_df.to_csv(Path(project_path) / 'data' / 'static' / config['station_rainfall_file'], index=False)

    config['station_value_last_update'] = date_path
    print(config)
    open('%sconfig.yml'%(config['project_path']), 'w').write(yaml.dump(config))