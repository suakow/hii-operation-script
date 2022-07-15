__author__ = "Puri Phakmongkol"
__author_email__ = "me@puri.in.th"

"""
* HII Operation Script
*
* Created date : 12/07/2022
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
o      o  _-_-_-_- Generate Rainfall Map
    o           +
+      +     o        o      +
""" 

import pandas as pd
import numpy as np

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

import numpy as np
import pandas as pd

import rioxarray
import xarray as xr

import cartopy.crs as ccrs
import cartopy

from matplotlib import colors

import numpy.ma as ma

import wradlib.ipol as ipol
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
import geopandas as gpd

# %config InlineBackend.figure_format = 'svg'

import seaborn as sns
sns.set(rc={'figure.figsize':(6, 4)})
sns.set(rc={'figure.dpi':230})

config = yaml.load(
    open('config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

import matplotlib as mpl 
mpl.font_manager.fontManager.addfont(project_path + 'data/static/thsarabunnew-webfont.ttf')
# /home/ubuntu/workspace/hii-operation-script/data/static/thsarabunnew-webfont.ttf
mpl.rc('font', family='TH Sarabun New', size=13)
plt = mpl.pyplot

thai_month_to_number_dict = {
    'มกราคม' : 1,
    'กุมภาพันธ์' : 2,
    'มีนาคม' : 3,
    'เมษายน' : 4,
    'พฤษภาคม' : 5,
    'มิถุนายน' : 6,
    'กรกฎาคม' : 7,
    'สิงหาคม' : 8,
    'กันยายน' : 9,
    'ตุลาคม' : 10,
    'พฤศจิกายน' : 11,
    'ธันวาคม' : 12,
}

number_to_thai_month_dict = { b: a for a, b in thai_month_to_number_dict.items() }

def create_prediction_dict(pred_df, 
                           pred_columns: str, 
                           station_name_columns: str = 'station_name',
                           year_fc_columns: str = 'year_fc',
                           month_fc_columns: str = 'month_fc') :
    station_pred_result = {}
    station_pred_result['year_list'] = [ (int(_), int(_)) for _ in list(pred_df[year_fc_columns].unique())]
    station_pred_result['month_list'] = {}
    for _ in station_pred_result['year_list'] :
        station_pred_result['month_list'][int(_[0])] = [ int(x) for x in list(pred_df[pred_df[year_fc_columns] == int(_[0])][month_fc_columns].unique())]

    station_pred_result['station_prediction'] = {}
    for _ in list(pred_df[station_name_columns].unique()) :
        _in_df = pred_df[pred_df[station_name_columns] == _]
        _station_result = []
        for year in station_pred_result['year_list'] :
            _i = _in_df[_in_df['year_fc'] == (year[0])]
            _i = list(_i[pred_columns])
            _station_result.append(np.array(_i))

        station_pred_result['station_prediction'][_] = np.array(_station_result)

    return station_pred_result

def hii_rainfall_plot(rds_plot, 
                      template_rds, 
                      plot_title:str = 'hii_rainfall_plot', 
                      self_template: bool = False,
                      figsize: tuple = (16, 10),
                      save_location='') :
    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines() 
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

    if self_template :
        template_rds = rds_plot.copy()
        template_rds.values = ma.masked_where(template_rds.where(template_rds >= 0).values >= 0, template_rds.where(template_rds >= 0).values).mask

    bounds = [-3, 0, 10, 20, 40, 80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 
          220, 230, 240, 250, 260, 270, 280, 290, 300, 400, 500, 700]

    colors_bound_list = [
        np.array([255/255, 255/255, 255/255, 1]), #0
        np.array([255/255, 255/255, 255/255, 1]), #0
        np.array([255/255, 255/255, 204/255, 1]), #10
        np.array([255/255, 255/255, 153/255, 1]),
        np.array([255/255, 255/255, 0/255, 1]),
        np.array([204/255, 255/255, 0/255, 1]),
        np.array([153/255, 255/255, 0/255, 1]), #100
        np.array([102/255, 255/255, 0/255, 1]), #120
        np.array([51/255, 255/255, 0/255, 1]), #140
        np.array([0/255, 255/255, 51/255, 1]), #150
        np.array([0/255, 255/255, 102/255, 1]), #160
        np.array([0/255, 255/255, 153/255, 1]),
        np.array([0/255, 255/255, 153/255, 1]),
        np.array([0/255, 255/255, 255/255, 1]),
        np.array([0/255, 204/255, 255/255, 1]), #200
        np.array([0/255, 153/255, 255/255, 1]),
        np.array([0/255, 102/255, 255/255, 1]),
        np.array([0/255, 51/255, 255/255, 1]),
        np.array([0/255, 0/255, 255/255, 1]),
        np.array([67/255, 0/255, 248/255, 1]),
        np.array([102/255, 0/255, 255/255, 1]),
        np.array([102/255, 0/255, 255/255, 1]),
        np.array([204/255, 0/255, 255/255, 1]),
        np.array([255/255, 0/255, 170/255, 1]), #290
        np.array([255/255, 0/255, 94/255, 1]), #300
        np.array([237/255, 5/255, 32/255, 1]), #400
        np.array([204/255, 4/255, 27/255, 1]), #500
        np.array([184/255, 0/255, 0/255, 1]), #700
        # np.array([100/255, 0/255, 0/255, 1]), #9000
    ]
    custom_cmap = colors.ListedColormap(colors_bound_list)
    norm = colors.BoundaryNorm(bounds, custom_cmap.N)

    rds_plot.where(template_rds).plot(cmap=custom_cmap, levels=bounds, cbar_kwargs={'label': 'Rain (mm)'})
    plt.title(plot_title)
    if save_location == '' :
        plt.show()

    else :
        plt.savefig(save_location, format='png')

    plt.clf()

def hii_rainfall_anomaly_plot(rds_plot, 
                      template_rds, 
                      year_label,
                      month_label,
                      rainfall,
                      rainfall_diff,
                      plot_title:str = 'hii_rainfall_plot', 
                      self_template: bool = False,
                      figsize: tuple = (16, 10),
                      save_location='') :
    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines() 
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

    rainfall = ('+' if rainfall >= 0 else '')  + str(round(rainfall, 2))
    rainfall_diff = ('+' if rainfall_diff >= 0 else '')  + str(round(rainfall_diff, 2))
    month_label = number_to_thai_month_dict[month_label]

    if self_template :
        template_rds = rds_plot.copy()
        template_rds.values = ma.masked_where(template_rds.where(template_rds >= -1000).values >= -1000, template_rds.where(template_rds >= -1000).values).mask


    bounds = [-400, -200, -180, -160, -140, -120, -100, -80, -60, -40, -20, -5, 0,
              5, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 400]

    colors_bound_list = [
        np.array([122/255, 67/255, 24/255, 1]), #-400
        np.array([138/255, 75/255, 28/255, 1]), #-200
        np.array([156/255, 85/255, 31/255, 1]), #-180
        np.array([166/255, 98/255, 43/255, 1]), #-160
        np.array([176/255, 113/255, 55/255, 1]), #-140
        np.array([186/255, 128/255, 69/255, 1]), #-120
        np.array([196/255, 144/255, 84/255, 1]), #-100
        np.array([207/255, 162/255, 99/255, 1]), #-80
        np.array([217/255, 179/255, 117/255, 1]), #-60
        np.array([227/255, 197/255, 134/255, 1]), #-40
        np.array([237/255, 216/255, 152/255, 1]), #-20
        np.array([245/255, 234/255, 171/255, 1]), #-5
        np.array([252/255, 252/255, 252/255, 1]), #0
        np.array([234/255, 242/255, 187/255, 1]), #5
        np.array([212/255, 230/255, 184/255, 1]), #20
        np.array([190/255, 217/255, 180/255, 1]), #40
        np.array([167/255, 201/255, 173/255, 1]), #60
        np.array([145/255, 189/255, 168/255, 1]), #80
        np.array([123/255, 176/255, 163/255, 1]), #100
        np.array([104/255, 166/255, 161/255, 1]), #120
        np.array([82/255, 153/255, 156/255, 1]), #140
        np.array([60/255, 141/255, 150/255, 1]), #160
        np.array([33/255, 130/255, 145/255, 1]), #180
        np.array([29/255, 114/255, 128/255, 1]), #200
        np.array([26/255, 103/255, 115/255, 1]), #400
    ]
    custom_cmap = colors.ListedColormap(colors_bound_list)
    norm = colors.BoundaryNorm(bounds, custom_cmap.N)

    # fig, ax = plt.subplots()

    t = rds_plot.where(template_rds).plot(cmap=custom_cmap, levels=bounds, cbar_kwargs={'label': 'Rain (mm)'}, 
                                          )
    ax2 = t.axes
    # width, height
    ax2.text(0.45, 0.16,f'คาดการณ์ฝนปี {year_label}\nเดือน{month_label}\nปริมาณฝน {rainfall} mm\nต่างจากค่าปกติ {rainfall_diff}%', 
             transform=ax.transAxes)
    plt.title(plot_title)

    if save_location == '' :
        plt.show()

    else :
        plt.savefig(save_location, format='png')

    plt.clf()


def hii_rainfall_save_map(rds_plot, 
                            template_rds, 
                            location: str,
                            self_template: bool = False) :

    if self_template :
        template_rds = rds_plot.copy()
        template_rds.values = ma.masked_where(template_rds.where(template_rds >= 0).values >= 0, template_rds.where(template_rds >= 0).values).mask

    rds_plot.where(template_rds).rio.to_raster(location)

def rainfall_interpolation(pred_rds, 
                           in_value_list, 
                           station_long_list, 
                           station_lat_list, 
                           nnearest: int = 6,
                           p: int = 2) :
    pred_rds = pred_rds.copy()
    src = np.vstack((np.array(station_long_list), np.array(station_lat_list))).transpose()
    vals = np.array(in_value_list)
    trg = np.meshgrid(pred_rds.x.values, pred_rds.y.values)
    trg = np.vstack((trg[0].ravel(), trg[1].ravel())).T

    idw = ipol.Idw(src, trg, nnearest=nnearest, p=p)
    idw_result = idw(np.array(in_value_list))
    pred_rds.values = idw_result.reshape(1, pred_rds.values.shape[1], pred_rds.values.shape[2])
    return pred_rds

def create_prediction_map(result_dict, 
                          station_location_df, 
                          n_year, 
                          n_month, 
                          station_columns = 'name_tmd_en') :
    pred_rds = xr.zeros_like(template_rds)
    pred_rds.attrs = {} 
    station_lat_list = []
    station_long_list = []
    pred_mean_value_list = []

    for station in list(result_dict['station_prediction'].keys()) :
        t_result = result_dict['station_prediction'][station][n_year][n_month]
        t_result = 0.0 if t_result < 0.0 else t_result

        lat = station_location_df[station_location_df[station_columns] == station].values[0, 3]
        long = station_location_df[station_location_df[station_columns] == station].values[0, 4]
        pred_rds.sel(y=lat, x=long, method="nearest").values[0] = t_result
        station_lat_list.append(lat)
        station_long_list.append(long)
        pred_mean_value_list.append(t_result)
    
    pred_rds = rainfall_interpolation(pred_rds, pred_mean_value_list, station_long_list, station_lat_list)

    return pred_rds

if __name__ == '__main__' :
    """
    * Load Station
    """
    station_location_df = pd.read_excel(Path(project_path) / 'data' / 'static' / config['station_location_file'])
    station_location_df = station_location_df[['code_tmd', 'name_tmd_en', 'main_basin', 'new_lat', 'new_long']]
    station_location_df['name_tmd_en'] = station_location_df['name_tmd_en'].apply(lambda _: _.replace('\t', ''))

    """
    * Load Thailand Shapefile
    """
    thailand_gdf = gpd.read_file(Path(project_path) / 'data' / 'shape_file' / 'Thai' / 'TH_WGS.shp')
    thailand_gdf["area"] = thailand_gdf.area

    """
    * Load Prediction
    """
    if config['last_prediction'] == '' or config['last_prediction'] == None or 'last_prediction' not in config :
        print('There are no prediction data')
        exit()

    hii_tmd_newforecast_test2122_df = pd.read_pickle(Path(project_path) / 'output' / config['last_prediction'] / 'simidx_v2' / 'output_df.bin')
    # print(hii_tmd_newforecast_test2122_df)

    station_pred_result = create_prediction_dict(hii_tmd_newforecast_test2122_df, 'SimIDXV2')
    
    """
    * Prepare map components
    """
    template_rds = pickle.load(open(Path(project_path) / 'data' / 'static' / 'template_rds_y1485_x829-v2.bin', 'rb'))
    pred_rds = xr.zeros_like(template_rds)

    if not (Path(project_path) / 'output' / date_path).exists() :
        os.mkdir(Path(project_path) / 'output' / date_path)
        
    if not (Path(project_path) / 'output' / date_path / 'rainfall_map').exists() :
        os.mkdir(Path(project_path) / 'output' / date_path / 'rainfall_map')

    """
    * Create Prediction Map
    Path(project_path) / 'output' / date_path / 'rainfall_map'
    """
    for n_year in range(len(station_pred_result['year_list'])) :
        this_year = station_pred_result['year_list'][n_year][0]
        print(this_year)

        for n_month in range(len(station_pred_result['month_list'][this_year])) :
            this_month = station_pred_result['month_list'][this_year][n_month]
            f_m = str(this_month) if len(str(this_month)) == 2 else '0'+str(this_month)
            print(f'Y{this_year} M{this_month}')

            """
            * Average files for Anamaly calculation
            """
            month_average = rioxarray.open_rasterio(Path(project_path) / 'data' / 'rainfall_map' / 'rain_anomaly' / f'avg30y_{f_m}_update202111.asc')

            prediction_rds = create_prediction_map(station_pred_result, station_location_df, n_year, n_month)
            
            """
            * Save Rainfall Map
            """
            hii_rainfall_plot(prediction_rds, template_rds, f'Y{this_year} M{this_month} SimIDX V2 Prediction',
                              save_location=Path(project_path) / 'output' / date_path / 'rainfall_map' / f'Y{this_year}M{this_month}_prediction.png')
            hii_rainfall_save_map(prediction_rds, template_rds, Path(project_path) / 'output' / date_path / 'rainfall_map' / f'Y{this_year}M{this_month}_prediction.tif')

            """
            * Anomaly Calculation and Save Map
            """
            anomaly_rds = xr.zeros_like(template_rds)
            anomaly_rds.attrs = {}
            anomaly_rds.values = prediction_rds.values - month_average.values

            month_average = month_average.rio.write_crs('EPSG:4326')
            anomaly_rds_mean = float(anomaly_rds.rio.clip(thailand_gdf.geometry, thailand_gdf.crs, drop=False).mean().values)
            month_average_mean = float(np.ma.masked_array(month_average.where(month_average >= 0).values, mask=np.where(month_average < 0, 1, 0)).mean())

            rainfall_diff = anomaly_rds_mean - month_average_mean
            rainfall_diff_percent = (rainfall_diff / month_average_mean) * 100

            hii_rainfall_anomaly_plot(anomaly_rds, template_rds, this_year, this_month, rainfall_diff, rainfall_diff_percent, f'Y{this_year} M{this_month} SimIDX V2 Prediction Anomaly',
                                      save_location=Path(project_path) / 'output' / date_path / 'rainfall_map' / f'Y{this_year}M{this_month}_prediction_anomaly.png')