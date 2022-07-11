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
    gdf = gpd.read_file( Path(project_path) / 'data' / 'shape_file' / 'MainBasin_ONWR_WGS84_4K_3A_With_Island/MainBasin_ONWR_WGS84.shp')
    gdf["area"] = gdf.area

    print(gdf)