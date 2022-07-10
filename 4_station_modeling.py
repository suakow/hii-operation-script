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
import argparse
import json

import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as shc

from dtaidistance import dtw
from dtaidistance import dtw_ndim

import tensorflow as tf

config = yaml.load(
    open('config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

"""
* Station Prediction Model
"""
def build_model() :
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(12, 6)))
    model.add(tf.keras.layers.GRU(64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(12))

    return model

if __name__ == '__main__' :
    pass
