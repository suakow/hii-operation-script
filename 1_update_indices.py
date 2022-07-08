__author__ = "Puri Phakmongkol"
__author_email__ = "me@puri.in.th"

"""
* HII Operation Script
*
* Created date : 05/07/2022
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
o      o  _-_-_-_- Update Indices
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

config = yaml.load(
    open('/home/studio-lab-user/sagemaker-studiolab-notebooks/workspace/hii_operation/config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

s_selected_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV','DEC']

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

def oni_mon_to_number(example) : 
    oni_mon_map = {
        'DJF' : 1,
        'JFM' : 2,
        'FMA' : 3,
        'MAM' : 4,
        'AMJ' : 5,
        'MJJ' : 6,
        'JJA' : 7,
        'JAS' : 8,
        'ASO' : 9,
        'SON' : 10,
        'OND' : 11,
        'NDJ' : 12
    }
    return oni_mon_map[example]

def dmi_mon_to_number(example) : 
    dmi_mon_map = {
        'jan' : 1,
        'feb' : 2,
        'mar' : 3,
        'apr' : 4,
        'may' : 5,
        'jun' : 6,
        'jul' : 7,
        'aug' : 8,
        'sep' : 9,
        'oct' : 10,
        'nov' : 11,
        'dec' : 12
    }
    return dmi_mon_map[example]

if __name__ == '__main__' :
    print('Updating ONI')
    
    oni_df = pd.read_fwf('https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt', sep=' ')
    oni_df.columns = ['oni_mon', 'year', 'sst34', 'oni']
    oni_df['oni_mon'] = oni_df['oni_mon'].map(lambda _ : oni_mon_to_number(_))
    oni_df = oni_df[['oni_mon', 'year', 'oni']]
    oni_df = oni_df.rename(columns={'oni_mon' : 'month'})
    
    print('Updating IOD')
    
    dmi_df = pd.read_csv('http://tiservice.hii.or.th/opendata/clmidx/dmi_had.csv', na_values=['-9999.000'])
    dmi_df = dmi_df.melt(id_vars=['year'], value_vars=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    dmi_df = dmi_df.rename(columns={'variable': 'month', 'value' : 'iod'})
    dmi_df['month'] = dmi_df['month'].map(lambda _: dmi_mon_to_number(_))
    dmi_df = dmi_df.sort_values(['year','month'])
    dmi_df = dmi_df[['month', 'year', 'iod']]
    
    print('Updating PDO')
    
    pdo_df = pd.read_table('https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat', skiprows=1, sep="\s+")
    
    for co in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] :
        pdo_df[co] = pdo_df[co].apply(lambda _ : float(str(_).replace('-99.99', '')))
        
    pdo_df.columns = ['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    pdo_df = pdo_df.melt(id_vars=['year'], value_vars=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    pdo_df = pdo_df.rename(columns={'variable': 'month', 'value' : 'pdo'})
    pdo_df['month'] = pdo_df['month'].map(lambda _: dmi_mon_to_number(_))
    pdo_df = pdo_df.sort_values(['year','month'])
    pdo_df = pdo_df[['month', 'year', 'pdo']]
    
    print('Updating EMI')
    emifile_data = requests.get('https://apcc21.org/cmm/fms/FileDown2.do;jsessionid=5FA508AC0BA3B7435DCB6BCAB7C5F527?atchFileId=EMI_2D.txt&fileSn=1').text
    emifile_data = emifile_data.split('\n')
    emifile_data = [ re.sub(r'\s+', ',', _) for _ in emifile_data ]
    emifile_data_str = functools.reduce(lambda a,b: a+'\n'+b, emifile_data)

    with io.StringIO() as f :
        f.write(emifile_data_str)
        f.seek(0)
        emi_df = pd.read_csv(f, na_values=-999.999)
        
    emi_df.columns = ['year'] + ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    emi_df = emi_df.melt(id_vars=['year'], value_vars=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    emi_df = emi_df.rename(columns={'variable': 'month', 'value' : 'emi'})
    emi_df['month'] = emi_df['month'].map(lambda _: dmi_mon_to_number(_))
    emi_df = emi_df.sort_values(['year','month'])
    emi_df = emi_df[['month', 'year', 'emi']]

    print('Updating MJO')
    mjo_files = requests.get('http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt',
        headers={
            'User-Agent' : 'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)',
        }).text

    file_data = mjo_files.split('\n')
    file_data = file_data[1:]
    file_data[0] = ',' + file_data[0]
    file_data = [ file_data[0]] + [ re.sub(r'\s+', ',', _) for _ in file_data[1:] ]
    file_data = [ _[1:] if (_[-1] != ',') else _[1: -1] for _ in file_data if ('Missing_value' not in _) and len(_) != 0 ]

    file_data_str = functools.reduce(lambda a,b: a+'\n'+b, file_data)

    with io.StringIO() as f :
        f.write(file_data_str)
        f.seek(0)
        mjo_df = pd.read_csv(f)
        
    mjo_df = mjo_df.reset_index()
    mjo_df.columns = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'note']
    mjo_df['th_year'] = mjo_df['year'] + 543
    
    print('Updating Monsoon')
    
    ind_df = pd.read_csv('http://tiservice.hii.or.th/opendata/era_clim_indices/ts.monsoon_ind.im', sep=' ', header=None)
    ind_df.columns = ['count', 'date', 'im_value']
    
    wp_df = pd.read_csv('http://tiservice.hii.or.th/opendata/era_clim_indices/ts.monsoon_ind.wnpm', sep=' ', header=None)
    wp_df.columns = ['count', 'date', 'wp_value']
    
    monsoon_df = wp_df.drop(columns=['count']).join(ind_df.drop(columns=['count']).set_index('date'), on='date', how='left')
    
    print('Merge MJO + Monsoon')
    
    mjo_df['date'] = mjo_df.apply(lambda _: str(_['year'])+'-'+ (str(_['month']) if len(str(_['month'])) == 2 else '0' + str(_['month'])) + '-' + (str(_['day']) if len(str(_['day'])) == 2 else '0' + str(_['day'])), axis=1)
    
    monsoon_mjo_df = monsoon_df.join(mjo_df.set_index('date'), on='date', how='left')
    
    print('Preparing save path')
    
    if not Path(f'%sdata/{date_path}'%(config['project_path'])).exists() :
        os.mkdir(Path(f'%sdata/{date_path}'%(config['project_path'])))
        
    monsoon_mjo_df.to_csv(Path(f'%sdata/{date_path}'%(config['project_path'])) / 'mjo_monsoon_indices.csv', index=False)
    oni_df.to_csv(Path(f'%sdata/{date_path}'%(config['project_path'])) / 'oni_df.csv', index=False)
    dmi_df.to_csv(Path(f'%sdata/{date_path}'%(config['project_path'])) / 'dmi_df.csv', index=False)
    pdo_df.to_csv(Path(f'%sdata/{date_path}'%(config['project_path'])) / 'pdo_df.csv', index=False)
    emi_df.to_csv(Path(f'%sdata/{date_path}'%(config['project_path'])) / 'emi_df.csv', index=False)
    
    config['last_update'] = date_path
    print(config)
    open('%sconfig.yml'%(config['project_path']), 'w').write(yaml.dump(config))
    
    print('Done')