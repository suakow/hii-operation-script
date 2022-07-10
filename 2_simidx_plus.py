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

config = yaml.load(
    open('config.yml', 'r').read(),
    Loader=SafeLoader
)

print(config)

project_path = config['project_path']
latest_indices_update = config['last_update']

time_now = datetime.datetime.now()
date_path = time_now.isoformat().split('T')[0]

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

def cal_distance_metrix(data, distance_function) :
    n_sample = data.shape[0]
    result = []
    for x in range(n_sample) :
        x_result = []
        for y in range(n_sample) :
            distance = distance_function(data[x], data[y])
            x_result.append(distance)

        result.append(x_result)

    return np.array(result)

def dwt_distance2(p1, p2) :
    _p1 = np.array([ _ for _ in p1 if ~np.isnan(_)])
    _p2 = np.array([ _ for _ in p2 if ~np.isnan(_)])

    distance = dtw.distance(_p1, _p2)
    return distance

def multi_dwt_distance(p1, p2) :
    _p1 = p1
    _p2 = p2

    distance = dtw_ndim.distance(_p1, _p2)
    return distance

def compare_distance(sryear, targetyears) :
    distance_dict = {}
    x_sryear = s_df[s_df['f_thyear'] == sryear][list(sub_df.columns)].values.reshape(-1, 4, 12)[0]
    
    # Current Dataset does not contain 12month of overall rainfall after 2562
    for _ in targetyears :
        if _ >= 2562 :
            continue

        x_taryear = s_df[s_df['f_thyear'] == _][list(sub_df.columns)].values.reshape(-1, 4, 12)[0]
        dis = multi_dwt_distance(x_sryear, x_taryear)
        distance_dict[_] = dis
    year_list = list(distance_dict.keys())
    dist_list = list(distance_dict.values())
    # print(min(dist_list))
    return year_list[dist_list.index(min(dist_list))]


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
    rain_df = pd.read_csv(Path(f'{project_path}data/static') / config['annual_rainfall_file'])
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

    if latest_indices_update == '' or latest_indices_update == None :
        print('There are no indices data')
        exit()

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
    
    """
    * Define Label
    """
    
    s_df['label'] = s_df.apply(lambda _: '%s_%s_%s_%s'%(_['f_year'], _['f_thyear'], _['group'], _['sRain']), axis=1)
    s_df['label_rain_th'] = s_df.apply(lambda _: '%s(%s mm)'%(_['f_thyear'], _['sRain']), axis=1)
    s_df['label_th'] = s_df['f_thyear']
    
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
    * Heirarchical Clustering
    """
    
    sub_data_dwt = np.array([ np.stack(_, axis=1) for _ in sub_df.values.reshape(-1, 4, 12) ])
    sub_distance_matrix = cal_distance_metrix(sub_data_dwt, multi_dwt_distance)
    distance_matrix = ssd.squareform(sub_distance_matrix)
    
    clustering_result = shc.dendrogram(shc.linkage(distance_matrix, method='average'), 
               labels=s_df['label_th'].values, 
               orientation='right')
    
    groups = shc.cut_tree(shc.linkage(sub_df, method='average'), n_clusters=12)[:, 0]
    
    # Save Figure
    if not Path(f'{project_path}output/{date_path}').exists() :
        os.mkdir(Path(f'{project_path}output/{date_path}'))
        
    if not Path(f'{project_path}output/{date_path}/simidx_plus').exists() :
        os.mkdir(Path(f'{project_path}output/{date_path}/simidx_plus'))
        
    plt.savefig(Path(f'{project_path}output/{date_path}/simidx_plus/clustering.png'), format='png', bbox_inches='tight')
    
    """
    * Extract Clustering result
    """
    icoord = np.array(clustering_result['icoord'])
    dcoord = np.array(clustering_result['dcoord'])
    color_list = np.array(clustering_result['color_list'])
    result_year_list = np.array(clustering_result['ivl'])
    
    current_group = ''
    group_result = {}
    group_result_long = {}
    grouped_item = []
    in_groupitem = []
    group_item_loc = 0

    """
    * Extract Clustering Process
    """
    # Step 1 : Find Group
    for tree_value in list(zip(icoord, dcoord, color_list)) :
        if tree_value[2] != current_group :
            current_group = tree_value[2]
            if len(in_groupitem) != 0 :
                grouped_item.append(in_groupitem)
                in_groupitem = []

        in_groupitem.append(tree_value)

    grouped_item.append(in_groupitem)
    # print(grouped_item)

    # Step 1.5 : Assign only 1 members group to others group
    skip_item = []
    skip_i = 0
    skip_i_list = []
    for item in grouped_item[-1] :
        if item[1][0] == 0.0 or item[1][-1] == 0.0 :
            skip_item.append(item)
            skip_i_list.append(skip_i)

        skip_i += 1

    # Find average location of each group
    group_avg_location = {}
    locaion_i = 0
    for gg in grouped_item[:-1] :
        group_location = [ _[0][0] for _ in gg ] + [ _[0][-1] for _ in gg ]
        group_avg_location[locaion_i] = np.mean(group_location)
        locaion_i += 1

    for sk in skip_item :
        sk_location = sk[0][0] if sk[1][0] == 0 else sk[0][-1]
        sk_distance = [ abs(_ - sk_location) for _ in group_avg_location.values() ]
        sk_group = sk_distance.index(min(sk_distance))
        grouped_item[sk_group].append(sk)

    temp = []
    for II in range(len(grouped_item[-1])) :
        if II not in skip_i_list :
            temp.append(grouped_item[-1][II])
    grouped_item[-1] = temp

    # End Step 1.5

    for in_group in grouped_item :
        group_mem = []
        in_group_height_list = {}
        group_no = 1
        # print(in_group)

        # Step 2 : Find Directly Match Item
        for item in in_group :
            # print(item)
            in_match = False
            match_group = 0
            if item[1][0] == 0 and item[1][-1] == 0 :
                in_group_height_list[group_no] = item[1][1]
                match_group = group_no
                group_no += 1
                in_match = True

            if item[1][0] == 0 :
                group_mem.append({
                    'location' : item[0][0],
                    'height' : item[1][1],
                    'match' : in_match,
                    'match_height' : item[1][-1],
                    'group' : match_group,
                    'sub_group' : 0,
                })

            if item[1][-1] == 0 :
                group_mem.append({
                    'location' : item[0][-1],
                    'height' : item[1][-2],
                    'match' : in_match,
                    'match_height' : item[1][0],
                    'group' : match_group,
                    'sub_group' : 0,
                })

        # print(group_mem)

        #Step 3 : Match non-matched item to nearest group
        for item in range(len(group_mem)) :
            # print(group_mem[item]) 
            if not(group_mem[item]['match']) :
                dif_height = {}
                min_k = 99
                min_v = 99
                for k, v in in_group_height_list.items() :
                    dif_height[k] = abs(group_mem[item]['match_height'] - v)
                    if dif_height[k] < min_v :
                        min_v = dif_height[k]
                        min_k = k

                group_mem[item]['sub_group'] = min_k
                group_mem[item]['group'] = group_no
                in_group_height_list[group_no] = group_mem[item]['height']
                group_no += 1

        # Step 4 : Sort item by location
        group_mem_s = sorted(group_mem, key = lambda i: i['location'])
        # print(group_mem_s)

        # Step 5 : Assign pair of year
        in_year = result_year_list[group_item_loc : group_item_loc + len(group_mem_s)]
        group_item_loc = group_item_loc + len(group_mem_s)
        for i in range(len(in_year)) : 
            group_mem_s[i]['year'] = in_year[i]

            pair_year_index = []
            if group_mem_s[i]['match'] :
                for j in range(len(group_mem_s)) :
                    if group_mem_s[i]['group'] == group_mem_s[j]['group'] :
                        pair_year_index.append(j)

            else :
                for j in range(len(group_mem_s)) :
                    if group_mem_s[i]['sub_group'] == group_mem_s[j]['group'] :
                        pair_year_index.append(j)

            pair_year = [ in_year[_] for _ in pair_year_index if in_year[_] != in_year[i]]
            group_mem_s[i]['year_pair'] = pair_year
            # print(pair_year)

        # print(group_mem_s)
        for _ in group_mem_s :
            # print(_)
            group_result[_['year'].tolist()] = [ xx.tolist() for xx in _['year_pair']]
            group_result_long[_['year']] = _
    
    # print(group_result)
    # print(type(list(group_result.items())[0][0]))
    # print(list(group_result.items())[0][0])
    # group_result = { k.tolist(): v.tolist() for k, v in list(group_result.items()) }
    
    # group_result = zip([ _.tolist() for _ in group_result.keys() ], [ _.tolist() if type(_) != 'list' else [ x.tolist() for x in _ ] for _ in group_result.values() ])
    
    
    
    """
    * End Extract Clustering Process
    """
    group_result_filtered = { _: g[0] if len(g) == 1 else compare_distance(_, g) for _, g in group_result.items() }
    # group_result_filtered = { k.tolist(): v.tolist() for k, v in list(group_result_filtered.items()) }
    
    print(group_result_filtered)
    
    """
    * Save clustering result
    """
    if not Path(f'{project_path}output/{date_path}').exists() :
        os.mkdir(Path(f'{project_path}output/{date_path}'))
        
    if not Path(f'{project_path}output/{date_path}/simidx_plus').exists() :
        os.mkdir(Path(f'{project_path}output/{date_path}/simidx_plus'))
    
    open(Path(f'{project_path}output/{date_path}/simidx_plus/result.json'), 'w').write(json.dumps(group_result))
    open(Path(f'{project_path}output/{date_path}/simidx_plus/result_filter.json'), 'w').write(json.dumps(group_result_filtered))
    open(Path(f'{project_path}output/{date_path}/simidx_plus/config.json'), 'w').write(json.dumps(config))
    
    config['simidx_last_update'] = date_path
    print(config)
    open('%sconfig.yml'%(config['project_path']), 'w').write(yaml.dump(config))
    