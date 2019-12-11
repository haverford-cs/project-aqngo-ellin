#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

df = pd.read_csv("final.csv")


station_list = df['STATION'].unique()
num_station = len(station_list)
for ix_, station in enumerate(station_list):
    orig = df.query("STATION == '{}'".format(station)).sort_values(by=['DATE'])
    n,p = orig.shape
    data = np.empty([n//2+1,15], dtype=object)

    for i in range(0,n,2):
        try:
            data[i//2] = np.concatenate((orig.values[i], orig.values[i+1][-5:]))
        except:
            continue
    A = pd.DataFrame(data=data,
          columns=['STATION', 'LAT', 'LON', 'ELEV', 'DATE', 
                   'PRCP1', 'TMAX1', 'TMIN1', 'SNOW1', 'SNOWD1', 
                   'PRCP2', 'TMAX2', 'TMIN2', 'SNOW2', 'SNOWD2'])
    A.to_csv("pair2/{}.csv".format(station), index=False)
    print("Finish {} out of {} stations".format(ix_, num_station))