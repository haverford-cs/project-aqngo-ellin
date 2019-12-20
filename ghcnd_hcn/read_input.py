"""Parse the data into a dataframe and then write it to .csv file.
Note that the data/ folder has been ommited from the Github Repository.
The folder can be downloaded from NOOA Data Center Online Website

Author: Jason Ngo
Authors: Jason Ngo and Emily Lin
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, date


def get_location(station_id):
    """Given a station_id, return its location
    
    Arguments:
        station_id {str} -- Station ID
    
    Returns:
        [tuple] -- [(lon, lat, elev) of the station]
    """
    f = open("ghcnd-stations.txt")
    lines = f.readlines()
    f.close()
    for row in lines:
        row = row.split()
        if row[0] == station_id:
            location = np.array([row[1], row[2], row[3]]).astype(float)
            return location
    return -1


def get_source_df(file):
    """Parse the source data file into a pandas dataframe
    
    Arguments:
        file {str} -- File name for each station
    
    Returns:
        df -- Dataframe for that station
    """
    f = open("data/"+test_file)
    lines = f.readlines()
    f.close()
    num_lines = len(lines)

    VARIABLES = ["ID", "YEAR", "MONTH", "ELEMENT"]
    for i in range(1, 32):
        VARIABLES.append("VALUE"+str(i))
        VARIABLES.append("MFLAG"+str(i))
        VARIABLES.append("QFLAG"+str(i))
        VARIABLES.append("SFLAG"+str(i))

    rows_list = []
    for i in range(num_lines):
        dict1 = {}
        dict1[VARIABLES[0]] = lines[i][0:11]
        dict1[VARIABLES[1]] = lines[i][11:15]
        dict1[VARIABLES[2]] = lines[i][15:17]
        dict1[VARIABLES[3]] = lines[i][17:21]

        start = 21
        for j in range(4, len(VARIABLES)):
            if j % 4 == 0:
                end = start+5
            else:
                end = start+1
            dict1[VARIABLES[j]] = lines[i][start:end]
        rows_list.append(dict1)

    src_df = pd.DataFrame(rows_list)
    for i in range(1, 32):
        src_df = src_df.drop(
            ["MFLAG{}".format(i), "QFLAG{}".format(i), "SFLAG{}".format(i)], axis=1)
        src_df["VALUE{}".format(i)] = src_df["VALUE{}".format(i)].astype(int)

    src_df = src_df.query("ELEMENT in @core_vars")

    src_prcp_df = src_df.query("ELEMENT == 'PRCP'")
    src_tmax_df = src_df.query("ELEMENT == 'TMAX'")
    src_tmin_df = src_df.query("ELEMENT == 'TMIN'")
    src_snow_df = src_df.query("ELEMENT == 'SNOW'")
    src_snwd_df = src_df.query("ELEMENT == 'SNWD'")

    src_df_lst = [src_prcp_df, src_tmax_df,
                  src_tmin_df, src_snow_df, src_snwd_df]
    return src_df_lst


def get_clean_df(src_df_lst):
    """Given a list of sources dataframes, drop missing values and merge
    them into a massive dataframe for all the stations
    
    Arguments:
        src_df_lst {lst} -- list of station-by-staion dataframes
    
    Returns:
        df -- a merged dataframe for all the stations
    """
    out_df_lst = []
    for i, src_dataframe in enumerate(src_df_lst):
        rows_list = []
        input_rows = src_dataframe.values
        for row in input_rows:
            for k in range(1, 32):
                dict1 = dict()
                # get input row in dictionary format
                if k < 10:
                    day = "0" + str(k)
                else:
                    day = str(k)
                dict1['DATE'] = row[1] + "-" + row[2] + "-" + day
                dict1[core_vars[i]] = row[k+3]
                rows_list.append(dict1)

        out_df = pd.DataFrame(rows_list).set_index('DATE')
        out_df_lst.append(out_df)

    clean_df = pd.concat(out_df_lst, axis=1, sort=False)
    clean_df = clean_df.replace(-9999, np.nan)
    clean_df = clean_df.dropna(axis=0)
    clean_df = clean_df.reset_index().rename(columns={"index": "DATE"})
    return clean_df


def insert_location(clean_df, station_id):
    """Since the dataframe does not contain station ID, this function inserts
    three columns into the dataframe to indicate each station's location
    
    Arguments:
        clean_df {df} -- a cleaned dataframe for all the stations
        station_id {str} -- Station ID
    
    Returns:
        df -- a dataframe with all station information
    """
    lat, lon, elev = get_location(station_id)
    nrows = clean_df.shape[0]
    clean_df.insert(0, "STATION", [station_id]*nrows)
    clean_df.insert(1, "LAT", [lat]*nrows)
    clean_df.insert(2, "LON", [lon]*nrows)
    clean_df.insert(3, "ELEV", [elev]*nrows)
    return clean_df


if __name__ == "__main__":
    all_files = os.listdir("data/")
    for j_ in range(len(all_files)):
        test_file = all_files[j_]
        station_id = str(test_file.split('.')[0])

        f = open("data/"+test_file)
        lines = f.readlines()
        f.close()
        core_vars = ['PRCP', 'TMAX', 'TMIN', 'SNOW', 'SNWD']

        src_df_lst = get_source_df(f)
        clean_df = get_clean_df(src_df_lst)
        clean_df = insert_location(clean_df, station_id)

        clean_df.to_csv("csv/{}.csv".format(station_id), index=False)
