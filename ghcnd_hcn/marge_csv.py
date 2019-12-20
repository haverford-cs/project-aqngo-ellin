"""Intermediate pre-processing file to merge all independent dataframes into
one big dataframe for all stations.
"""

import pandas as pd
import os

all_files = os.listdir("pair2/")

# for i,chunk in enumerate(chunks):
df_from_each_file = (pd.read_csv("pair2/"+f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df = concatenated_df.dropna()
concatenated_df.to_csv("pair_final.csv", index=False)