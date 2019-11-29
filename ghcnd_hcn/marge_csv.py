import pandas as pd
import os

all_files = os.listdir("merge/")

# all_files = os.listdir("csv/")
# chunks = [all_files[x:x+100] for x in range(0, len(all_files), 100)]

# for i,chunk in enumerate(chunks):
df_from_each_file = (pd.read_csv("merge/"+f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
# concatenated_df.to_csv("merge{}.csv".format(i), index=False)
concatenated_df.to_csv("all.csv", index=False)