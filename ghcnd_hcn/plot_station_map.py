"""This function returns a map of all the stations scattered across the 
United States
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from read_input import get_location


df = pd.read_csv("pair_final.csv")

locations = np.array([
    get_location(station_id) for station_id in df['STATION'].unique()
])
new_df = pd.DataFrame()
new_df['LON'] = locations[:, 0]
new_df['LAT'] = locations[:, 1]
new_df['STATION'] = df['STATION'].unique()
new_df = new_df.assign(cnt=1)
new_df


fig = go.Figure(data=go.Scattergeo(
    lon=new_df['LAT'],
    lat=new_df['LON'],
    text=new_df['STATION'],
    mode='markers',
    marker_size=8,
    marker_color=new_df['cnt'],
))

fig.update_layout(
    title="Map of HCN Weather Stations",
    geo_scope='usa',
)

fig.show()
