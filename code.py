import numpy as np
import pandas as pd


data = 'traffic_violaions.csv'
df = pd.read_csv(data, header=None)
#print(df.head())

col_names = ['date','time','cname','gender','raw_age','age','race','raw_violation','violation','search','type','outcome','arrested','duration','drugs']

df.columns = col_names
print(df.columns)
