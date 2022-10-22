from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd


data = 'traffic_violaions.csv'
df = pd.read_csv(data, header=None)
#df.head()

col_names = ['date','time','cname','gender','raw_age','age','race','raw_violation','violation','search','type','outcome','arrested','duration','drugs']

df.columns = col_names
print(df.columns)

categorical = [var for var in df.columns if df[var].dtype == 'O']
print(df[categorical].isnull().sum())
print(df.dropna(thresh=2))

X = df.drop(['arrested'], axis=1)

y = df['arrested']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

model = BernoulliNB()
