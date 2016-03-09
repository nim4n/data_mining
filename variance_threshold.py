import pandas as pd


df = pd.read_pickle('processed_data/dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values