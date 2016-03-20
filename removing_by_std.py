import numpy as np
import pandas as pd


df = pd.read_pickle('processed_data/dataframe.pd')
for column in df.columns:
    if column != 'Status':
        std = df[column].std()
        remove_outlier = (df[column] < (2 * std)) & (df[column] > (-2 * std))
        df[column][~remove_outlier] = np.NaN
df.to_pickle('processed_data/dataframe_by_std.pd')
