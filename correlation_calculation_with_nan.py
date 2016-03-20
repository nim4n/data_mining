import numpy as np
import pandas as pd

df = pd.read_pickle('processed_data/dataframe_by_std.pd')
#create new dict for saving correlation between column and Cancer
corr_dict = {}
corr_list = []

for column in df.columns:
    if column != 'Status':
        corr = df.Status.corr(df[column])
        corr_list.append(corr)
        corr_dict[column] = corr

del df['Status']
corr_df = pd.DataFrame(data={'correlations': np.array(corr_list)}, index=df.columns)

#saving pearson correlation in a numpy file for future use
np.save('processed_data/correlation_with_nan.npy', corr_dict)

