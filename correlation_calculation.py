import numpy as np
import pandas as pd

#read processed_dataframe from file
df = pd.read_pickle('processed_data/dataframe.pd')

#change catogorical data column to boolean
try:
    d = {'non-relapse': False, 'relapse': True}
    df['Status'] = df['Class'].map(d)
    del df['Class']
    df.to_pickle('processed_data/dataframe.pd')
except Exception as e:
    print e
#create new dict for saving correlation between column and Cancer
corr_dict = {}
corr_list = []
for i in df.columns:
    if i != 'Status':
        corr = df.Status.corr(df[i])
        corr_list.append(corr)
        corr_dict[i] = corr

del df['Status']
corr_df = pd.DataFrame(data={'correlations': np.array(corr_list)}, index=df.columns)
corr_df.to_pickle('processed_data/correlation_dataframe.pd')
print corr_df.describe()

#saving pearson correlation in a numpy file for future use
#np.save('processed_data/correlation.npy', corr_dict)

