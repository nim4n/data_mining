import numpy as np
import pandas as pd

#read processed_dataframe from file
df = pd.read_pickle('processed_data/dataframe.pd')

#change catogorical data column to boolean
d = {'non-relapse': False, 'relapse': True}
df['Status'] = df['Class'].map(d)
del df['Class']

#create new dict for saving correlation between column and Cancer
new_dict = {}
for i in df.columns:
    corr = df.Status.corr(df[i])
    new_dict[i] = corr

#saving pearson correlation in a numpy file for future use
np.save('processed_data/correlation.npy', new_dict)

