import numpy as np
import pandas as pd

df = pd.read_pickle('processed_data/dataframe.pd')
corr_dict = np.load('processed_data/correlation.npy').item()
count = 0
for data in corr_dict:
    if corr_dict[data] > 0.19:
        count += 1
        print count
    else:
        try:
            del df[data]
        except:
            print data

df.to_pickle('processed_data/correlation_dataframe.pd')
