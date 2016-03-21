import numpy as np
import pandas as pd

df = pd.read_pickle('processed_data/dataframe_by_std.pd')
corr_dict = np.load('processed_data/correlation_with_nan.npy').item()
count = 0
for data in corr_dict:
    if corr_dict[data] > 0.32:
        count += 1
        print count
    else:
        try:
            del df[data]
        except:
            print data

df.to_pickle('processed_data/correlation_dataframe_with_nan_0.32.pd')
