import numpy as np
import pandas as pd

df = pd.read_pickle('processed_data/dataframe_by_diff_greater_0.05.pd')
corr_dict = np.load('processed_data/correlation_with_nan_diff.npy').item()
count = 0
for data in corr_dict:
    if corr_dict[data] > 0.35:
        count += 1
        print count
    else:
        try:
            del df[data]
        except:
            print data

df.to_pickle('processed_data/correlation_dataframe_with_nan_diff_0.35.pd')
