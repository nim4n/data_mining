import numpy as np
import pandas as pd
import time

df = pd.read_pickle('processed_data/dataframe_lung.pd')
counter = 0
for column in df.columns:
    counter += 1
    if counter % 100 == 0:
        time.sleep(2)
        counter = 1
        print 'sleep'
    if column != 'Status':
        #remove little expression
        remove_little_expression = df[column] < 20
        df[column][remove_little_expression] = 20
        if (df[column].max() - df[column].min()) < 500:
            del df[column]

        else:
            print df.shape
            std = df[column].std()
            remove_outlier = (df[column] < (2 * std)) & (df[column] > (-2 * std))
            df[column][~remove_outlier] = 20

df.to_pickle('processed_data/pre_process_dataframe_lung.pd')

