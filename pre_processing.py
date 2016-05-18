import numpy as np
import pandas as pd


df = pd.read_pickle('processed_data/dataframe_lung.pd')
counter = 0
for column in df.columns:
    try:
        if column != 'Status':
            #remove little expression
            remove_little_expression = df[column] < 20
            df[column][remove_little_expression] = 20
            if (df[column].max() - df[column].min()) < 500:
                del df[column]
                counter += 1
            else:
                print df.shape
                std = df[column].std()
                remove_outlier = (df[column] < (2 * std)) & (df[column] > (-2 * std))
                df[column][~remove_outlier] = 20
    except Exception as e:
        print e


df.to_pickle('processed_data/pre_process_dataframe_lung.pd')

