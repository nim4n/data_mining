import math
import pandas as pd


df = pd.read_pickle('processed_data/pre_process_dataframe.pd')

d = {True: 2, False: 1}
df['Marker'] = df['Status'].map(d)
print df['Marker']
status = df['Status']
del df['Status']

def shannon(col):
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in col])
    return entropy

sh_df = df.loc[:].apply(shannon, axis=0)
sh_df.to_pickle('processed_data/shannon_dataframe.pd')

print(sh_df)
