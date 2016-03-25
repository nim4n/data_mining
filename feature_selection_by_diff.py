import pandas as pd

df = pd.read_pickle('processed_data/dataframe_by_std.pd')

positice_df = df.loc[df['Status'] == True]

negative_df = df.loc[df['Status'] == False]

print positice_df.shape

print negative_df.shape

counter = 0
for column in df.columns:
    if column != 'Status':
        diff = positice_df[column].mean() - negative_df[column].mean()
        if diff > 0.05:
            counter += 1
        else:
            del df[column]
        print diff
print counter
#diff_df = pd.DataFrame(data=new_dict, index=df.columns)
#print diff_df
df.to_pickle('processed_data/dataframe_by_diff_greater_0.05.pd')
