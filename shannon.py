import math
import itertools
import pandas as pd
import numpy as np
import datetime


def shannon(col):
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in col])
    return entropy


def entropy(*X):
    result = np.sum(-p * np.log2(p) if p > 0 else 0 for p in
                    (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
                    for classes in itertools.product(*[set(x) for x in X])))
    return result


def shannon_normolizer(col1, col2):
    return ((entropy_df[col1] + entropy_df[col2]) - entropy(df[col1].values, df[col2].values)) / (entropy_df[col1] + entropy_df[col2])



def calculate_rank(chunk):
    pair_list = []
    total_marker = 0
    total_with_each_other = 0
    entropy_with = {}
    for col1 in chunk:
        entropy_with[col1] = {}
        marker = shannon_normolizer(col1, 'Marker')
        total_marker += marker
        entropy_with[col1]['Marker'] = marker
        for col2 in chunk:
            if col1 != col2 and ((col1, col2) not in pair_list):
                pair_list.append((col2, col1))
                eachother = shannon_normolizer(col1, col2)
                entropy_with[col1][col2] = eachother
                total_with_each_other += eachother
    print total_marker - total_with_each_other
    print '____________________________________________-'
    return total_marker - total_with_each_other


def dataframe_generator(df):
    #create dataframe base on shannon rank of data
    result = {}
    pair_list = []
    count = 0
    for col1 in df.columns:
        count += 1
        result[col1] = {}
        result[col1]['Marker'] = shannon_normolizer(col1, 'Marker')
        counter = 0
        for col2 in df.columns:
            if col1 != col2 and ((col1, col2) not in pair_list):
                if counter % 1000 == 0:
                    print count, counter
                    print datetime.datetime.now()
                counter += 1
                pair_list.append((col2, col1))
                result[col1][col2] = shannon_normolizer(col1, col2)
        new_df = pd.DataFrame.from_dict(data=result)
    new_df.to_pickle('processed_data/entropy.pd')




df = pd.read_pickle('processed_data/pre_process_dataframe.pd')
d = {True: 2, False: 1}
marker = df['Status'].map(d)

status = df['Status']
del df['Status']
chunks = [df.columns[40*i:40*(i+1)] for i in range(len(df.columns)/40 + 1)]
df['Marker'] = marker
print len(df.columns)
#entropy dataframe
entropy_df = df.loc[:].apply(shannon, axis=0)
#dataframe_generator(df)
#entropy_df.to_pickle('processed_data/shannon_dataframe.pd')

'''
for count, chunk in enumerate(chunks):

    result = calculate_rank(chunk)
    f = open('processed_data/entropy_result.txt', 'a')
    f.write('{0}     {1}\n'.format(count, result))
    f.close()
    print count
'''







