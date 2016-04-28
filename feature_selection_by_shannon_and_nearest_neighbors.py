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
        if marker_df[col1] > 0.06:
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


def marker_dataframe_generator(df):
    #create dataframe base on shannon rank of data
    count = 0
    result = {'marker': {}}
    for col1 in df.columns:
        count += 1
        result['marker'][col1] = shannon_normolizer(col1, 'Marker')
        print count

    marker_df = pd.DataFrame.from_dict(data=result)
    marker_df = marker_df.sort(['marker'], ascending=[0])
    print marker_df.describe()
    marker_df.to_pickle('processed_data/marker_dataframe.pd')


def find_little_entropy(columns):
    #create dataframe base on shannon rank of data
    count = 0
    result = {'total_entropy': {}}
    pair_list = []
    for col1 in df.columns:
        if col1 != 'Marker':
            total_entropy = 0
            count += 1
            result[col1] = {}
            print count, col1
            counter = 0
            for col2 in columns:
                if col1 != col2 and ((col1, col2) not in pair_list) and col2 != 'Marker':
                    if counter % 1000 == 0:
                        print datetime.datetime.now()
                    counter += 1
                    pair_list.append((col2, col1))
                    caculate_result = shannon_normolizer(col1, col2)
                    total_entropy += caculate_result
                    result[col1][col2] = caculate_result
            result['total_entropy'][col1] = total_entropy
    shannon_nearest_df = pd.DataFrame.from_dict(data=result)
    shannon_nearest_df = shannon_nearest_df.sort(['total_entropy'], ascending=[1])
    print shannon_nearest_df.describe()
    shannon_nearest_df.to_pickle('processed_data/shannon_nearest_entropy_dataframe.pd')


def select_feature(best_entropy_df):
    best_entropy_df = best_entropy_df.sort(['total_entropy'], ascending=[1])
    print best_entropy_df
    selected_df = df[best_entropy_df[:7].index.tolist()].copy()
    print selected_df.shape
    selected_df['Status'] = status
    print selected_df.shape
    selected_df.to_pickle('processed_data/feature_by_shannon_dataframe.pd')

def calculate_best_col():
    best_entropy_df = lt_entropy_df.copy()
    for col in lt_entropy_df.index.tolist():
        best_entropy_df.loc[col, 'total_entropy'] = marker_df.loc[col, 'marker'] - lt_entropy_df.loc[col, 'total_entropy']
    best_entropy_df = best_entropy_df.sort(['total_entropy'], ascending=[0])
    print best_entropy_df.describe()
    best_entropy_df.to_pickle('processed_data/total_feature_by_shannon_dataframe.pd')

df = pd.read_pickle('processed_data/pre_process_dataframe.pd')
d = {True: 2, False: 1}
marker = df['Status'].map(d)

status = df['Status']
del df['Status']
chunks = [df.columns[40*i:40*(i+1)] for i in range(len(df.columns)/40 + 1)]
df['Marker'] = marker
feature_selected_by_classification = [u'1898_at', u'32166_at', u'32314_g_at', u'32780_at', u'33371_s_at',
                                      u'33891_at', u'34162_at', u'34407_at', u'34775_at', u'35177_at',
                                      u'35742_at', u'35803_at', u'36040_at', u'36159_s_at', u'36192_at',
                                      u'36627_at', u'36659_at', u'36792_at', u'37225_at', u'37230_at',
                                      u'37366_at', u'37630_at', u'37639_at', u'37716_at',  u'33412_at',
                                      u'37958_at', u'38028_at', u'38047_at', u'38396_at', u'38717_at',
                                      u'38772_at', u'39099_at', u'39243_s_at', u'39366_at', u'39714_at',
                                      u'39940_at', u'40069_at', u'40071_at', u'40113_at', u'40567_at',
                                      u'40841_at', u'41388_at', u'41468_at', u'575_s_at']

#entropy dataframe
#entropy_df = df.loc[:].apply(entropy, axis=0)
entropy_df = pd.read_pickle('processed_data/shannon_dataframe.pd')
#marker_dataframe_generator(df)
#entropy_df.to_pickle('processed_data/shannon_dataframe.pd')
marker_df = pd.read_pickle('processed_data/marker_dataframe.pd')
#marker_df = marker_df.sort(['marker'], ascending=[0])
#find_little_entropy(marker_df[:200].index.tolist())
#lt_entropy_df = pd.read_pickle('processed_data/lt_entropy_dataframe.pd')
#best_entropy_df = pd.read_pickle('processed_data/total_feature_by_shannon_dataframe.pd')
#calculate_best_col()
#select_feature(best_entropy_df)
find_little_entropy(feature_selected_by_classification)



'''
for count, chunk in enumerate(chunks):

    result = calculate_rank(chunk)
    f = open('processed_data/entropy_result.txt', 'a')
    f.write('{0}     {1}\n'.format(count, result))
    f.close()
    print count
'''







