import pandas as pd
from scipy.io import arff

f = open('data/lungCancer_test.arff', 'r')
f2 = open('data/lungCancer_train.arff', 'r')


data, meta = arff.loadarff(f)
data2, meta2 = arff.loadarff(f2)
columns = {}
counter = 0
for i in meta:
    new_list = []
    for j in data:
        new_list.append(j[counter])
    for b in data2:
        new_list.append(b[counter])
    columns[i] = new_list
    counter += 1
df = pd.DataFrame(data=columns)
d = {'Mesothelioma': False, 'ADCA': True}
df['Status'] = df['Class'].map(d)
del df['Class']
df.to_pickle('processed_data/dataframe_lung.pd')
