import pandas as pd
from scipy.io import arff

f = open('data/breastCancer-train.arff', 'r')


data, meta = arff.loadarff(f)
index = {}
counter = 0
for i in meta:
    new_list = []
    for j in data:
        new_list.append(j[counter])
    index[i] = new_list
    counter += 1
df = pd.DataFrame(data=index)
df.to_pickle('processed_data/dataframe.pd')
