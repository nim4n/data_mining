import pandas as pd
from sklearn.feature_selection import VarianceThreshold

df = pd.read_pickle('processed_data/dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
#define threshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
new_features = sel.fit_transform(features)
print len(features[0]), len(new_features[0])

