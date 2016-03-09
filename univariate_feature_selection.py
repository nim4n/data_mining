import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df = pd.read_pickle('processed_data/dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
print features.shape

features = np.abs(features)
new_features = SelectKBest(chi2, k=1000).fit_transform(features, labels)

print new_features.shape
