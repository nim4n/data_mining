
import pandas as pd
import pickle
from sklearn import svm

df = pd.read_pickle('processed_data/correlation_dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
clf = svm.SVC()
clf.fit(features, labels)
s = pickle.dumps(clf)
