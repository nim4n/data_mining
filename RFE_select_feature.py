from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import pandas as pd

df = pd.read_pickle('processed_data/dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=200, step=1)
rfe.fit(features, labels)
ranking = rfe.ranking_.reshape(features[0].shape)

print ranking