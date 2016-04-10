import numpy as np
import pandas as pd
from sklearn import random_projection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold

df = pd.read_pickle('processed_data/rank_classification_dataframe_remove_relation.pd')
labels = df["Status"].values
del df['Status']
print df.columns
#df = pd.read_pickle('processed_data/rank_classification_dataframe.pd')
features = df[list(df.columns)].values
print features.shape
#transformer = random_projection.GaussianRandomProjection(n_components=82)
#features = transformer.fit_transform(features)
print features.shape


kf = KFold(len(features), n_folds=7, shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=1)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))


