from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold

df = pd.read_pickle('processed_data/dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
#define threshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
new_features = sel.fit_transform(features)
print len(features[0]), len(new_features[0])

kf = KFold(len(new_features), n_folds=5, shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=4)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
    classifier.fit(new_features[training], labels[training])
    prediction = classifier.predict(new_features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))

