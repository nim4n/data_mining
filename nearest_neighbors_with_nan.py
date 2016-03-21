import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df = pd.read_pickle('processed_data/correlation_dataframe_with_nan_0.32.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
print len(features[0])
imp.fit(features)
features = imp.transform(features)
sel = VarianceThreshold(threshold=(.005))
features = sel.fit_transform(features)
print len(features[0])
kf = KFold(len(features), n_folds=4, shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
    imp.fit(features[training])
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))

