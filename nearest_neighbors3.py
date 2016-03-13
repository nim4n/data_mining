import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold

df = pd.read_pickle('processed_data/correlation_dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
kf = KFold(len(features), n_folds=10, shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
new_features = sel.fit_transform(features)
print len(features[0]), len(new_features[0])
