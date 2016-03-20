import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
df = pd.read_pickle('processed_data/correlation_dataframe_with_nan_0.25.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
kf = KFold(len(features), n_folds=5, shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=7)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
    imp.fit(features[training])
    training_imp = imp.transform(features[training])
    testing_imp = imp.transform(features[testing])
    classifier.fit(training_imp, labels[training])
    prediction = classifier.predict(testing_imp)
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))

