import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold

df = pd.read_pickle('processed_data/dataframe_by_std.pd')
labels = df["Status"].values

kf = KFold(len(labels), n_folds=5, shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=1)
status = df['Status']
del df['Status']
for column in df.columns:
    features = df[column].values
    features = df[list([column])].values
    #print features
    means = []
    for training, testing in kf:
        classifier.fit(features[training], labels[training])
        prediction = classifier.predict(features[testing])
        mean = np.mean(prediction == labels[testing])
        means.append(mean)

    print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))





#df.to_pickle('processed_data/correlation_dataframe.pd')