import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold

#this is my first try for predecting
df = pd.read_pickle('processed_data/dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values

#this choose 5 sample data for test and training data
kf = KFold(len(features), n_folds=5, shuffle=True)

classifier = KNeighborsClassifier(n_neighbors=2)

means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))

