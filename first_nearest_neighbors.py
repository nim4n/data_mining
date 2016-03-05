import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold


df = pd.read_pickle('processed_data/dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
print len(labels)
print len(features)
print labels
'''
#this choose 5 sample data for test and training data
kf = KFold(len(features), n_folds=5, shuffle=True)

classifier = KNeighborsClassifier(n_neighbors=1)

means = []
for training,testing in kf:
    # We learn a model for this fold with `fit` and then apply it to the
    # testing data with `predict`:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])

    # np.mean on an array of booleans returns fraction
    # of correct decisions for this fold:
    curmean = np.mean(prediction == labels[testing])
    means.append(curmean)
print('Mean accuracy: {:.1%}'.format(np.mean(means)))

'''