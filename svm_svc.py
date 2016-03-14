import pickle
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold

df = pd.read_pickle('processed_data/correlation_dataframe.pd')
labels = df["Status"].values
del df['Status']
features = df[list(df.columns)].values
classifier = svm.SVC()
classifier.fit(features, labels)
kf = KFold(len(features), n_folds=5, shuffle=True)
print len(labels)




means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))
