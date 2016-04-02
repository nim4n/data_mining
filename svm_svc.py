import pickle
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold

df = pd.read_pickle('processed_data/rank_svm_classification_dataframe_remove_relation.pd')
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


'''
output with correlation greater than 0.1 :

Fold predicting accuracy mean is: 37.5%
Fold predicting accuracy mean is: 56.2%
Fold predicting accuracy mean is: 56.2%
Fold predicting accuracy mean is: 60.0%
Fold predicting accuracy mean is: 46.7%
Total Mean accuracy is: 51.3%


Fold predicting accuracy mean is: 92.9%
Fold predicting accuracy mean is: 100.0%
Fold predicting accuracy mean is: 96.3%
Fold predicting accuracy mean is: 92.6%
Fold predicting accuracy mean is: 85.2%
Total Mean accuracy is: 93.4%


'''
