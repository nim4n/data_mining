
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold

df = pd.read_pickle('processed_data/dataframe_by_pre_process.pd')

feature_selected_by_classification = ['32166_at', '40567_at', '32598_at', '38269_at', '995_g_at',
                                      '39054_at', '34315_at', '37958_at', '1356_at', '1450_g_at',
                                      '40282_s_at', '39366_at', '41242_at', '41458_at']
labels = df["Status"].values
del df['Status']
df2 = df[feature_selected_by_classification]
#df = pd.read_pickle('processed_data/rank_classification_dataframe.pd')
features = df2[list(df2.columns)].values

classifier = svm.SVC()
classifier.fit(features, labels)
kf = KFold(len(features), n_folds=5, shuffle=True)

means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))