import numpy as np
import pandas as pd
from sklearn import random_projection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold

#df = pd.read_pickle('processed_data/feature_by_shannon_dataframe.pd')
df = pd.read_pickle('processed_data/rank_classification_dataframe_remove_relation.pd')
feature_selected_by_classification = [u'120_at', u'1736_at', u'1898_at', u'288_s_at', u'31831_at',
       u'32109_at', u'32133_at', u'32239_at', u'32314_g_at', u'32535_at',
       u'32780_at', u'33222_at', u'33405_at', u'33412_at', u'33850_at',
       u'33891_at', u'34162_at', u'34407_at', u'34775_at', u'35177_at',
       u'35742_at', u'35803_at', u'36040_at', u'36159_s_at', u'36192_at',
       u'36627_at', u'36792_at', u'37225_at', u'37230_at', u'37366_at',
       u'37617_at', u'37630_at', u'37639_at', u'37716_at', u'37958_at',
       u'38028_at', u'38047_at', u'38396_at', u'38469_at', u'38700_at',
       u'38717_at', u'39099_at', u'39243_s_at', u'39366_at', u'39714_at',
       u'39790_at', u'39940_at', u'40069_at', u'40071_at', u'40113_at',
       u'40567_at', u'40841_at', u'41273_at', u'41388_at', u'41468_at',
       u'575_s_at', u'755_at', 'Status']
df = df[feature_selected_by_classification]

labels = df["Status"].values
del df['Status']
print df.columns
#df = pd.read_pickle('processed_data/rank_classification_dataframe.pd')
features = df[list(df.columns)].values
print features.shape

print features.shape


kf = KFold(len(features), n_folds=7, shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=1)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    mean = np.mean(prediction == labels[testing])
    print 'Fold predicting accuracy mean is: {:.1%}'.format(mean)
    means.append(mean)
print('Total Mean accuracy is: {:.1%}'.format(np.mean(means)))


