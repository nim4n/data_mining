import numpy as np
import pandas as pd
from sklearn import random_projection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold

#df = pd.read_pickle('processed_data/feature_by_shannon_dataframe.pd')
df = pd.read_pickle('processed_data/dataframe_by_pre_process.pd')
'''
feature_selected_by_classification = [u'120_at', u'1736_at', u'1898_at', u'32109_at', u'32133_at',
                                      u'32166_at', u'32239_at', u'32314_g_at', u'32535_at', u'32780_at',
                                      u'33222_at', u'33371_s_at', u'33405_at', u'33412_at', u'33850_at',
                                      u'33891_at', u'34162_at', u'34407_at', u'34775_at', u'35177_at',
                                      u'35742_at', u'35803_at', u'36040_at', u'36159_s_at', u'36192_at',
                                      u'36627_at', u'36659_at', u'36792_at', u'37225_at', u'37230_at',
                                      u'37366_at', u'37617_at', u'37630_at', u'37639_at', u'37716_at',
                                      u'37958_at', u'38028_at', u'38047_at', u'38396_at', u'38717_at',
                                      u'38772_at', u'39099_at', u'39243_s_at', u'39366_at', u'39714_at',
                                      u'39940_at', u'40069_at', u'40071_at', u'40113_at', u'40567_at',
                                      u'40841_at', u'41388_at', u'41468_at', u'575_s_at', 'Status']
'''

feature_selected_by_classification = ['32166_at', '40567_at', '32598_at', '38269_at', '995_g_at',
                                      '39054_at', '34315_at', '37958_at', '1356_at', '1450_g_at',
                                      '40282_s_at', '39366_at', '41242_at', '41458_at']

removed = ['1768_s_at', '41728_at', '37929_at', '39147_g_at', '36814_at', '38469_at', '41764_at', '39711_at', '37230_at', '39634_at', '36958_at', '36192_at']





labels = df["Status"].values
del df['Status']

def find_new_feature():
    for col in df.columns:
        if col not in feature_selected_by_classification:
            feature_selected_by_classification.append(col)
            try:
                df2 = df[feature_selected_by_classification]
                #df = pd.read_pickle('processed_data/rank_classification_dataframe.pd')
                features = df2[list(df2.columns)].values



                kf = KFold(len(features), n_folds=7, shuffle=True)
                classifier = KNeighborsClassifier(n_neighbors=1)
                classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

                means = []
                for training, testing in kf:
                    classifier.fit(features[training], labels[training])
                    prediction = classifier.predict(features[testing])
                    mean = np.mean(prediction == labels[testing])
                    means.append(mean)
                total_mean = np.mean(means)
                if total_mean > 0.9855:
                    print(total_mean, col)
            except:
                print 'error'
            feature_selected_by_classification.remove(col)


def remove_feature():
    feature_selected_by_classification2 = list(feature_selected_by_classification)
    for col in feature_selected_by_classification2:
        feature_selected_by_classification.remove(col)

        try:
            df2 = df[feature_selected_by_classification]
            #df = pd.read_pickle('processed_data/rank_classification_dataframe.pd')
            features = df2[list(df2.columns)].values



            kf = KFold(len(features), n_folds=7, shuffle=True)
            classifier = KNeighborsClassifier(n_neighbors=1)
            classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

            means = []
            for training, testing in kf:
                classifier.fit(features[training], labels[training])
                prediction = classifier.predict(features[testing])
                mean = np.mean(prediction == labels[testing])
                means.append(mean)
            total_mean = np.mean(means)
            if total_mean > 0.95:
                print(total_mean, col)
                print '_______________________'
            else:
                print total_mean, col
        except:
            print 'error'
        feature_selected_by_classification.append(col)



def last_result():
    df2 = df[feature_selected_by_classification]
    #df = pd.read_pickle('processed_data/rank_classification_dataframe.pd')
    features = df2[list(df2.columns)].values



    kf = KFold(len(features), n_folds=7, shuffle=True)
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

    means = []
    for training, testing in kf:
        classifier.fit(features[training], labels[training])
        prediction = classifier.predict(features[testing])
        mean = np.mean(prediction == labels[testing])
        means.append(mean)
    total_mean = np.mean(means)
    print total_mean

