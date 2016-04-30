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
feature_selected_by_classification = [u'120_at', u'33222_at', u'36159_s_at', u'36192_at', u'36659_at', u'37230_at', u'40567_at',
                                      u'40841_at', '32598_at', '37572_at', '39054_at', '39184_at', '41728_at', '39711_at',
                                      '33215_g_at', '32223_at', '36814_at', '39147_g_at', '34315_at', '32575_at', '39168_at']
feature_selected_by_classification = [u'40567_at', '32598_at', '39711_at', '38057_at',
                                      '1531_at', '38482_at', 'AFFX-PheX-5_at', '38651_at',
                                      '37639_at', '41271_at', '39608_at', '36950_at',
                                      '38764_at', '33442_at', '38400_at', '37982_at',
                                      '1104_s_at', '32166_at', '41484_r_at', '39336_at',
                                      '40488_at']








labels = df["Status"].values
del df['Status']


def find_new_feature(accouracy):
    accouracy += 0.002
    new_features = {}
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
                if total_mean > accouracy:
                    new_features[col] = [1, total_mean]
            except:
                pass
            feature_selected_by_classification.remove(col)
    return new_features


def remove_feature(accouracy):
    accouracy -= 0.04
    remove_list = {}
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
            if total_mean >= accouracy:
                remove_list[col] = total_mean
        except Exception as e:
            pass
        feature_selected_by_classification.append(col)
    return remove_list


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
    return total_mean


def main():
    check = True
    best_total_accourate = 0
    while check:
        accouracy_means = []
        for _ in range(4):
            accouracy_means.append(last_result())
        current_accouracy = np.mean(accouracy_means)
        if current_accouracy > best_total_accourate:
            best_total_accourate = current_accouracy
        print current_accouracy
        new_features = {}
        for _ in range(4):
            result = find_new_feature(best_total_accourate)
            for col in result:
                if col in new_features:
                    count1, accourate1 = new_features[col]
                    count2, accourate2 = result[col]
                    count1 += 1
                    new_features[col] = [count1, (accourate1 + accourate2)]
                else:
                    new_features[col] = result[col]

        best_choice = None
        max_count = 1
        best_accourate = 0
        for col in new_features:
            count, accourate = new_features[col]
            accourate = accourate / count
            if count > max_count:
                best_choice = col
                max_count = count
                best_accourate = accourate
            elif count == max_count:
                if accourate > best_accourate:
                    best_accourate = accourate
                    best_choice = col
        print 'add this feature', best_choice, best_total_accourate, max_count
        remove_features_list = {}
        for _ in range(4):
            result = remove_feature(best_total_accourate)
            for col in result:
                if col in remove_features_list:
                    count, accourate1 = remove_features_list[col]
                    accourate2 = result[col]
                    count += 1
                    if accourate2 > accourate1:
                        remove_features_list[col] = [count, (accourate1 + accourate2)]
                else:
                    remove_features_list[col] = [1, result[col]]

        best_choice_to_remove = None
        max_count = 1
        best_accourate = 0
        for col in remove_features_list:
            count, accourate = remove_features_list[col]
            accourate = accourate / count
            if count > max_count:
                best_choice_to_remove = col
                max_count = count
                best_accourate = accourate
            elif count == max_count:
                if accourate > best_accourate:
                    best_accourate = accourate
                    best_choice_to_remove = col

        print 'remove this feature', best_choice_to_remove, best_total_accourate, max_count
        if best_choice:
            feature_selected_by_classification.append(best_choice)
        if best_choice_to_remove:
            feature_selected_by_classification.remove(best_choice_to_remove)
        if not best_choice and not best_choice_to_remove:
            check = False
            print feature_selected_by_classification
        print '________________________________________________________________'
        print feature_selected_by_classification
        print '________________________________________________________________'



main()