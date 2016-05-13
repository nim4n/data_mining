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

filehandler = open("processed_data/feature_selected_by_classification", 'rb')
feature_selected_by_classification = pickle.load(filehandler)
filehandler.close()

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