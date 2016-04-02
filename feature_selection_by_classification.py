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
count = 0
for column in df.columns:
    try:
        features = df[column].values
        features = df[list([column])].values
        #print features
        means = []
        for training, testing in kf:
            classifier.fit(features[training], labels[training])
            prediction = classifier.predict(features[testing])
            mean = np.mean(prediction == labels[testing])
            means.append(mean)
        total_mean = np.mean(means)
        if total_mean > 0.50:
            count += 1
            print('Total Mean accuracy is: {:.1%}'.format(total_mean))
        else:
            del df[column]
    except Exception as e:
        del df[column]
        print e

df.to_pickle('processed_data/rank_classification_dataframe.pd')
print count


corr_df = df.corr()
removed_columns = []
exist_columns = []
for column in corr_df:
    try:
        corr_df.drop(column, inplace=True)
        max_column_name = corr_df[column].idxmax()
        if max_column_name in removed_columns:
            corr_df.drop(max_column_name, inplace=True)
            max_column_name = corr_df[column].idxmax()
        max_corr = corr_df[column].max()
        if max_corr > 0.90 and not column in exist_columns:
            del df[column]
            removed_columns.append(column)
            exist_columns.append(max_column_name)
    except Exception as e:
        pass
df['Status'] = status
for column in df.columns:
    if column != 'Status':
        corr = df.Status.corr(df[column])
        if corr > 0.15:
            print corr
        else:
            del df[column]
df.to_pickle('processed_data/rank_classification_dataframe_remove_relation.pd')
print df.shape