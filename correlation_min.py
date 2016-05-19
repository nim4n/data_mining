import pandas as pd
import numpy as np
import pickle

df = pd.read_pickle('processed_data/pre_process_dataframe_lung.pd')

pre_df = pd.read_pickle('processed_data/pre_feature_selection_dataframe.pd')
del pre_df['Status']
feature_selected_by_classification = list(pre_df.columns.values)
print feature_selected_by_classification

status = df['Status']
del df['Status']


def calculate_correlations(min_corr):
    result = {'total_correlation': {}, 'status_correlation': {}}
    for col1 in df.columns:
        if col1 not in feature_selected_by_classification:
            sum_correlation = sum([df[col2].corr(df[col1]) for col2 in feature_selected_by_classification])
            result['total_correlation'][col1] = sum_correlation
            result['status_correlation'][col1] = status.corr(df[col1])

    corr_df = pd.DataFrame.from_dict(data=result)
    marker1 = corr_df['status_correlation'] < 0
    marker2 = corr_df['total_correlation'] < 0
    corr_df['status_correlation'][marker1] = corr_df['status_correlation'][marker1] * -1
    corr_df['total_correlation'][marker2] = corr_df['total_correlation'][marker2] * -1
    corr_df['result'] = corr_df['status_correlation'] - corr_df['total_correlation']
    corr_df = corr_df.sort(['result'], ascending=[0])
    for col in corr_df.index.tolist():
        if corr_df.loc[col, 'status_correlation'] > min_corr:
            print 'new candidate is: ', col, corr_df.loc[col, 'status_correlation'], corr_df.loc[col, 'total_correlation']
            return col



def calculate_internal_correlations(min_corr):
    result = {'total_correlation': {}, 'status_correlation': {}}
    for col1 in feature_selected_by_classification:
        new_list = []
        for col2 in feature_selected_by_classification:
            if col1 != col2:

                new_list.append(df[col1].corr(df[col2]))
        sum_correlation = sum(new_list)
        status_corr = status.corr(df[col1])
        if status_corr < 0:
            status_corr = status_corr * -1
        if status_corr < min_corr:
            print '___________', col1, status_corr
            feature_selected_by_classification.remove(col1)
            continue
        result['total_correlation'][col1] = sum_correlation
        result['status_correlation'][col1] = status_corr

    corr_df = pd.DataFrame.from_dict(data=result)
    marker1 = corr_df['status_correlation'] < 0
    marker2 = corr_df['total_correlation'] < 0
    corr_df['status_correlation'][marker1] = corr_df['status_correlation'][marker1] * -1
    corr_df['total_correlation'][marker2] = corr_df['total_correlation'][marker2] * -1
    corr_df['result'] = corr_df['status_correlation'] - corr_df['total_correlation']
    corr_df = corr_df.sort(['result'], ascending=[1])
    print 'lowest condidate is: ', corr_df[:1]['result'].index[0], corr_df[:1]['status_correlation'].values[0], corr_df[:1]['total_correlation'].values[0]
    return corr_df, corr_df[:1]['result'].index[0], corr_df[:1]['result'].values[0], corr_df['result'].sum()


#calculate_internal_correlations(0.3)

def main(min_corr, number_of_feature):
    check = True
    count = 0
    while check == True:
        count += 1
        internal_df, last_col, last_col_result, current_sum = calculate_internal_correlations(min_corr)
        new_col = calculate_correlations(min_corr)
        if len(feature_selected_by_classification) > number_of_feature:
            feature_selected_by_classification.remove(last_col)
        print len(feature_selected_by_classification)
        feature_selected_by_classification.append(new_col)
        print len(feature_selected_by_classification)
        tarsh1, trash2, trash3, new_sum = calculate_internal_correlations(min_corr)
        if new_sum < current_sum and len(feature_selected_by_classification) > number_of_feature:
            try:
                feature_selected_by_classification.remove(new_col)
                feature_selected_by_classification.append(last_col)
                filehandler = open("processed_data/feature_selected_by_classification", "wb")
                pickle.dump(feature_selected_by_classification,filehandler)
                filehandler.close()
                internal_df.to_pickle('processed_data/min_correlation_dataframe.pd')
                print feature_selected_by_classification
            except:
                print feature_selected_by_classification, new_col
            check = False
        else:
            print 'candidate is accepted, better result is: ', new_sum
            filehandler = open("processed_data/feature_selected_by_classification", "wb")
            pickle.dump(feature_selected_by_classification,filehandler)
            filehandler.close()
        if count % 10 == 0:
            print count

main(0.3, 20)

