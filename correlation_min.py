import pandas as pd
import numpy as np


df = pd.read_pickle('processed_data/pre_process_dataframe.pd')

'''
feature_selected_by_classification = [u'32166_at',
                                      u'36192_at',
                                      u'37230_at',
                                      u'40567_at',
                                      u'32598_at',
                                      u'41728_at', u'1768_s_at',
                                      u'39711_at', u'38269_at', u'1706_at',
                                      '35749_at', '39054_at', '39147_g_at', '36814_at', '34315_at',
                                      '41764_at', '37958_at', '38469_at']

[u'120_at', u'33222_at', u'36159_s_at', u'36192_at', u'36659_at', u'37230_at', u'40567_at',
 u'40841_at', '32598_at', '37572_at', '39054_at', '39184_at', '41728_at', '39711_at',
  '33215_g_at', '32223_at', '36814_at', '39147_g_at', '34315_at', '32575_at', '39168_at']

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
                                      u'40841_at', u'41388_at', u'41468_at', u'575_s_at']


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
                internal_df.to_pickle('processed_data/min_correlation_dataframe.pd')
                print feature_selected_by_classification
            except:
                print feature_selected_by_classification, new_col
            check = False
        else:
            print 'candidate is accepted, better result is: ', new_sum
        if count % 10 == 0:
            print count

main(0.3, 20)

