import pandas as pd
import numpy as np


df = pd.read_pickle('processed_data/pre_process_dataframe.pd')


feature_selected_by_classification = [u'32166_at',
                                      u'36192_at',
                                      u'37230_at',
                                      u'38717_at',
                                      u'40567_at',
                                      u'32598_at',
                                      u'40125_at',
                                      u'41728_at', u'1768_s_at',
                                      u'39711_at', u'38269_at', u'1706_at',
                                      '35749_at', '39054_at', '39147_g_at', '36814_at', '34792_at', '34315_at',
                                      '41764_at', '37958_at']

removed = ['37958_at', '40069_at', '32780_at', '33891_at', '34162_at', '39099_at', '34407_at', '37225_at', '38396_at', '40841_at',
           '36040_at', '36659_at', '33371_s_at', '39366_at', '36159_s_at', '39940_at', '37630_at', '40113_at', '41468_at', '37366_at',
           '38047_at', '1898_at', '37716_at', '40176_at', '1998_i_at', '37648_at', '2075_s_at', '34508_r_at', '575_s_at',
           '38028_at', '33920_at', '37042_at', '40452_at', '40219_at', '35803_at', '39435_at', '31816_at',
           '32314_g_at', '40071_at', '34775_at', '34884_at', '36611_at', '33215_g_at', '1664_at', '33412_at',
           '38772_at']
status = df['Status']
del df['Status']
def calculate_correlations():
    result = {'total_correlation': {}, 'status_correlation': {}}
    count = 1
    for col1 in df.columns:
        if col1 not in feature_selected_by_classification:
            sum_correlation = sum([df[col2].corr(df[col1]) for col2 in feature_selected_by_classification])
            result['total_correlation'][col1] = sum_correlation
            result['status_correlation'][col1] = status.corr(df[col1])
            count += 1
            print count


    corr_df = pd.DataFrame.from_dict(data=result)
    marker1 = corr_df['status_correlation'] < 0
    marker2 = corr_df['total_correlation'] < 0
    corr_df['status_correlation'][marker1] = corr_df['status_correlation'][marker1] * -1
    corr_df['total_correlation'][marker2] = corr_df['total_correlation'][marker2] * -1
    corr_df['result'] = corr_df['status_correlation'] - corr_df['total_correlation']
    corr_df = corr_df.sort(['result'], ascending=[0])
    print corr_df
    corr_df.to_pickle('processed_data/corr_little_dataframe.pd')



def calculate_internal_correlations():
    result = {'total_correlation': {}, 'status_correlation': {}}
    count = 1
    for col1 in feature_selected_by_classification:
        new_list = []
        for col2 in feature_selected_by_classification:
            if col1 != col2:

                new_list.append(df[col1].corr(df[col2]))
        sum_correlation = sum(new_list)
        print col1, sum_correlation
        result['total_correlation'][col1] = sum_correlation
        result['status_correlation'][col1] = status.corr(df[col1])
        count += 1
        print count

    corr_df = pd.DataFrame.from_dict(data=result)
    marker1 = corr_df['status_correlation'] < 0
    marker2 = corr_df['total_correlation'] < 0
    corr_df['status_correlation'][marker1] = corr_df['status_correlation'][marker1] * -1
    corr_df['total_correlation'][marker2] = corr_df['total_correlation'][marker2] * -1
    corr_df['result'] = corr_df['status_correlation'] - corr_df['total_correlation']
    corr_df = corr_df.sort(['result'], ascending=[0])
    print corr_df
    corr_df.to_pickle('processed_data/internal_corr_dataframe.pd')


calculate_internal_correlations()
#calculate_correlations()