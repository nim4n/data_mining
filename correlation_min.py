import pandas as pd
import numpy as np


df = pd.read_pickle('processed_data/pre_process_dataframe.pd')

result = {'total_correlation': {}, 'status_correlation': {}}
feature_selected_by_classification = [u'1898_at', u'32166_at', u'32314_g_at', u'32780_at', u'33371_s_at',
                                      u'33891_at', u'34162_at', u'34407_at', u'34775_at', u'35177_at',
                                      u'35742_at', u'35803_at', u'36040_at', u'36159_s_at', u'36192_at',
                                      u'36627_at', u'36659_at', u'36792_at', u'37225_at', u'37230_at',
                                      u'37366_at', u'37630_at', u'37639_at', u'37716_at',  u'33412_at',
                                      u'37958_at', u'38028_at', u'38047_at', u'38396_at', u'38717_at',
                                      u'38772_at', u'39099_at', u'39243_s_at', u'39366_at', u'39714_at',
                                      u'39940_at', u'40069_at', u'40071_at', u'40113_at', u'40567_at',
                                      u'40841_at', u'41388_at', u'41468_at', u'575_s_at']
status = df['Status']
del df['Status']
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
corr_df['status_correlation'] = corr_df['status_correlation'][marker1] * -1
corr_df['total_correlation'] = corr_df['total_correlation'][marker2] * -1
corr_df['result'] = corr_df['status_correlation'] - corr_df['total_correlation']
corr_df = corr_df.sort(['result'], ascending=[0])

corr_df.to_pickle('processed_data/corr_little_dataframe.pd')

