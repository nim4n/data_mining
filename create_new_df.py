import pandas as pd

new_df = pd.DataFrame()
marker_df = pd.read_pickle('processed_data/marker_dataframe.pd')
marker_df = marker_df * 60
df = pd.read_pickle('processed_data/shannon_nearest_entropy_dataframe.pd')

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
new_df['marker'] = pd.Series(marker_df.loc[feature_selected_by_classification]['marker'], index = feature_selected_by_classification)
new_df['total'] = pd.Series(df.loc[feature_selected_by_classification]['total_entropy'], index=feature_selected_by_classification)
new_df['result'] = new_df['marker'] - new_df['total']
new_df = new_df.sort(['result'], ascending=[0])


new_df.to_pickle('processed_data/final_entropy_df.pd')
