""" Functions to prepare data for Graphpad Prism export """
import standard_pipeline.behavior_import as behavior
import standard_pipeline.performance_check as performance
import multisession_analysis.batch_analysis as batch
import pandas as pd
#%% Endothelin behavior
path = [r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch6']
data = performance.load_performance_data(roots=path, norm_date='20210413', stroke=['M71', 'M72', 'M74', 'M76', 'M75', 'M78', 'M80'])

mice = ('M71', 'M72', 'M74', 'M75',
        'M76', 'M77', 'M79')
pre_range = [('20210225', '20210301'),('20210224', '20210228'),('20210223', '20210227'),('20210225', '20210301'),
             ('20210224', '20210228'),('20210225', '20210301'),('20210223', '20210227')]

norm_data = []
for i in range(len(mice)):
    norm_data.append(performance.normalize_performance(data.loc[data['mouse'] == mice[i]], session_range=pre_range[i]))
norm_data = pd.concat(norm_data)

batch.exp_to_prism_mouse_avg(df=norm_data, field='licking_binned_norm',
                             directory=r'W:\Neurophysiology-Storage1\Wahl\Hendrik\PhD\Data\Batch6\analysis')
