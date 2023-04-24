import pickle
import numpy as np
import os


top_level_folders = {
    'MSLR': '/ocean/projects/iri180031p/houc/ranking/MSLR-WEB30K/ssl/simsiam_sweep/',
    'Yahoo1': '/ocean/projects/iri180031p/houc/ranking/yahoo/avgs/',
    'Yahoo2': '/ocean/projects/iri180031p/houc/ranking/yahoo/set2avg/',
    'Istella': '/ocean/projects/iri180031p/houc/ranking/istella/avg/'
}
subfolders = {
    'MSLR': ['SimSiam_zeroes0.7_0.0005_dim_64_layers_5_to_finetune_1e-05_trial{0}_shrink0.001/', 'Scratch_0.001_layers5_trial{0}_shrink0.001/', 
             'SimCLR_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
             'RankNeg_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
             'RankNeg_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.2_scale0.0_gumbel1.0_numnegatives100/',
             'RankNeg_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.5_scale0.0_gumbel1.0_numnegatives100/',
             'RankNeg_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.7_scale0.0_gumbel1.0_numnegatives100/',
             ],
    'Yahoo1': ['SimSiam_zeroes0.8_0.0005_dim_64_layers_5_to_finetune_1e-05_trial{0}_shrink0.001/', 'Scratch_0.001_layers5_trial{0}_shrink0.001/',
               'SimCLR_zeroes0.05_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.3_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.2_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.5_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.7_scale0.0_gumbel1.0_numnegatives100/',],
    'Yahoo2': ['SimSiam_zeroes0.9_0.0005_dim_64_layers_5_to_finetune_1e-05_trial{0}_shrink0.01/', 'Scratch_0.0001_layers5_trial{0}_shrink0.01/',
               'SimCLR_zeroes0.01_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.01_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.8_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.01_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.8_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.01_blend0.2_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.8_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.01_blend0.5_scale0.0_gumbel1.0_numnegatives100/',
               'RankNeg_zeroes0.8_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.01_blend0.7_scale0.0_gumbel1.0_numnegatives100/',],
    'Istella': ['SimSiam_zeroes0.3_0.0005_dim_64_layers_5_to_finetune_1e-05_trial{0}_shrink0.001/', 'Scratch_0.001_layers5_trial{0}_shrink0.001/',
                'SimCLR_zeroes0.1_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
                'RankNeg_zeroes0.95_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/',
                'RankNeg_zeroes0.95_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.2_scale0.0_gumbel1.0_numnegatives100/',
                'RankNeg_zeroes0.95_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.5_scale0.0_gumbel1.0_numnegatives100/',
                'RankNeg_zeroes0.95_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend0.7_scale0.0_gumbel1.0_numnegatives100/',]
}
length=10
metric_list = ['test/ndcg@5', 'test/robust-ndcg@5']
for dataset in ['MSLR', 'Yahoo1', 'Yahoo2', 'Istella']:
    for subfolder in subfolders[dataset]:
        for metric in metric_list:
            metrics_set = []
            for trial in range(length):
                
                metrics_file = top_level_folders[dataset] + subfolder.format(trial) + 'metrics.pickle'
                try:
                    with open(metrics_file, 'rb') as handle:
                        b = pickle.load(handle)
                    metrics_set.append(b[metric])
                except:
                    print('metrics file', metrics_file, 'does not exist')
            print(metric, metrics_file, 'mean', np.mean(metrics_set))
            print(metric, 'stderr', np.std(metrics_set))




