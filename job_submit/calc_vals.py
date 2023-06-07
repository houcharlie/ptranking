import pickle
import numpy as np
import os



metric_list = ['test/ndcg@5', 'test/robust-ndcg@5']
# metric_list = ['test/ndcg@5']

# for method in ['SimSiam', 'SimCLR']:
#     for augmentation in ['zeroes', 'qg', 'gaussian']:
#         for dataset in ['mslr', 'set1', 'istella']:
#             for metric in metric_list:
#                 percent = 0.7
#                 metrics_set = []
#                 for trial in range(5):
#                     if method == 'SimSiam':
#                         metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{0}/augmentations/'
#                         '{1}_{2}{3}_0.0005_dim_64_layers_5_to_finetune_1e-05_trial{4}_shrink0.001/metrics.pickle').format(dataset, method, augmentation, percent, trial)
#                     elif method == 'SimCLR':
#                         metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{0}/augmentations/'
#                                         'SimCLR_{1}{2}_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01'
#                                         '_mix0.25_trial{3}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/metrics.pickle').format(dataset, augmentation, percent, trial)
#                     with open(metrics_file, 'rb') as handle:
#                         b = pickle.load(handle)
#                     metrics_set.append(b[metric])
#                 print(method, dataset, augmentation, percent, metric, np.mean(metrics_set), np.std(metrics_set))
for method in ['SimCLR']:
    for dataset in ['mslr', 'set1', 'istella']:
        metrics_set = []
        for metric in metric_list:
            for trial in range(5):
                if method == 'SimSiam':
                    metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/'
                                'SimSiam_zeroes0.7_0.01_dim_64_layers_5_to_finetune_0.001_trial{0}_shrink0.001/metrics_0_1_1_2.pickle').format(trial, dataset)
                elif method == 'SimCLR':
                    # metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/SimCLR'
                    #                 '_gaussian0.7_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/metrics_0_5_0_0.pickle').format(trial, dataset)
                    metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/SimCLR'
                        '_gaussian0.5_0.001_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/metrics_0_3_1_2.pickle').format(trial, dataset)
                with open(metrics_file, 'rb') as handle:
                    b = pickle.load(handle)
                metrics_set.append(b[metric])
            print(method, dataset, metric, np.mean(metrics_set), np.std(metrics_set))
