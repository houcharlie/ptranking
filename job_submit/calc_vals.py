import pickle
import numpy as np
import os



# metric_list = ['test/robust-ndcg@5']
metric_list = ['test/ndcg@5', 'test/robust-ndcg@5']
# for metric in metric_list:
    
#     for method in ['SimCLR', 'SimSiam']:
#         # for augmentation in ['zeroes', 'gaussian']:
#         #     if augmentation == 'zeroes':
#         #         percents = [0.1, 0.7]
#         #     else:
#         #         percents = [1.0, 2.0]
#         #     for percent in percents:
#         if method == 'SimCLR':
#             augmentation = 'gaussian'
#             percent = 1.0
#         else:
#             augmentation = 'zeroes'
#             percent = 0.1
#         for dataset in ['mslr', 'set1', 'istella']:
#             for shrink in [0.001, 0.002, 0.005, 0.01, 0.1, 0.5, 1.0]:
#                 metrics_set = []
#                 for trial in range(3):
#                     try:
#                         if method == 'SimSiam':
#                             metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{0}/iclr/'
#                             '{1}_{2}{3}_0.0005_dim_64_layers_5_to_finetune_5e-05_trial{4}_shrink0.01/metrics_0_3_1_4_{5}.pickle').format(dataset, method, augmentation, percent, trial, shrink)
#                         elif method == 'SimCLR':
#                             metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{0}/iclr/'
#                                             'SimCLR_{1}{2}_0.0005_dim_64_layers_5_to_finetune_5e-05_temp0.01'
#                                             '_mix0.0_trial{3}_shrink0.01_blend1.0_scale0.0_gumbel0.01_numnegatives100/metrics_0_3_1_4_{4}.pickle').format(dataset, augmentation, percent, trial, shrink)
#                         with open(metrics_file, 'rb') as handle:
#                             b = pickle.load(handle)
#                         metrics_set.append(b[metric])
#                     except:
#                         print("missing", metrics_file)
#                 print(method, dataset, shrink, augmentation, percent, metric, round(np.mean(metrics_set), 4), round(np.std(metrics_set), 4))
print_list = []

for metric in metric_list:
    for method in ['Scratch', 'SimCLR', 'SimSiam']:
        
        if method in ['SimCLR', 'SimSiam']:
            for augmentation in ['zeroes', 'gaussian']:
                if augmentation == 'zeroes':
                    percents = [0.1, 0.7]
                elif augmentation == 'gaussian':
                    percents = [1.0, 2.0]
                for percent in percents:
                    curr_row = []
                    for dataset in ['mslr', 'set1', 'istella']:
                    # for dataset in ['set1']:
                        metrics_set = []
                        for trial in range(3):
                            if method == 'SimSiam':
                                metrics_file = (f'/ocean/projects/cis230033p/houc/ranking/{dataset}/iclr/SimSiam_{augmentation}{percent}_0.0005_dim_64_layers_5_to_finetune_5e-05_trial{trial}_shrink0.01/metrics_0_1_1_4.25_1.0.pickle').format(trial, dataset)
                            elif method == 'SimCLR':
                                # metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/SimCLR'
                                #                 '_gaussian0.7_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/metrics_0_5_0_0.pickle').format(trial, dataset)
                                metrics_file = f'/ocean/projects/cis230033p/houc/ranking/{dataset}/iclr/SimCLR_{augmentation}{percent}_0.0005_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.0_trial{trial}_shrink0.01_blend1.0_scale1.0_gumbel0.01_numnegatives100/metrics_0_1_1_4.25_1.0.pickle'
                            with open(metrics_file, 'rb') as handle:
                                b = pickle.load(handle)
                            try:
                                metrics_set.append(b[metric])
                            except:
                                print(b)
                        print(method, dataset, augmentation, percent, metric, np.mean(metrics_set), np.std(metrics_set))
                        curr_row.extend([np.mean(metrics_set), np.std(metrics_set)])
                    print_list.append([f"{method} {augmentation} {percent}"] +  curr_row)

        else:
            curr_row = []
            for dataset in ['mslr', 'set1', 'istella']:
            # for dataset in ['set1']:
                for shrink in [1.0]:
                    metrics_set = []
                    for trial in range(3):
                        metrics_file = f'/ocean/projects/cis230033p/houc/ranking/{dataset}/shift_sparse_t=4.0_tau=4.25/Scratch_0.001_layers5_trial{trial}_shrink1.0/metrics_0_0_0_0.0_1.0.pickle'
                        with open(metrics_file, 'rb') as handle:
                            b = pickle.load(handle)
                        metrics_set.append(b[metric])
                    print('Scratch', shrink, dataset, metric, np.mean(metrics_set), np.std(metrics_set))
                    curr_row.extend([np.mean(metrics_set), np.std(metrics_set)])
            
            print_list.append([f"No pretrain"] + curr_row)
print(print_list)
for currlist in print_list:
    typedcurrlist = []
    for i, entry in enumerate(currlist):
        if i == 0:
            typedcurrlist.append(entry)
        else:
            typedcurrlist.append(str(round(entry, 4)))
    print(','.join(typedcurrlist))

            

# for dataset in ['mslr', 'set1', 'istella']:
#     for metric in metric_list:
#         metrics_set = []
#         for trial in range(5):        
#             metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/Scratch_0.001_layers5_trial{0}_shrink0.001/checkpoint_metrics.pickle').format(trial, dataset)
#             with open(metrics_file, 'rb') as handle:
#                 b = pickle.load(handle)
#             metrics_set.append(b[metric])
#         print("non pretrained", dataset, metric, round(np.mean(metrics_set), 4), round(np.std(metrics_set), 4))
