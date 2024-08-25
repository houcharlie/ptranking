import pickle
import numpy as np
import os



# metric_list = ['test/robust-ndcg@5']
metric_list = ['test/robust-ndcg@5']
# metric_list = ['test/ndcg@5']
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
#             for shrink in [0.002]:
#                 metrics_set = []
#                 for trial in range(3):
#                     try:
#                         if method == 'SimSiam':
#                             metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{0}/icml-rebuttal/'
#                             '{1}_{2}{3}_0.0005_dim_64_layers_5_to_finetune_5e-05_trial{4}_shrink0.01/metrics_0_3_1_4_{5}.pickle').format(dataset, method, augmentation, percent, trial, shrink)
#                         elif method == 'SimCLR':
#                             metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{0}/icml-rebuttal/'
#                                             'SimCLR_{1}{2}_0.0005_dim_64_layers_5_to_finetune_5e-05_temp0.01'
#                                             '_mix0.0_trial{3}_shrink0.01_blend1.0_scale0.0_gumbel0.01_numnegatives100/metrics_0_3_1_4_{4}.pickle').format(dataset, augmentation, percent, trial, shrink)
#                         with open(metrics_file, 'rb') as handle:
#                             b = pickle.load(handle)
#                         metrics_set.append(b[metric])
#                     except:
#                         print("missing", metrics_file)
#                 print(method, dataset, shrink, augmentation, percent, metric, round(np.mean(metrics_set), 4), round(np.std(metrics_set), 4))
# print_list = []

# for metric in metric_list:
#     # for method in ['Scratch', 'SimCLR', 'SimSiam']:
#     for method in ['SimCLR']: 
#     # for method in ['RankNeg']:
#         if method in ['SimCLR', 'SimSiam', 'RankNeg']:
#             # for augmentation in ['zeroes', 'gaussian', 'scarf', '']:
#             for augmentation in ['zeroes', 'gaussian']:
#             # for augmentation in ['vime']:
#                 if augmentation == 'zeroes':
#                     percents = [0.1, 0.7]
#                 elif augmentation == 'gaussian':
#                     percents = [1.0, 2.0]
#                 elif augmentation == 'scarf' or augmentation == 'dacl':
#                     percents = [0.3, 0.6, 0.9]
#                 elif augmentation == 'vime':
#                     percents = [0.3, 0.5, 0.7]
#                 for percent in percents:
#                     curr_row = []
#                     for dataset in ['mslr', 'set1', 'istella']:
#                     # for dataset in ['set1']:
#                         metrics_set = []
#                         for trial in range(3):
#                             if method == 'SimSiam':
#                                 metrics_file = (f'/ocean/projects/cis230033p/houc/ranking/{dataset}/iclr/SimSiam_{augmentation}{percent}_0.0005_dim_64_layers_5_to_finetune_5e-05_trial{trial}_shrink0.01/metrics_0_1_1_4.25_1.0.pickle').format(trial, dataset)
#                             elif method == 'SimCLR' or method == 'RankNeg':
#                                 # metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/SimCLR'
#                                 #                 '_gaussian0.7_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/metrics_0_5_0_0.pickle').format(trial, dataset)
#                                 metrics_file = f'/ocean/projects/cis230033p/houc/ranking/{dataset}/sub/{method}_{augmentation}{percent}_0.0005_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.0_trial{trial}_shrink0.01_blend1.0_scale1.0_gumbel0.01_numnegatives100/metrics_0_3_0_7.0_0.002.pickle'
#                             with open(metrics_file, 'rb') as handle:
#                                 b = pickle.load(handle)
#                             try:
#                                 metrics_set.append(b[metric])
#                             except:
#                                 print(b)
#                         print(method, dataset, augmentation, percent, metric, np.mean(metrics_set), np.std(metrics_set))
#                         curr_row.extend([np.mean(metrics_set), np.std(metrics_set)])
#                     print_list.append([f"{method} {augmentation} {percent}"] +  curr_row)

#         else:
#             curr_row = []
#             for dataset in ['mslr', 'set1', 'istella']:
#             # for dataset in ['set1']:
#                 for shrink in [1.0]:
#                     metrics_set = []
#                     for trial in range(3):
#                         metrics_file = f'/ocean/projects/cis230033p/houc/ranking/{dataset}/shift_sparse_t=4.0_tau=4.25/Scratch_0.001_layers5_trial{trial}_shrink1.0/metrics_0_0_0_0.0_1.0.pickle'
#                         with open(metrics_file, 'rb') as handle:
#                             b = pickle.load(handle)
#                         metrics_set.append(b[metric])
#                     print('Scratch', shrink, dataset, metric, np.mean(metrics_set), np.std(metrics_set))
#                     curr_row.extend([np.mean(metrics_set), np.std(metrics_set)])
            
#             print_list.append([f"No pretrain"] + curr_row)
# print(print_list)
# for currlist in print_list:
#     typedcurrlist = []
#     for i, entry in enumerate(currlist):
#         if i == 0:
#             typedcurrlist.append(entry)
#         else:
#             typedcurrlist.append(str(round(entry, 4)))
#     print(','.join(typedcurrlist))

            

# for dataset in ['mslr', 'set1', 'istella']:
#     for shrink in [0.001, 0.002, 0.005, 0.01, 0.1, 0.5, 1.0]:
#         for metric in metric_list:
#             # for dropout in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8]:
#             if dataset == 'mslr' or dataset == 'istella':
#                 dropout = 0.7
#             else:
#                 dropout = 0.0
#             metrics_set = []
#             for trial in range(3):        
#                 metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{1}/icml-rebuttal/SimCLR_gaussian1.0_0.0005_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.0_trial{0}_shrink0.01_blend1.0_scale1.0_gumbel0.01_numnegatives100/metrics_0_3_1_1.0_{2}.pickle').format(trial, dataset, shrink)
#                 # metrics_file = ('/ocean/projects/cis230033p/houc/ranking/{3}/icml-rebuttal/SimCLR_gaussian1.0_0.0005_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.0_trial{0}_shrink0.01_blend1.0_scale1.0_gumbel0.01_numnegatives100/metrics_0_1_1_11.0_{1}_{2}.pickle').format(trial, shrink, dropout, dataset)
#                 # try:
#                 with open(metrics_file, 'rb') as handle:
#                     b = pickle.load(handle)
                
#                 metrics_set.append(b[metric])
#                 # except:
#                 #     print('metrics_file not available', metrics_file)
#             print(round(np.mean(metrics_set), 4), round(np.std(metrics_set), 4))
#     print()
# for dataset in ['mslr', 'set1', 'istella']:
#     for shrink in [0.001]:
#         for metric in metric_list:
#             for aug_percent in [0.2, 0.7]:
#                 metrics_set = []
#                 for trial in range(3):        
#                     metrics_file = '/ocean/projects/cis230033p/houc/ranking/{2}/neurips/SubTab_zeroes{1}_5e-05_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.0_trial{0}_shrink0.01_blend1.0_scale1.0_gumbel0.01_numnegatives100/metrics_0_1_0_0.0_0.001_0.01.pickle'.format(trial, aug_percent, dataset)
#                     with open(metrics_file, 'rb') as handle:
#                         b = pickle.load(handle)
                    
#                     metrics_set.append(b[metric])
#                     # except:
#                     #     print('metrics_file not available', metrics_file)
#                 print(aug_percent, round(np.mean(metrics_set), 4), np.std(metrics_set))
#         print()
    
for dataset in ['mslr', 'set1', 'istella']:
    for shrink in [0.001]:
        for augment in ['gaussian']:
            for metric in metric_list:
                for aug_percent in [1.0]:
                    metrics_set = []
                    for trial in range(3):        
                        metrics_file = '/ocean/projects/cis230033p/houc/ranking/{2}/icml-rebuttal/SimCLR_{3}{1}_5e-05_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.0_trial{0}_shrink0.01_blend1.0_scale1.0_gumbel0.01_numnegatives100/metrics_0_1_0_0.0_0.001_0.01.pickle'.format(trial, aug_percent, dataset, augment)
                        if not os.path.exists(metrics_file):
                            continue
                        with open(metrics_file, 'rb') as handle:
                            b = pickle.load(handle)
                        
                        metrics_set.append(b[metric])
                        # except:
                        #     print('metrics_file not available', metrics_file)
                    print(augment, aug_percent, round(np.mean(metrics_set), 4), round(np.std(metrics_set), 4))
            print()


# for dataset in ['mslr', 'set1', 'istella']:
#     for shrink in [0.001, 0.002, 0.005, 0.01, 0.1, 0.5, 1.0]:
#         for pseudo in [0]:
#             best_mean = 0.
#             best_stddev = 0.
#             for pca in [0]:
#                 metrics_set = []
#                 robust_metrics_set = []
#                 for trial in range(3):
#                     metrics_file = '/ocean/projects/cis230033p/houc/ranking/{4}/tree/result_trial{0}_pcadim{1}_shrink{2}_pseudo{3}.pickle'.format(trial, pca, shrink, pseudo, dataset)
#                     try:
#                         with open(metrics_file, 'rb') as handle:
#                             b = pickle.load(handle)
#                         metrics_set.append(b['test/ndcg@5'])
#                         robust_metrics_set.append(b['test/robust-ndcg@5'])
#                     except:
#                         print('metrics_file not available', metrics_file)
#                 mean = round(np.mean(metrics_set), 4)
#                 stddev = round(np.std(metrics_set), 4)
#                 if mean > best_mean:
#                     best_mean = mean
#                     best_stddev = stddev
#                     best_robust_mean = np.mean(robust_metrics_set)
#                     best_robust_stddev = np.std(robust_metrics_set)

#             print(f"{best_robust_mean},{best_robust_stddev}")
#     print('')

# num = 1
# mslr = []
# yahoo = []
# istella = []
# for i in range(3):
#     for s in range(3):
#         for lr in range(2):
#             num_list = []
#             for trial in range(3):
#                 path = '/jet/home/houc/ptranking/job_submit/old_old_runs/deepfm_runs/runlist_240322_02.24.37/output_{0}.out'.format(num)
#                 with open(path, 'r') as f:
#                     for current_line_number, line in enumerate(f):
#                         if (current_line_number == 435 and i == 0) or (current_line_number == 434 and i == 1) or (current_line_number == 434 and i == 2):
                
#                             split_on_ndcg5 = line.split('nDCG@5:')[1]
#                             ndcg5 = float(split_on_ndcg5.split(",")[0])
#                             num_list.append(ndcg5)
#                 num += 1
#             # print(round(np.mean(num_list), 4), round(np.std(num_list), 4))
#             currmean = round(np.mean(num_list), 4)
#             if i == 0 and lr == 1:
#                 mslr.append(currmean)
#             elif i == 1 and lr == 1:
#                 yahoo.append(currmean)
#             elif i == 2 and lr == 1:
#                 istella.append(currmean)
    
# all_scores = [mslr, yahoo, istella]
# for i in range(3):
#     for j in range(len(mslr)-1, -1, -1):
#         print(all_scores[i][j])
#     print()



