import json
def_setting = {
    "pretrain_lr": 0.01,
    "finetune_lr": 1e-5,
    "aug_percent": 0.5,
    "dim": 64,
    "layers": 5,
    "temp": 0.01,
    'pretrainer': 'RankNeg',
    'mix': 1.0,
    'shrink': 0.001,
    'blend': 1.0,
    'scale': 1.0,
    'gumbel': 1e-2,
    'num_negatives': 100,
    'freeze': 0,
    'probe_layers': 0,
    'finetune_only': 0,
    'finetune_trials': 0
}

def format_string(new_setting, trial, aug, dataset):
    json_dir = '/jet/home/houc/ptranking/job_submit/inputs/' + dataset
    return '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}\n'.format(
        new_setting['pretrain_lr'], new_setting['finetune_lr'], trial, aug,
        new_setting['aug_percent'], new_setting['dim'], new_setting['layers'],
        new_setting['temp'], new_setting['pretrainer'], new_setting['mix'],
        new_setting['shrink'], new_setting['blend'], new_setting['scale'], new_setting['gumbel'], new_setting['num_negatives'], json_dir, new_setting['freeze'], new_setting['probe_layers'], new_setting['finetune_only'], new_setting['finetune_trials'])

with open('./run_checkpoint.txt', 'w+') as f:
    f.write('\n')
    # for pretrainer in ['SimSiam', 'SimCLR']:
    #     if pretrainer == 'SimSiam':
    #         stem = '/ocean/projects/cis230033p/houc/ranking/{dataset}/augmentations/{pretrainer}_{augtype}{aug_percent}_0.0005_dim_64_layers_5_to_finetune_1e-05_trial{trial}_shrink0.001/'
    #     else:
    #         stem = '/ocean/projects/cis230033p/houc/ranking/{dataset}/augmentations/{pretrainer}_{augtype}{aug_percent}_0.0005_dim_64_layers_5_to_finetune_1e-05_temp0.01_mix0.25_trial{trial}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/'
    #     for dataset in ['mslr', 'set1', 'istella']:
    #         for aug_type in ['zeroes', 'qg', 'gaussian']:
    #             for aug_percent in [0.2, 0.7]:
    #                 for trial in range(5):
    #                     if aug_type == 'gaussian' and aug_percent == 0.2:
    #                         continue
    #                     ckpt_path = stem.format(dataset=dataset, pretrainer=pretrainer, augtype=aug_type, aug_percent=aug_percent, trial=trial)
    #                     if dataset == 'mslr':
    #                         app = 'MSLRWEB30K'
    #                     elif dataset == 'set1':
    #                         app = 'Set1'
    #                     elif dataset == 'istella':
    #                         app = 'Istella_S'
    #  
    #                    f.write('{0} {1}'.format('/jet/home/houc/ptranking/job_submit/inputs/' + app, ckpt_path + '\n'))
    # for trial in range(5):
    #     for dataset in ['mslr', 'set1', 'istella']:
    #         ckpt_path = '/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/Scratch_0.001_layers5_trial{0}_shrink0.001/'.format(trial, dataset)
    #         if dataset == 'mslr':
    #             app = 'MSLRWEB30K'
    #         elif dataset == 'set1':
    #             app = 'Set1'
    #         elif dataset == 'istella':
    #             app = 'Istella_S'
    #         f.write('{0} {1}'.format('/jet/home/houc/ptranking/job_submit/inputs/' + app, ckpt_path + '\n'))

    for trial in range(5):
        for dataset in ['istella']:
            ckpt_path = '/ocean/projects/cis230033p/houc/ranking/{1}/augmentations/SimCLR_gaussian0.5_0.001_dim_64_layers_5_to_finetune_5e-05_temp0.01_mix0.25_trial{0}_shrink0.001_blend1.0_scale0.0_gumbel1.0_numnegatives100/'.format(trial, dataset)
            if dataset == 'mslr':
                app = 'MSLRWEB30K'
            elif dataset == 'set1':
                app = 'Set1'
            elif dataset == 'istella':
                app = 'Istella_S'
            f.write('{0} {1}'.format('/jet/home/houc/ptranking/job_submit/inputs/' + app, ckpt_path + '\n'))

with open('./runlist.txt', 'w+') as f:
    f.write('\n')
    for pretrainer in ['scratch']:
        for dataset in ['MSLRWEB30K', 'Set1', 'Istella_S']:
        # for dataset in ['MSLRWEB30K', 'Istella_S']:
        # for dataset in ['Set1']:
            for shrink in [1.0]:
                new_setting = def_setting.copy()
                new_setting['pretrainer'] = pretrainer
                new_setting["finetune_lr"] = 1e-3
                new_setting['shrink'] = shrink
                for trial in range(3):
                    f.write(format_string(new_setting, trial, 'none', dataset))

    # for pretrainer in ['SimCLR', 'SimSiam']:
    #     for dataset in ['MSLRWEB30K', 'Set1', 'Istella_S']:
    #         # for aug_percent in [0.1, 0.2]:
    #         #     for trial in range(5):
    #         #         new_setting = def_setting.copy()
    #         #         new_setting['pretrainer'] = pretrainer
    #         #         new_setting['aug_percent'] = aug_percent
    #         #         new_setting['finetune_only'] = 0
    #         #         new_setting['probe_layers'] = 1
    #         #         new_setting['pretrain_lr'] = 0.0005
    #         #         new_setting['finetune_lr'] = 5e-5
    #         #         new_setting['gumbel'] = 1e-4
    #         #         new_setting['shrink'] = 0.001
    #         #         f.write(format_string(new_setting, trial, 'zeroes', dataset))
    #         if pretrainer == 'SimCLR':
    #             aug_percent = 1.0
    #             augmentation = 'gaussian'
    #         if pretrainer == 'SimSiam':
    #             aug_percent = 0.1
    #             augmentation = 'zeroes'
    #         for shrink in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    #             for trial in range(3):
    #                 new_setting = def_setting.copy()
    #                 new_setting['pretrainer'] = pretrainer
    #                 new_setting['aug_percent'] = aug_percent
    #                 new_setting['probe_layers'] = 3
    #                 new_setting['finetune_only'] = 1
    #                 new_setting['finetune_trials'] = 40
    #                 new_setting['pretrain_lr'] = 0.0005
    #                 new_setting['finetune_lr'] = 5e-5
    #                 new_setting['shrink'] = shrink
    #                 f.write(format_string(new_setting, trial, augmentation, dataset))
    for pretrainer in ['SimCLR', 'SimSiam']:
        for dataset in ['MSLRWEB30K', 'Set1', 'Istella_S']:
        # for dataset in ['MSLRWEB30K', 'Istella_S']:
        # for dataset in ['Set1']:
            for augmentation in ['gaussian', 'zeroes']:
                shrink = 1.0
                if augmentation == 'gaussian':
                    aug_params = [1.0, 2.0]
                if augmentation == 'zeroes':
                    aug_params = [0.1, 0.7]
                for aug_percent in aug_params:
                    for trial in range(3):
                        new_setting = def_setting.copy()
                        new_setting['pretrainer'] = pretrainer
                        new_setting['aug_percent'] = aug_percent
                        new_setting['probe_layers'] = 1
                        new_setting['finetune_only'] = 1
                        new_setting['finetune_trials'] = 4.25
                        new_setting['pretrain_lr'] = 0.0005
                        new_setting['finetune_lr'] = 5e-5
                        new_setting['shrink'] = shrink
                        f.write(format_string(new_setting, trial, augmentation, dataset))
    # for pretrainer in ['SimCLR']:
    #     for dataset in ['MSLRWEB30K', 'Set1', 'Istella_S']:
    #         for aug_percent in [0.7]:     
    #             if pretrainer == 'SimSiam':
    #                 augmentation = 'qg'
    #             else:
    #                 augmentation = 'gaussian'
    #             for trial in range(5):    
    #                 new_setting = def_setting.copy()
    #                 new_setting['pretrainer'] = pretrainer
    #                 new_setting['aug_percent'] = aug_percent
    #                 new_setting['finetune_only'] = 1
    #                 new_setting['probe_layers'] = 3
    #                 new_setting['pretrain_lr'] = 5e-4
    #                 new_setting['finetune_lr'] = 1e-5
    #                 f.write(format_string(new_setting, trial, augmentation, dataset))

    # for pretrainer in ['SimCLR']:
    #     for dataset in ['MSLRWEB30K', 'Set1', 'Istella_S']:
    #         if pretrainer == 'SimSiam':
    #             augmentation = 'qg'
    #         else:
    #             augmentation = 'gaussian'
    #         for trial in range(5):           
    #             new_setting = def_setting.copy()
    #             new_setting['pretrainer'] = pretrainer
    #             new_setting['aug_percent'] = 0.7
    #             new_setting['finetune_only'] = 1
    #             f.write(format_string(new_setting, trial, augmentation, dataset))



    
    
