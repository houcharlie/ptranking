import json
def_setting = {
    "pretrain_lr": 0.01,
    "finetune_lr": 1e-5,
    "aug_percent": 0.5,
    "dim": 64,
    "layers": 5,
    "temp": 0.01,
    'pretrainer': 'RankNeg',
    'mix': 0.25,
    'shrink': 0.001,
    'blend': 1.0,
    'scale': 0.,
    'gumbel': 1.0,
    'num_negatives': 100,
    'freeze': 0,
    'probe_layers': 1,
    'finetune_only': 1,
    'finetune_trials': 2
}

def format_string(new_setting, trial, aug, dataset):
    json_dir = '/jet/home/houc/ptranking/job_submit/inputs/' + dataset
    return '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19}\n'.format(
        new_setting['pretrain_lr'], new_setting['finetune_lr'], trial, aug,
        new_setting['aug_percent'], new_setting['dim'], new_setting['layers'],
        new_setting['temp'], new_setting['pretrainer'], new_setting['mix'],
        new_setting['shrink'], new_setting['blend'], new_setting['scale'], new_setting['gumbel'], new_setting['num_negatives'], json_dir, new_setting['freeze'], new_setting['probe_layers'], new_setting['finetune_only'], new_setting['finetune_trials'])


with open('./runlist.txt', 'w+') as f:
    f.write('\n')
    # for pretrainer in ['SimCLR']:
    #     for dataset in ['MSLRWEB30K', 'Set1', 'Istella_S']:
    #         if pretrainer == 'SimCLR':
    #             new_setting = def_setting.copy()
    #             new_setting['pretrainer'] = pretrainer
    #             new_setting['aug_percent'] = 0.7
    #             new_setting['finetune_only'] = 1
    #             for trial in range(5):
    #                 for finetune_trials in range(1):
    #                     new_setting['finetune_trials'] = finetune_trials
    #                     f.write(format_string(new_setting, trial, 'gaussian', dataset))
    #         if pretrainer == 'SimSiam':
    #             new_setting = def_setting.copy()
    #             new_setting['pretrainer'] = pretrainer
    #             new_setting['aug_percent'] = 0.7
    #             new_setting['finetune_only'] = 1
    #             for finetune_trials in range(0):
    #                 new_setting['finetune_trials'] = finetune_trials
    #                 f.write(format_string(new_setting, 0, 'qg', dataset))

    for pretrainer in ['SimCLR']:
        for dataset in ['MSLRWEB30K', 'Set1', 'Istella_S']:
            for aug_percent in [0.5]:     
                if pretrainer == 'SimSiam':
                    augmentation = 'qg'
                else:
                    augmentation = 'gaussian'
                for trial in range(5):    
                    new_setting = def_setting.copy()
                    new_setting['pretrainer'] = pretrainer
                    new_setting['aug_percent'] = aug_percent
                    new_setting['finetune_only'] = 1
                    new_setting['probe_layers'] = 3
                    new_setting['pretrain_lr'] = 0.001
                    new_setting['finetune_lr'] = 5e-5
                    f.write(format_string(new_setting, trial, augmentation, dataset))


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



    
    
