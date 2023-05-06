# with open('./runlist.txt', 'w+') as f:
#     for finetune_lr in [0.001, 0.0005]:
#         for aug_percent in [0.8, 0.9]:
#             for aug_type in ['qg', 'zeroes', 'none']:
#                 for trial in range(3):
#                     f.write('{0} {1} {2} {3}\n'.format(finetune_lr, aug_percent, aug_type, trial))

# default_setting = {
#     "pretrain_lr": 1e-4,
#     "finetune_lr": 1e-4,
#     "aug_percent": 0.8,
#     "dim": 64,
#     "layers": 5,
#     "temp": 0.01,
#     'pretrainer': 'RankNeg',
#     'mix': 0.5,
#     'shrink': 0.0025,
#     'blend': 0.5,
#     'scale': 0.01,
#     'gumbel': 0.1
# }
def_setting = {
    "pretrain_lr": 5e-4,
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
    'num_negatives': 100
}
# def_setting = {
#     "pretrain_lr": 5e-4,
#     "finetune_lr": 1e-5,
#     "aug_percent": 0.5,
#     "dim": 64,
#     "layers": 5,
#     "temp": 0.01,
#     'pretrainer': 'RankNeg',
#     'mix': 0.25,
#     'shrink': 0.01,
#     'blend': 1.0,
#     'scale': 0.,
#     'gumbel': 1.0,
#     'num_negatives': 100
# }

def format_string(new_setting, trial, aug):
    return '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}\n'.format(
        new_setting['pretrain_lr'], new_setting['finetune_lr'], trial, aug,
        new_setting['aug_percent'], new_setting['dim'], new_setting['layers'],
        new_setting['temp'], new_setting['pretrainer'], new_setting['mix'],
        new_setting['shrink'], new_setting['blend'], new_setting['scale'], new_setting['gumbel'], new_setting['num_negatives'])


with open('./runlist.txt', 'w+') as f:
    # simsiam_setting = {
    #     "pretrain_lr": 1e-4,
    #     "finetune_lr": 1e-4,
    #     "aug_percent": 0.9,
    #     "dim": 64,
    #     "layers": 5,
    #     "temp": 0.01,
    #     'pretrainer': 'SimSiam',
    #     'mix': 0.25,
    #     'shrink': 0.0025,
    #     'blend': 0.5,
    #     'scale': 0.5,
    #     'gumbel': 1.
    # }
    # f.write(format_string(simsiam_setting, 0, 'zeroes'))
    
    # for trial in range(5):
    #     new_setting = simsiam_setting.copy()
    #     f.write(format_string(new_setting, trial, 'zeroes'))
    #     new_setting['pretrainer'] = 'RankNeg'
    #     f.write(format_string(new_setting, trial, 'zeroes'))
    # f.write('\n')
    # for aug_percent in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    #     new_setting = def_setting.copy()
    #     new_setting['pretrainer'] = 'SimSiam'
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    
    # for aug_percent in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7]:
    #     new_setting = def_setting.copy()
    #     new_setting['pretrainer'] = 'SimCLR'
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'zeroes'))

    # for aug_percent in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    #     new_setting = def_setting.copy()
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    
    # for finetune_lr in [1e-3, 5e-4, 1e-4, 5e-5]:
    #     new_setting = def_setting.copy()
    #     new_setting['finetune_lr'] = finetune_lr
    #     f.write(format_string(new_setting, 0, 'none'))

    # for aug_percent in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    #     new_setting = def_setting.copy()
    #     new_setting['pretrainer'] = 'SimSiam'
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting,1, 'zeroes'))
    
    # for aug_percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     new_setting = def_setting.copy()
    #     new_setting['pretrainer'] = 'RankNeg'
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 1, 'qg'))
    
    for trial in range(5):
        new_setting = def_setting.copy()
        new_setting['aug_percent'] = 0.7
        new_setting['pretrainer'] = 'RankNeg'
        f.write(format_string(new_setting, trial, 'qg'))

    # for trial in range(10):
    #     new_setting = def_setting.copy()
    #     new_setting['pretrainer'] = 'SimCLR'
    #     new_setting['aug_percent'] = 0.1
    #     f.write(format_string(new_setting, trial, 'zeroes'))

    # for trial in range(10):
    #     new_setting = def_setting.copy()
    #     new_setting['aug_percent'] = 0.95
    #     f.write(format_string(new_setting, trial, 'zeroes'))
    
    # for trial in range(10):
    #     new_setting = def_setting.copy()
    #     new_setting['finetune_lr'] = 1e-3
    #     f.write(format_string(new_setting, trial, 'none'))

    # for trial in range(10):
    #     new_setting = def_setting.copy()
    #     new_setting['aug_percent'] = 0.95
    #     new_setting['blend'] = 0.2
    #     f.write(format_string(new_setting, trial, 'zeroes'))
    
    # for trial in range(10):
    #     new_setting = def_setting.copy()
    #     new_setting['aug_percent'] = 0.95
    #     new_setting['blend'] = 0.5
    #     f.write(format_string(new_setting, trial, 'zeroes'))
    
    # for trial in range(10):
    #     new_setting = def_setting.copy()
    #     new_setting['aug_percent'] = 0.95
    #     new_setting['blend'] = 0.7
    #     f.write(format_string(new_setting, trial, 'zeroes'))
    
    # for trial in range(10):
    #     new_setting = def_setting.copy()
    #     new_setting['pretrain_lr'] = 5e-4
    #     new_setting['finetune_lr'] = 1e-5
    #     new_setting['aug_percent'] = 0.1
    #     new_setting['pretrainer'] = 'SimCLR'
    #     f.write(format_string(new_setting, trial, 'zeroes'))

    

    # ndcg_setting = def_setting.copy()
    # ndcg_setting['aug_percent'] = 0.1
    # ndcg_setting['gumbel'] = 0.1
    # ndcg_setting['scale'] = 0.5
    # for trial in range(20):
    #     f.write(format_string(ndcg_setting, trial, 'zeroes'))
    # ndcg_setting = def_setting.copy()
    # ndcg_setting['aug_percent'] = 0.5
    # ndcg_setting['gumbel'] = 1.0
    # ndcg_setting['scale'] = 0.5
    # for trial in range(20):
    #     f.write(format_string(ndcg_setting, trial, 'zeroes'))

    # for aug_percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     for scale in [1.0, 0.5, 0.1, 0.]:
    #         new_setting = def_setting.copy()
    #         new_setting['aug_percent'] = aug_percent
    #         new_setting['scale'] = scale
    #         new_setting['pretrainer'] = 'SimSiam'
    #         f.write(format_string(new_setting, 0, 'zeroes'))

    # f.write('\n')
    # settings = [(0.5, 0.5, 200)]
    # for setting in settings:
    #     for trial in range(6,11):
    #         new_setting = def_setting.copy()
    #         new_setting['aug_percent'] = setting[0]
    #         new_setting['num_negatives'] = setting[2]
    #         new_setting['gumbel'] = setting[1]
    #         f.write(format_string(new_setting, trial, 'zeroes'))
    # for gumbel in [1.0, 0.5, 0.1, 0.05, 0.01]:
    #     for aug_percent in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #         new_setting = def_setting.copy()
    #         new_setting['aug_percent'] = aug_percent
    #         new_setting['gumbel'] = gumbel
    #         # f.write(format_string(new_setting, 1, 'zeroes'))
    #         f.write(format_string(new_setting, 2, 'zeroes'))
    

    # for aug_percent in [0.8, 0.9, 0.95]:
    #     new_setting = def_setting.copy()
    #     new_setting['aug_percent'] = aug_percent
    #     new_setting['pretrainer'] = 'SimSiam'
    #     # f.write(format_string(new_setting, 1, 'zeroes'))
    #     f.write(format_string(new_setting, 2, 'zeroes'))

    # for gumbel in [1., 0.1, 0.01, 0.001]:
    #     new_setting = simsiam_setting.copy()
    #     new_setting['pretrainer'] = 'RankNeg'
    #     new_setting['gumbel'] = gumbel
    #     f.write(format_string(new_setting, 2, 'zeroes'))
    #     new_setting['finetune_lr'] = 1e-5
    #     f.write(format_string(new_setting, 2, 'zeroes'))

    # simclr_setting = {
    #     "pretrain_lr": 1e-3,
    #     "finetune_lr": 1e-4,
    #     "aug_percent": 0.85,
    #     "dim": 64,
    #     "layers": 5,
    #     "temp": 0.1,
    #     'pretrainer': 'SimCLR',
    #     'mix': 0.25,
    #     'shrink': 0.0025,
    #     'blend': 0.5,
    #     'scale': 0.5,
    #     'gumbel': 1.
    # }
    # f.write(format_string(simclr_setting, 0, 'qg'))

    # simsiam_setting = {
    #     "pretrain_lr": 1e-4,
    #     "finetune_lr": 1e-4,
    #     "aug_percent": 0.7,
    #     "dim": 64,
    #     "layers": 5,
    #     "temp": 0.01,
    #     'pretrainer': 'SimSiam',
    #     'mix': 0.25,
    #     'shrink': 0.0025,
    #     'blend': 0.5,
    #     'scale': 0.5,
    #     'gumbel': 1.
    # }
    # f.write(format_string(simsiam_setting, 0, 'qg'))

    # simclr_setting = {
    #     "pretrain_lr": 1e-3,
    #     "finetune_lr": 1e-4,
    #     "aug_percent": 0.7,
    #     "dim": 64,
    #     "layers": 5,
    #     "temp": 0.1,
    #     'pretrainer': 'SimCLR',
    #     'mix': 0.25,
    #     'shrink': 0.0025,
    #     'blend': 0.5,
    #     'scale': 0.5,
    #     'gumbel': 1.
    # }
    # f.write(format_string(simclr_setting, 0, 'zeroes'))


    # scratch_setting = {
    #     "pretrain_lr": 1e-4,
    #     "finetune_lr": 1e-4,
    #     "aug_percent": 0.7,
    #     "dim": 64,
    #     "layers": 5,
    #     "temp": 0.1,
    #     'pretrainer': 'SimSiam',
    #     'mix': 0.25,
    #     'shrink': 0.0025,
    #     'blend': 0.5,
    #     'scale': 0.5,
    #     'gumbel': 1.
    # }
    # f.write(format_string(scratch_setting, 0, 'none'))
    
    # f.write(format_string(default_setting, 0, 'qz'))

    # for blend in [0., 0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
    #     new_setting = default_setting.copy()
    #     new_setting['blend'] = blend
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for aug_percent in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    #     new_setting = default_setting.copy()
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    
    # for gumbel in [1.0, 0.1, 0.01, 0.001, 0.0005]:
    #     new_setting = default_setting.copy()
    #     new_setting['gumbel'] = gumbel
    #     f.write(format_string(new_setting, 0, 'qz'))
    # for pretrain_lr in [1e-4, 1e-5]:
    #     for finetune_lr in [1e-4, 8e-5, 5e-5, 3e-5, 2e-5, 1e-5]:
    #         new_setting = default_setting.copy()
    #         new_setting['finetune_lr'] = finetune_lr
    #         new_setting['pretrain_lr'] = pretrain_lr
    #         f.write(format_string(new_setting, 0, 'zeroes'))
    
    # for scale in [0.01, 0.1, 1.]:
    #     new_setting = default_setting.copy()
    #     new_setting['scale'] = scale
    #     f.write(format_string(new_setting, 0, 'qz'))
        
    # for mix in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    #     new_setting = default_setting.copy()
    #     new_setting['mix'] = mix
    #     f.write(format_string(new_setting, 0, 'qz'))
    
    


    

    
    # for scale in [0.001, 0.005, 0.01, 1., 10., 20.]:
    #     new_setting = default_setting.copy()
    #     new_setting['scale'] = scale
    #     f.write(format_string(new_setting, 0, 'qz'))
    
    # for mix in [0., 0.25, 0.5, 0.75, 1.0]:
    #     new_setting = default_setting.copy()
    #     new_setting['mix'] = mix
    #     f.write(format_string(new_setting, 0, 'qz'))





    




    # new_setting = default_setting.copy()
    # f.write(format_string(new_setting, 0, 'qz'))
    # for finetune_lr in [1e-3, 1e-4, 1e-5]:
    #     for shrink in [0.01, 1.0]:
    #         new_setting = default_setting.copy()
    #         new_setting['finetune_lr'] = finetune_lr
    #         new_setting['shrink'] = shrink
    #         for trial in range(1):
    #             f.write(format_string(new_setting, trial, 'none'))

    # for blend in [0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.8, 0.9]:
    #     new_setting = default_setting.copy()
    #     new_setting['blend'] = blend
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for mix in [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1.]:
    #     new_setting = default_setting.copy()
    #     new_setting['mix'] = mix
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for scale in [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.]:
    #     new_setting = default_setting.copy()
    #     new_setting['scale'] = scale
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for mix in [0., 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    #     new_setting = default_setting.copy()
    #     new_setting['mix'] = mix
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for aug_percent in [0.5, 0.8, 0.9]:
    #     new_setting = default_setting.copy()
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for pretrain_lr, finetune_lr in [(1e-4, 1e-4), (1e-3, 5e-4), (1e-5, 1e-5)]:
    #     new_setting = default_setting.copy()
    #     new_setting['finetune_lr'] = finetune_lr
    #     new_setting['pretrain_lr'] = pretrain_lr
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for dim in [8, 32, 64, 100]:
    #     new_setting = default_setting.copy()
    #     new_setting['dim'] = dim
    #     f.write(format_string(new_setting, 0, 'qz'))

    ## simrank
    # pretrainer = 'SimSiamRank'
    # for aug_percent in [0.3, 0.5, 0.7, 0.75, 0.85, 0.9, 0.95, 0.99]:
    #     new_setting = default_setting.copy()
    #     new_setting['pretrainer'] = pretrainer
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'qz'))
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    #     f.write(format_string(new_setting, 0, 'qg'))

    # # simsiam
    # pretrainer = 'SimSiamRank'
    # for pretrain_lr, finetune_lr in [(1e-5, 1e-4), (1e-5, 1e-5), (1e-5, 1e-3), (1e-5, 5e-6)]:
    #     new_setting = default_setting.copy()
    #     new_setting['pretrainer'] = pretrainer
    #     new_setting['finetune_lr'] = finetune_lr
    #     new_setting['pretrain_lr'] = pretrain_lr
    #     f.write(format_string(new_setting, 0, 'qg'))
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    #     f.write(format_string(new_setting, 0, 'qz'))

    # pretrainer = 'SimSiamRank'
    # for dim in [8, 32, 64, 100]:
    #     new_setting = default_setting.copy()
    #     new_setting['pretrainer'] = pretrainer
    #     new_setting['dim'] = dim
    #     f.write(format_string(new_setting, 0, 'qg'))
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    #     f.write(format_string(new_setting, 0, 'qz'))

    # pretrainer = 'SimCLR'
    # for pretrain_lr, finetune_lr in [(1e-3, 1e-4), (1e-4, 1e-5), (1e-5, 1e-6)]:
    #     new_setting = default_setting.copy()
    #     new_setting['pretrainer'] = pretrainer
    #     new_setting['finetune_lr'] = finetune_lr
    #     new_setting['pretrain_lr'] = pretrain_lr
    #     f.write(format_string(new_setting, 0, 'qg'))
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for aug_percent in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #     new_setting = default_setting.copy()
    #     new_setting['pretrainer'] = pretrainer
    #     new_setting['aug_percent'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'qg'))
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for temp in [0.01, 0.1, 1., 10]:
    #     new_setting = default_setting.copy()
    #     new_setting['pretrainer'] = pretrainer
    #     new_setting['temp'] = aug_percent
    #     f.write(format_string(new_setting, 0, 'qg'))
    #     f.write(format_string(new_setting, 0, 'zeroes'))
    #     f.write(format_string(new_setting, 0, 'qz'))

    # for pretrainer in ['SimRank', 'SimSiam']:
    #     # explore lrs

    #     # if pretrainer =='SimRank':
    #     #     for mix in [0.0, 0.25, 0.5, 0.75, 1.0]:
    #     #         new_setting = default_setting.copy()
    #     #         new_setting['mix'] = mix
    #     #         for trial in range(1):
    #     #             f.write(format_string(new_setting, trial, 'qg'))
    #     #             f.write(format_string(new_setting, trial, 'zeroes'))

    #     for pretrain_lr, finetune_lr in [(1e-3, 1e-4), (1e-4, 1e-5), (1e-5, 1e-6)]:
    #         new_setting = default_setting.copy()
    #         new_setting['finetune_lr'] = finetune_lr
    #         new_setting['pretrain_lr'] = pretrain_lr
    #         new_setting['pretrainer'] = pretrainer
    #         for trial in range(1):
    #             f.write(format_string(new_setting, trial, 'qg'))
    #             f.write(format_string(new_setting, trial, 'zeroes'))

    #     for aug_percent in [0.7, 0.9, 0.95, 0.98]:
    #         new_setting = default_setting.copy()
    #         if pretrainer == 'SimSiam':
    #             new_setting['pretrain_lr'] = 1e-5
    #             new_setting['finetune_lr'] = 0.0005
    #         new_setting['aug_percent'] = aug_percent
    #         new_setting['pretrainer'] = pretrainer
    #         for trial in range(1):
    #             f.write(format_string(new_setting, trial, 'qg'))
    #             f.write(format_string(new_setting, trial, 'zeroes'))

    #     if pretrainer == 'SimRank':
    #         for temp in [0.01, 0.05, 0.1, 0.5, 1., 10.]:
    #             new_setting = default_setting.copy()
    #             new_setting['temp'] = temp
    #             for trial in range(1):
    #                 f.write(format_string(new_setting, trial, 'qg'))
    #                 f.write(format_string(new_setting, trial, 'zeroes'))
