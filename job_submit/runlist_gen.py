
# with open('./runlist.txt', 'w+') as f:
#     for finetune_lr in [0.001, 0.0005]:
#         for aug_percent in [0.8, 0.9]:
#             for aug_type in ['qg', 'zeroes', 'none']:
#                 for trial in range(3):
#                     f.write('{0} {1} {2} {3}\n'.format(finetune_lr, aug_percent, aug_type, trial))

default_setting = {
    "pretrain_lr": 1e-5,
    "finetune_lr": 1e-6,
    "aug_percent": 0.8,
    "dim": 16,
    "layers": 5,
    "temp": 0.07,
    'pretrainer': 'SimRank',
    'mix': 0.5
}
def format_string(new_setting, trial, aug):
    return '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n'.format(new_setting['pretrain_lr'], new_setting['finetune_lr'], trial, aug, new_setting['aug_percent'], new_setting['dim'], new_setting['layers'], new_setting['temp'], new_setting['pretrainer'], new_setting['mix'])
with open('./runlist.txt', 'w+') as f:
    # explore lrs    
    for pretrain_lr, finetune_lr in [(1e-5, 1e-6), (1e-4, 1e-5)]:
        new_setting = default_setting.copy()
        new_setting['finetune_lr'] = finetune_lr
        new_setting['pretrain_lr'] = pretrain_lr
        for trial in range(1):
            f.write(format_string(new_setting, trial, 'qg'))
            f.write(format_string(new_setting, trial, 'zeroes'))
    
    for aug_percent in [0.3, 0.5, 0.7, 0.8]:
        new_setting = default_setting.copy()
        new_setting['aug_percent'] = aug_percent
        for trial in range(1):
            f.write(format_string(new_setting, trial, 'qg'))
            f.write(format_string(new_setting, trial, 'zeroes'))
    
    for mix in [0., 0.25, 0.75, 1.]:
        new_setting = default_setting.copy()
        new_setting['mix'] = mix
        for trial in range(1):
            f.write(format_string(new_setting, trial, 'qg'))
            f.write(format_string(new_setting, trial, 'zeroes'))

    for temp in [0.01, 0.1]:
        new_setting = default_setting.copy()
        new_setting['temp'] = temp
        for trial in range(1):
            f.write(format_string(new_setting, trial, 'qg'))
            f.write(format_string(new_setting, trial, 'zeroes'))
    
    for dim in [8, 32, 100]:
        new_setting = default_setting.copy()
        new_setting['temp'] = temp
        for trial in range(1):
            f.write(format_string(new_setting, trial, 'qg'))
            f.write(format_string(new_setting, trial, 'zeroes'))
