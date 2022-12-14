
with open('./runlist.txt', 'w+') as f:
    for finetune_lr in [0.001, 0.0005]:
        for aug_percent in [0.8, 0.9]:
            for aug_type in ['qg', 'zeroes', 'none']:
                for trial in range(3):
                    f.write('{0} {1} {2} {3}\n'.format(finetune_lr, aug_percent, aug_type, trial))

