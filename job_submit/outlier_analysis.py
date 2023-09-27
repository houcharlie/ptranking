import numpy as np
import matplotlib.pyplot as plt

# val = np.load('/ocean/projects/cis230033p/houc/datasets/mslr/BufferedFold1/vali_MiR_1_MiD_10_PerQ_PreSort_QS_StandardScaler.np', allow_pickle=True)
# val = np.load('/ocean/projects/cis230033p/houc/datasets/yahoo/ltrc_yahoo/set1.valid_MiR_1_MiD_10_PerQ_PreSort_QS_StandardScaler.np', allow_pickle=True)
val = np.load('/ocean/projects/cis230033p/houc/datasets/istella-s-letor/sample/vali_MiR_1_MiD_10_PerQ_PreSort_QS_StandardScaler.np', allow_pickle=True)
val_stack = []
for i in range(len(val)):
    original_np = val[i][1]
    num_docs, num_features = original_np.shape
    flatten_np = np.reshape(original_np, newshape=(-1, num_features))
    val_stack.append(flatten_np)

all_data = np.concatenate(val_stack, axis = 0)
num_dims = all_data.shape[1]

# for i in range(num_dims):
plt.hist(all_data[:,174], log=True, bins=100)
plt.title('Histogram for Istella-validation dim=174'.format(i), fontsize=18)
plt.xlabel('Feature value', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.savefig('/jet/home/houc/ptranking/job_submit/outlier.png'.format(i), dpi=1200, pad_inches='tight')
plt.close()
