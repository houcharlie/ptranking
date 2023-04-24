import os
import pickle

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


metric_list = ['val/ndcg@3', 'val/ndcg@5','val/ndcg@10','val/ndcg@20']

top_folder = '/ocean/projects/iri180031p/houc/ranking/MSLR-WEB30K/ssl/rankneg_hparam/'

subdirs = get_immediate_subdirectories(top_folder)

max_ndcg = 0
best_ndcg_params = None
max_robust = 0
best_robust_params = None
for subdir in subdirs:
    curr_metrics = top_folder + subdir + '/metrics.pickle'
    curr_hparams = top_folder + subdir + '/hparam.pickle'
    with open(curr_metrics, 'rb') as handle:
        b = pickle.load(handle)
    with open(curr_hparams, 'rb') as handle:
        a = pickle.load(handle)
    if b['val/ndcg@5'] > max_ndcg:
        max_ndcg = b['val/ndcg@5']
        best_ndcg_params = a
    if b['val/robust-ndcg@5'] > max_robust:
        max_robust = b['val/robust-ndcg@5']
        best_robust_params = a


print('Best ndcg', max_ndcg, best_ndcg_params)
print('best robust', max_robust, best_robust_params)
    
