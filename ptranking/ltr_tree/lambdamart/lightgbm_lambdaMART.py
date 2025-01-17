#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import numpy as np
from itertools import product
from sklearn.datasets import load_svmlight_file
from ptranking.data.MSLR_dataset_filters import mslr_filters
from ptranking.data.yahoo1_dataset_filters import set1_filters
from ptranking.data.set2_dataset_filters import set2_filters
from ptranking.data.istella_filters import istella_filters
from scipy.sparse import vstack
import lightgbm as lgbm
import random
import torch
from lightgbm import Dataset
from scipy.sparse import hstack
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.data.data_utils import load_letor_data_as_libsvm_data, YAHOO_LTR, SPLIT_TYPE
from ptranking.data.binary_features import mslr_binary_features, yahoo_binary_features, istella_binary_features

from ptranking.ltr_tree.util.lightgbm_util import \
    lightgbm_custom_obj_lambdarank, lightgbm_custom_obj_ranknet, lightgbm_custom_obj_listnet,\
    lightgbm_custom_obj_lambdarank_fobj, lightgbm_custom_obj_ranknet_fobj, lightgbm_custom_obj_listnet_fobj

"""
The implementation of LambdaMART based on lightGBM,
 for details, please refer to https://github.com/microsoft/LightGBM
"""
def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)      
class LightGBMLambdaMART():
    """ LambdaMART based on lightGBM """

    def __init__(self, para_dict=None):
        self.id = 'LightGBMLambdaMART'
        self.custom_dict = para_dict['custom_dict']
        self.lightgbm_para_dict = para_dict['lightgbm_para_dict']

    def get_custom_obj(self, custom_obj_id, fobj=False):
        if fobj:
            if custom_obj_id == 'ranknet':
                return lightgbm_custom_obj_ranknet_fobj
            elif custom_obj_id == 'listnet':
                return lightgbm_custom_obj_listnet_fobj
            elif custom_obj_id == 'lambdarank':
                return lightgbm_custom_obj_lambdarank_fobj
            else:
                raise NotImplementedError
        else:
            if custom_obj_id == 'ranknet':
                return lightgbm_custom_obj_ranknet
            elif custom_obj_id == 'listnet':
                return lightgbm_custom_obj_listnet
            elif custom_obj_id == 'lambdarank':
                return lightgbm_custom_obj_lambdarank
            else:
                raise NotImplementedError
            

    def generate_robust_data(self, filters, batch_q_doc_vectors, group_test, y_test):
        group_cum = np.cumsum(group_test, dtype=np.int)
        split_vecs = np.split(batch_q_doc_vectors.toarray(), group_cum, axis=0)[:-1]
        split_labels = np.split(y_test, group_cum)[:-1]
        keep_indices = set()

        # split by query group
        for i, vec in enumerate(split_vecs):
            for j in range(len(filters)):
                direction = filters[j][0]
                curridx = filters[j][1]
                val_threshold = filters[j][2]
                if direction > 0:
                    if np.any((vec[:, curridx] > val_threshold).squeeze()):
                        keep_indices.add(i)
                else:
                    if np.any((vec[:, curridx] < val_threshold).squeeze()):
                        keep_indices.add(i)
        index_list = list(keep_indices)
        robust_vecs = [split_vecs[i] for i in index_list]
        robust_labels = [split_labels[i] for i in index_list]

        x_robust = np.vstack(robust_vecs)
        y_robust = np.concatenate(robust_labels)
        group_robust = group_test[index_list]
        print('Number of robust test groups', len(index_list))

        return x_robust, group_robust, y_robust

    
    def run(self, fold_k, file_train, file_vali, file_test, argobj, data_dict=None, eval_dict=None, save_model_dir=None):
        """
        Run lambdaMART model based on the specified datasets.
        :param fold_k:
        :param file_train:
        :param file_vali:
        :param file_test:
        :param data_dict:
        :param eval_dict:
        :return:
        """
        data_id, do_validation = data_dict['data_id'], eval_dict['do_validation']

        train_presort, validation_presort, test_presort = data_dict['train_presort'], data_dict['validation_presort'],\
                                                          data_dict['test_presort']
        
        
        # prepare training & testing datasets
        file_train_data, file_train_group = load_letor_data_as_libsvm_data(file_train, split_type=SPLIT_TYPE.Train,
                                                       data_dict=data_dict, eval_dict=eval_dict, presort=train_presort)
        x_train_full, y_train_full = load_svmlight_file(file_train_data)

        
        
        # x_train = x_train_full[:int(x_train_full.shape[0] * argobj.shrink),:]
        # y_train = y_train_full[:int(len(y_train_full) * argobj.shrink)]
        group_train_full = np.loadtxt(file_train_group)

        # pca = KernelPCA(
        #     n_components=5, kernel="rbf", gamma=1.0
        # )
        # poly_pca = KernelPCA(
        #     n_components=5, kernel="poly", degree=2, gamma=1.0
        # )
        pca = TruncatedSVD(n_components=argobj.dim)
        
            
        if argobj.shrink == 1.0:
            group_train = group_train_full
            x_train = x_train_full
            y_train = y_train_full
        else:
            group_train = group_train_full[:int(len(group_train_full) * argobj.shrink)]
            # num_groups = len(group_train)
            # group_indices = np.arange(num_groups)
            # group_start_indices = np.zeros(num_groups, dtype=int)
            # group_start_indices[1:] = np.cumsum(group_train)[:-1]
            # # Shuffle group indices
            # np.random.shuffle(group_indices)
            train_top_idx = np.sum(group_train)
            x_train = x_train_full[:int(train_top_idx),:]
            y_train = y_train_full[:int(train_top_idx)]
            # shuffled_X = []
            # shuffled_Y = []
            # for i in group_indices:
            #     start_idx = int(group_start_indices[i])
            #     end_idx = start_idx + int(group_train[i])
            #     x_slice = x_train[start_idx:end_idx, :]
            #     assert x_slice.ndim == 2 and x_slice.shape[0] > 0, "X slice is not 2D as expected"

            #     shuffled_X.append(x_slice)
            #     shuffled_Y.append(y_train[start_idx:end_idx])

            # x_train = vstack(shuffled_X)
            # y_train = np.concatenate(shuffled_Y)
            # group_train = group_train[group_indices]

        print(file_train.split('/')[-2])
        print(file_train)
        if file_train.split('/')[-2] == 'shift_sparse_t=4.0_tau=4.5':
            print('Load full dataset')
            dataset_dir = '/'.join(file_train.split('/')[:-2]) + '/'
            if data_dict['data_id'] == 'Set1':
                dataset_path = dataset_dir + 'set1.train.txt'
            else:
                dataset_path = dataset_dir + 'train.txt'

            file_train_data, file_train_group = load_letor_data_as_libsvm_data(dataset_path, split_type=SPLIT_TYPE.Train,
                                                       data_dict=data_dict, eval_dict=eval_dict, presort=train_presort)
            x_train_full, y_train_full = load_svmlight_file(file_train_data)
            group_train_full = np.loadtxt(file_train_group)
        # import ipdb; ipdb.set_trace()
        # print('Fitting the PCAs')
        # print(x_train_full.shape)

        # print(len(group_train))
        # print(np.sum(group_train))
        # print(y_train.shape)
        file_test_data, file_test_group = load_letor_data_as_libsvm_data(file_test, split_type=SPLIT_TYPE.Test,
                                                     data_dict=data_dict, eval_dict=eval_dict, presort=test_presort)
        x_test, y_test = load_svmlight_file(file_test_data)
        
        group_test = np.loadtxt(file_test_group)
        # print('test size', y_test.shape)
        # print(len(group_test))

        # test_set = Dataset(data=x_test, label=y_test, group=group_test)
        
        if data_dict['data_id'] == 'MSLRWEB30K':
            x_test_robust, group_test_robust, y_test_robust = self.generate_robust_data(mslr_filters, x_test, group_test, y_test)
            # rates = np.array([0.51380454, 0.32476065, 0.13444135, 0.01872947, 0.00826399])
            rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            categorical_features = list(mslr_binary_features.keys())
        elif data_dict['data_id'] == 'Set1':
            x_test_robust, group_test_robust, y_test_robust = self.generate_robust_data(set1_filters, x_test, group_test, y_test)
            # rates = np.array([0.27113375, 0.35319275, 0.27904, 0.07736027, 0.01927324])
            rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            categorical_features = list(yahoo_binary_features.keys())
        elif data_dict['data_id'] == 'Set2':
            x_test_robust, group_test_robust, y_test_robust = self.generate_robust_data(set2_filters, x_test, group_test, y_test)
        elif data_dict['data_id'] == 'Istella_S':
            x_test_robust, group_test_robust, y_test_robust = self.generate_robust_data(istella_filters, x_test, group_test, y_test)
            # rates = np.array([0.88218853, 0.02400172, 0.04109631, 0.02858308, 0.02413036])
            rates = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            categorical_features = list(istella_binary_features.keys())
        rates = np.ones(31)/31.0

        # PCA transformed
        if argobj.dim > 0:
            print('PCA dim', argobj.dim)
            fitted_transform = pca.fit(x_train_full)
            x_train_append = fitted_transform.transform(x_train)
            x_train = hstack([x_train, x_train_append])
            x_test_append = fitted_transform.transform(x_test)
            x_test = hstack([x_test, x_test_append])
            x_test_robust_append = fitted_transform.transform(x_test_robust)
            x_test_robust = np.concatenate((x_test_robust, x_test_robust_append), axis=1)

        train_set = Dataset(data=x_train, label=y_train, group=group_train, free_raw_data=False)


        if do_validation: # prepare validation dataset if needed
            file_vali_data, file_vali_group=load_letor_data_as_libsvm_data(file_vali, split_type=SPLIT_TYPE.Validation,
                                                data_dict=data_dict, eval_dict=eval_dict, presort=validation_presort)
            x_valid, y_valid = load_svmlight_file(file_vali_data)
            print('val size', y_valid.shape)
            

            group_valid = np.loadtxt(file_vali_group)
            print(len(group_valid))
            valid_set = Dataset(data=x_valid, label=y_valid, group=group_valid, free_raw_data=False)

            if self.custom_dict['custom'] and self.custom_dict['use_LGBMRanker']:
                lgbm_ranker = lgbm.LGBMRanker()
                lgbm_ranker.set_params(**self.lightgbm_para_dict)
                '''
                objective : string, callable or None, optional (default=None)
                Specify the learning task and the corresponding learning objective or
                a custom objective function to be used (see note below).
                Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
                '''
                custom_obj_dict = dict(objective=self.get_custom_obj(custom_obj_id=self.custom_dict['custom_obj_id']))
                lgbm_ranker.set_params(**custom_obj_dict)
                '''
                eval_set (list or None, optional (default=None)) – A list of (X, y) tuple pairs to use as validation sets.
                cf. https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html
                '''
                lgbm_ranker.fit(x_train, y_train, group=group_train,
                                eval_set=[(x_valid, y_valid)], eval_group=[group_valid], eval_at=[5],
                                early_stopping_rounds=eval_dict['early_stop_or_boost_round'],
                                verbose=10)

            elif self.custom_dict['custom']:
                # use the argument of fobj
                lgbm_ranker = lgbm.train(params=self.lightgbm_para_dict, verbose_eval=10,
                                         train_set=train_set, valid_sets=[valid_set],
                                         early_stopping_rounds=eval_dict['early_stop_or_boost_round'],
                                         fobj=self.get_custom_obj(custom_obj_id=self.custom_dict['custom_obj_id'],
                                                                  fobj=True))
            else: # trained booster as ranker
                # train on small set
                self.lightgbm_para_dict['random_state'] = argobj.trial_num
                # self.lightgbm_para_dict['categorical_features'] = categorical_features
                # tuned in 0.5, 0.7, 0.9
                lgbm_ranker = lgbm.train(params=self.lightgbm_para_dict, verbose_eval=10,
                                         train_set=train_set, valid_sets=[valid_set],
                                         early_stopping_rounds=eval_dict['early_stop_or_boost_round'])
                print(self.lightgbm_para_dict)
                
                # pseudolabel
                if argobj.layers == 1:
                    if argobj.dim > 0:
                        x_train_full_append = fitted_transform.transform(x_train_full)
                        x_train_full = hstack([x_train_full, x_train_full_append])
                    print(x_train_full.shape)
                    y_pseudo_full = lgbm_ranker.predict(x_train_full)
                    # y_pseudo_min = np.min(y_pseudo_full)
                    # y_pseudo_pos = (y_pseudo_full - y_pseudo_min)
                    # y_max = np.max(y_pseudo_pos)
                    # rescaling = 4.4 / y_max
                    # y_pseudo_pos = y_pseudo_pos * rescaling
                    # y_pseudo_int = y_pseudo_pos.astype(int)
                    #============SORT BY CUMULATIVE================
                    sorted_indices = np.argsort(y_pseudo_full)
                    sorted_y = y_pseudo_full[sorted_indices]
                    cumulative_rates = np.cumsum(rates)
                    labels = np.zeros_like(sorted_y, dtype=int)  # Initialize an array for labels
                    n = len(y_pseudo_full)
                    threshold_indices = (cumulative_rates * n).astype(int)
                    prev_index = 0
                    for i, threshold in enumerate(threshold_indices):
                        labels[prev_index:threshold] = i
                        prev_index = threshold
                    original_order_labels = np.zeros_like(labels)
                    original_order_labels[sorted_indices] = labels
                    y_pseudo_int = original_order_labels
                    #============SORT BY CUMULATIVE================
                    # y_pseudo_int = np.argsort(np.argsort(y_pseudo_full)).astype(int)

                    print('Num unique relevances', len(np.unique(y_pseudo_int)))
                    # y_pseudo_int[:int(train_top_idx)] = y_train_full[:int(train_top_idx)]
                    y_pseudo = Dataset(data=x_train_full, label=y_pseudo_int, group=group_train_full)
                    print('\n\n==============Pseudolabeled===============\n\n')
                    lgbm_ranker = lgbm.train(params=self.lightgbm_para_dict, verbose_eval=10,
                                            train_set=y_pseudo, valid_sets=[valid_set],
                                            early_stopping_rounds=eval_dict['early_stop_or_boost_round'])




        else: # without validation
            if self.custom_dict['custom'] and self.custom_dict['use_LGBMRanker']:
                lgbm_ranker = lgbm.LGBMRanker()
                lgbm_ranker.set_params(**self.lightgbm_para_dict)

                custom_obj_dict = dict(objective=self.get_custom_obj(custom_obj_id=self.custom_dict['custom_obj_id']))
                lgbm_ranker.set_params(**custom_obj_dict)

                lgbm_ranker.fit(x_train, y_train, group=group_train, verbose=10, eval_at=[5],
                                early_stopping_rounds=eval_dict['early_stop_or_boost_round'])

            elif self.custom_dict['custom']: # use the argument of fobj
                lgbm_ranker = lgbm.train(params=self.lightgbm_para_dict, verbose_eval=10,
                                         train_set=train_set, early_stopping_rounds=eval_dict['early_stop_or_boost_round'],
                                         fobj=self.get_custom_obj(custom_obj_id=self.custom_dict['custom_obj_id'],
                                                                  fobj=True))

            else: # trained booster as ranker
                lgbm_ranker = lgbm.train(params=self.lightgbm_para_dict,  verbose_eval=10,
                                         train_set=train_set, early_stopping_rounds=eval_dict['early_stop_or_boost_round'])

        if data_id in YAHOO_LTR:
            model_file = save_model_dir + 'model.txt'
        else:
            model_file = save_model_dir + '_'.join(['fold', str(fold_k), 'model'])+'.txt'

        if self.custom_dict['custom'] and self.custom_dict['use_LGBMRanker']:
            lgbm_ranker.booster_.save_model(model_file)
        else:
            lgbm_ranker.save_model(model_file)

        y_pred = lgbm_ranker.predict(x_test)  # fold-wise prediction
        y_pred_robust = lgbm_ranker.predict(x_test_robust)

        return y_test, group_test, y_pred, y_test_robust, group_test_robust, y_pred_robust
        # return y_test, group_test, y_pred


###### Parameter of LambdaMART ######

class LightGBMLambdaMARTParameter(ModelParameter):
    ''' Parameter class for LambdaMART based on LightGBM '''

    def __init__(self, debug=False, para_json=None):
        super(LightGBMLambdaMARTParameter, self).__init__(model_id='LightGBMLambdaMART', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for LambdaMART
        :return:
        """
        # for custom setting
        #custom_dict = dict(custom=False, custom_obj_id='lambdarank', use_LGBMRanker=True) #
        custom_dict = dict(custom=False, custom_obj_id=None)

        # common setting when using in-built lightgbm's ranker
        lightgbm_para_dict = {'boosting_type': 'gbdt',   # ltr_gbdt, dart
                              'objective': 'lambdarank', # will be updated if performing customization
                              'metric': 'ndcg',
                              'learning_rate': 0.05,
                              'num_leaves': 400,
                              'num_trees': 1000,
                              'num_threads': 16,
                              'min_data_in_leaf': 50,
                              'min_sum_hessian_in_leaf': 200,
                              'eval_at': 5, # which matters much (early stopping), say setting as 5 is better than default
                              # 'lambdamart_norm':False,
                              # 'is_training_metric':True,
                              'verbosity': -1}

        self.para_dict = dict(custom_dict=custom_dict, lightgbm_para_dict=lightgbm_para_dict)

        return self.para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        para_dict = given_para_dict if given_para_dict is not None else self.para_dict
        lightgbm_para_dict = para_dict['lightgbm_para_dict']

        s1, s2 = (':', '\n') if log else ('_', '_')

        BT, metric, num_leaves, num_trees, min_data_in_leaf, min_sum_hessian_in_leaf, lr, eval_at = \
            lightgbm_para_dict['boosting_type'], lightgbm_para_dict['metric'], lightgbm_para_dict['num_leaves'],\
            lightgbm_para_dict['num_trees'], lightgbm_para_dict['min_data_in_leaf'],\
            lightgbm_para_dict['min_sum_hessian_in_leaf'], lightgbm_para_dict['learning_rate'],\
            lightgbm_para_dict['eval_at']

        para_string = s2.join([s1.join(['BT', BT]), s1.join(['Metric', metric]),
                               s1.join(['Leaves', str(num_leaves)]), s1.join(['Trees', str(num_trees)]),
                               s1.join(['MiData', '{:,g}'.format(min_data_in_leaf)]),
                               s1.join(['MSH', '{:,g}'.format(min_sum_hessian_in_leaf)]),
                               s1.join(['LR', '{:,g}'.format(lr)]), s1.join(['EvalAt', str(eval_at)])
                               ])

        return para_string

    def get_identifier(self):
        if self.para_dict['custom_dict']['custom'] and self.para_dict['custom_dict']['use_LGBMRanker']:
            return '_'.join([self.model_id, 'Custom', self.para_dict['custom_dict']['custom_obj_id']])

        elif self.para_dict['custom_dict']['custom']:
            return '_'.join([self.model_id, 'CustomFobj', self.para_dict['custom_dict']['custom_obj_id']])
        else:
            return self.model_id

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        """
        # for custom setting
        #custom_dict = dict(custom=False, custom_obj_id='lambdarank', use_LGBMRanker=False)
        custom_dict = dict(custom=False, custom_obj_id=None)

        if self.use_json:
            choice_BT = self.json_dict['BT']
            choice_metric = self.json_dict['metric']
            choice_leaves = self.json_dict['leaves']
            choice_trees = self.json_dict['trees']
            choice_MiData = self.json_dict['MiData']
            choice_MSH = self.json_dict['MSH']
            choice_LR = self.json_dict['LR']
            eval_at = self.json_dict['eval_at']
        else:
            # common setting when using in-built lightgbm's ranker
            choice_BT = ['gbdt'] if self.debug else ['gbdt']
            choice_metric = ['ndcg'] if self.debug else ['ndcg']
            choice_leaves = [400] if self.debug else [400]
            choice_trees = [1000] if self.debug else [1000]
            choice_MiData = [50] if self.debug else [50]
            choice_MSH = [200] if self.debug else [200]
            choice_LR = [0.05, 0.01] if self.debug else [0.05, 0.01]
            eval_at = 5

        for BT, metric, num_leaves, num_trees, min_data_in_leaf, min_sum_hessian_in_leaf, lr in product(choice_BT,
                                choice_metric, choice_leaves, choice_trees, choice_MiData, choice_MSH, choice_LR):
            lightgbm_para_dict = {'boosting_type': BT,  # ltr_gbdt, dart
                                     'objective': 'lambdarank',
                                     'metric': metric,
                                     'learning_rate': lr,
                                     'num_leaves': num_leaves,
                                     'num_trees': num_trees,
                                     'num_threads': 16,
                                     'min_data_in_leaf': min_data_in_leaf,
                                     'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
                                     # 'lambdamart_norm':False,
                                     # 'is_training_metric':True,
                                     'eval_at': eval_at, # which matters much (early stopping), say setting as 5 is better than default
                                  #'max_bin': 64,
                                  #'max_depth':4,
                                     'verbosity': -1}

            self.para_dict = dict(custom_dict=custom_dict, lightgbm_para_dict=lightgbm_para_dict)
            yield self.para_dict
