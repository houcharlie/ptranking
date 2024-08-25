#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
"""

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.base.utils import get_stacked_FFNet, get_resnet
from ptranking.metric.metric_utils import get_delta_ndcg
from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from torch import nn 
from ptranking.data.binary_features import mslr_binary_features, yahoo_binary_features, istella_binary_features


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class LambdaRank(AdhocNeuralRanker):
    '''
    Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
    Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(LambdaRank, self).__init__(id='LambdaRank', sf_para_dict=sf_para_dict, gpu=gpu, device=device)
        self.sigma = model_para_dict['sigma']
    def init(self):
        # self.point_sf = self.config_point_neural_scoring_function()
        # for i in range(2):
        #     nr_hn = nn.Linear(136, 136)
        #     self.point_sf.add_module('_'.join(['ff', 'scoring', str(i)]), nr_hn)
        # nr_hn = nn.Linear(136, 1)
        # self.point_sf.add_module('_'.join(['ff', 'scoring']), nr_hn)
        # self.point_sf.to(self.device)

        # self.scheduler = StepLR(optimizer=self.optimizer, step_size=40, gamma=1.)
        self.point_sf, self.linear_weight, self.embeddings, self.mappings, self.categorical_features, self.dataset, self.fm, self.linear1, self.linear2, self.linear3 = self.config_point_neural_scoring_function()
        self.config_optimizer()
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=40, gamma=1.)
   
    
    def config_point_neural_scoring_function(self):
        point_sf, linear_weight, embeddings, mappings, categorical_features, dataset, fm, linear1, linear2, linear3 = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: 
            point_sf = point_sf.to(self.device)
            linear_weight = linear_weight.to(self.device)
            embeddings = embeddings.to(self.device)
            fm = fm.to(self.device)
            linear1 = linear1.to(self.device)
            linear2 = linear2.to(self.device)
            linear3 = linear3.to(self.device)
            
        return point_sf, linear_weight, embeddings, mappings, categorical_features, dataset, fm, linear1, linear2, linear3

    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=136, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                    BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        if num_features == 136:
            dataset = 'mslr'
            categorical_features = mslr_binary_features
        elif num_features == 220:
            dataset = 'istella'
            categorical_features = istella_binary_features
        elif num_features == 700:
            dataset = 'yahoo'
            categorical_features = yahoo_binary_features
        else:
            print('Num features not matching any of the dataasets')
        

        # using 6 * (3) ^ 1/4, the dimension given in DCN v2 https://arxiv.org/pdf/2008.13535.pdf
        embeddings = nn.ModuleDict({
            str(key): nn.Embedding(len(value), 8) for key, value in categorical_features.items()
        })
        mappings = self.prepare_mappings(categorical_features)
        num_categorical_features = len(categorical_features)
        dnn_features = num_features - num_categorical_features + 8 * num_categorical_features
        linear1 = nn.Linear(dnn_features, dnn_features)
        linear2 = nn.Linear(dnn_features, dnn_features)
        linear3 = nn.Linear(dnn_features, dnn_features)
        
        # For DeepFM===========================================================================
        # point_sf = nn.Sequential(
        #     nn.Linear(dnn_features, 256), # First linear layer
        #     nn.ReLU(),                 # ReLU after first linear layer
        #     nn.Linear(256, 128),       # Second linear layer
        #     nn.ReLU(),                 # ReLU after second linear layer
        #     nn.Linear(128, 1)          # Final linear layer
        # )
        # linear_weight = nn.Linear(num_features - len(categorical_features.keys()), 1, bias=False)
        # For DeepFM===========================================================================

        # For DCN v2===========================================================================
        # point_sf = nn.Sequential(
        #     nn.Linear(dnn_features, 128), # First linear layer
        #     nn.ReLU(),                 # ReLU after first linear layer
        #     nn.Linear(128, 128),       # Second linear layer
        #     nn.ReLU(),                 # ReLU after second linear layer
        # )
        linear_weight = nn.Linear(128 + dnn_features, 1, bias=False)
        # For DCN v2===========================================================================

        # For normal===========================================================================
        point_sf = get_resnet(dnn_features, 136)
        point_sf.add_module('end_linear', nn.Linear(136, 1))
        # For normal===========================================================================
        fm = FM()
        return point_sf, linear_weight, embeddings, mappings, categorical_features, dataset, fm, linear1, linear2, linear3
    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        assert 'label_type' in kwargs and LABEL_TYPE.MultiLabel == kwargs['label_type']
        label_type = kwargs['label_type']
        assert 'presort' in kwargs and kwargs['presort'] is True  # aiming for direct usage of ideal ranking

        # sort documents according to the predicted relevance
        batch_descending_preds, batch_pred_desc_inds = torch.sort(batch_preds, dim=1, descending=True)
        # reorder batch_stds correspondingly so as to make it consistent.
        # BTW, batch_stds[batch_preds_sorted_inds] only works with 1-D tensor
        batch_predict_rankings = torch.gather(batch_std_labels, dim=1, index=batch_pred_desc_inds)

        batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=batch_descending_preds,
                                                             batch_std_labels=batch_predict_rankings,
                                                             sigma=self.sigma)

        batch_delta_ndcg = get_delta_ndcg(batch_ideal_rankings=batch_std_labels,
                                          batch_predict_rankings=batch_predict_rankings,
                                          label_type=label_type, device=self.device)

        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1),
                                             weight=torch.triu(batch_delta_ndcg, diagonal=1), reduction='none')

        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss

    def prepare_mappings(self, categorical_features):
        mappings = {}
        for feature_index, possible_values in categorical_features.items():
            # Convert possible values to tensor for efficient operations
            value_tensor = torch.tensor(possible_values, dtype=torch.float32).to(self.device)
            mappings[feature_index] = value_tensor
        return mappings
    
    def separate_and_convert_features(self, batch_q_doc_vectors):
        cat_features_list = []
        dense_features_indices = [i for i in range(batch_q_doc_vectors.shape[2]) if i not in self.categorical_features]

        for feature_index, possible_values in self.mappings.items():
            feature_values = batch_q_doc_vectors[:, :, feature_index]
            
            # Broadcast comparison to create a boolean mask
            comparison_mask = feature_values.unsqueeze(-1) == possible_values

            # Convert boolean mask to indices
            indices = torch.argmax(comparison_mask.float(), dim=-1)

            # Get embeddings for categorical features
            embedded_feature = self.embeddings[str(feature_index)](indices)
            cat_features_list.append(embedded_feature)

        # Extract dense features
        dense_features = batch_q_doc_vectors[:, :, dense_features_indices]

        # Concatenate all categorical features embeddings
        cat_features_embeddings = torch.stack(cat_features_list, dim=2)

        return dense_features, cat_features_embeddings

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        # [batch size, num docs, num dense features], [batch size, num docs, num categorical features, embedding dimension]
        dense_features, cat_feature_embeddings = self.separate_and_convert_features(batch_q_doc_vectors)
        _, _, num_cat_features, embed_size = cat_feature_embeddings.shape
        _, _, num_dense_features = dense_features.shape
        input_to_dnn = torch.cat([dense_features, cat_feature_embeddings.reshape(batch_size, num_docs, num_cat_features * embed_size)], dim=2)

        # # FM part
        # =======================================FM==============================================
        # input_to_fm = cat_feature_embeddings.reshape(batch_size * num_docs, num_cat_features, embed_size)
        # score = self.fm(input_to_fm).reshape(batch_size, num_docs)
        # input_to_linear_weight = dense_features.reshape(batch_size, num_docs, num_dense_features)
        # score += self.linear_weight(input_to_linear_weight).reshape(batch_size, num_docs)
        # input_to_linear_cat = cat_feature_embeddings.reshape(batch_size, num_docs, num_cat_features * embed_size)
        # score += torch.sum(input_to_linear_cat, dim=2).reshape(batch_size, num_docs)
        # score = self.point_sf(input_to_dnn).reshape(batch_size, num_docs)
        # =======================================FM==============================================

        # # DCNv2 part
        # ========================================DCNv2=============================================
        deep_out = self.point_sf(input_to_dnn).reshape(batch_size, num_docs, 128)
        x_0 = input_to_dnn
        dot_ = self.linear1(x_0)
        x_1 = torch.mul(x_0, dot_) + x_0
        dot_ = self.linear2(x_1)
        x_2 = torch.mul(x_1, dot_) + x_0
        dot_ = self.linear3(x_2)
        cross_out = torch.mul(x_2, dot_) + x_0
        input_to_linear = torch.cat([deep_out, cross_out], dim=2)
        score = self.linear_weight(input_to_linear).reshape(batch_size, num_docs)

        # ========================================DCNv2=============================================
        return score

###### Parameter of LambdaRank ######

class LambdaRankParameter(ModelParameter):
    ''' Parameter class for LambdaRank '''
    def __init__(self, debug=False, para_json=None):
        super(LambdaRankParameter, self).__init__(model_id='LambdaRank', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for LambdaRank
        :return:
        """
        self.lambda_para_dict = dict(model_id=self.model_id, sigma=1.0)
        return self.lambda_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        lambda_para_dict = given_para_dict if given_para_dict is not None else self.lambda_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        lambdarank_para_str = s1.join(['Sigma', '{:,g}'.format(lambda_para_dict['sigma'])])
        return lambdarank_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        """
        if self.use_json:
            choice_sigma = self.json_dict['sigma']
        else:
            choice_sigma = [5.0, 1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0

        for sigma in choice_sigma:
            self.lambda_para_dict = dict(model_id=self.model_id, sigma=sigma)
            yield self.lambda_para_dict
