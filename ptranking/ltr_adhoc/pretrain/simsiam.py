"""
Row-wise simsiam pretraining
"""

import torch
import torch.nn as nn
import os
import sys
from itertools import product
from ptranking.base.utils import get_stacked_FFNet
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from absl import logging
class SimSiam(NeuralRanker):
    ''' SimSiam '''
    """
    Original SimSiam uses a Resnet-50. 
    ---Original SimSiam specs--- 
    Encoder was: 50176 -> 2048 (divide by 24.8)
    Projector: 2048 -> 2048
    Predictor: 2048 -> 512 -> 2048 (divide by 4)
    ---Our proposed specs---
    Encoder: 138 -> dim
    Projector: dim -> dim
    Predictor: dim -> dim/4 -> dim
    """
    def __init__(self, id='SimSiamPretrainer', sf_para_dict=None, model_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(SimSiam, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.aug_percent = model_para_dict['aug_percent']
        self.dim = model_para_dict['dim']
        self.aug_type = model_para_dict['aug_type']
        if self.aug_type == 'zeroes':
            self.augmentation = zeroes
        elif self.aug_type == 'qg':
            self.augmentation = qgswap

    def init(self):
        self.point_sf = self.config_point_neural_scoring_function()
        self.projector, self.predictor = self.config_heads()
        self.loss = nn.CosineSimilarity(dim=1).to(self.device)
        self.config_optimizer()

    def config_point_neural_scoring_function(self):
        point_sf = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: point_sf = point_sf.to(self.device)
        return point_sf

    def config_heads(self):
        dim = self.dim
        prev_dim = self.point_sf.ff_7.weight.shape[0]
        projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim),
                                  nn.ReLU(), # first layer
                                  nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim, affine=False),
                                  nn.ReLU(inplace=True), # second layer
                                  nn.Linear(prev_dim, dim, bias=False),
                                  nn.BatchNorm1d(dim, affine=False))
        if self.gpu: projector = projector.to(self.device)

        pred_dim = int(dim // 4)
        
        predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                    nn.BatchNorm1d(pred_dim),
                                    nn.ReLU(inplace=True), # hidden layer
                                    nn.Linear(pred_dim, dim)) # output layer
        if self.gpu: predictor = predictor.to(self.device)

        return projector, predictor

    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.projector.parameters():
            all_params.append(param)
        for param in self.predictor.parameters():
            all_params.append(param)
        
        return all_params

    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=136, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                    BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        encoder_layers = num_layers
        ff_dims = [num_features]
        for i in range(encoder_layers):
            ff_dims.append(h_dim)
        ff_dims.append(out_dim)

        point_sf = get_stacked_FFNet(ff_dims=ff_dims, AF=AF, TL_AF=TL_AF, apply_tl_af=apply_tl_af, dropout=dropout,
                                     BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)
        return point_sf

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        x1 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)
        x2 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device)
        data_dim = batch_q_doc_vectors.shape[2]
        x1_flat = x1.reshape((-1, data_dim))
        x2_flat = x2.reshape((-1, data_dim))
        z1 = self.projector(self.point_sf(x1_flat))
        z2 = self.projector(self.point_sf(x2_flat))

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

    def eval_mode(self):
        self.point_sf.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.point_sf.state_dict(), dir + name)

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.point_sf.load_state_dict(torch.load(file_model, map_location=device))

    def get_tl_af(self):
        return self.sf_para_dict[self.sf_para_dict['sf_id']]['TL_AF']
    
    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        p1, p2, z1, z2 = batch_preds
        loss = -(self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) * 0.5
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train_op(self, batch_q_doc_vectors, batch_std_labels, **kwargs):
        '''
        The training operation over a batch of queries.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance labels for documents associated with the same query.
        @param kwargs: optional arguments
        @return:
        '''
        stop_training = False
        batch_preds = self.forward(batch_q_doc_vectors)

        return self.custom_loss_function(batch_preds, batch_std_labels, **kwargs), stop_training
    
    def validation(self, vali_data=None, vali_metric=None, k=5, presort=False, max_label=None, label_type=LABEL_TYPE.MultiLabel, device='cpu'):
        self.eval_mode() # switch evaluation mode

        num_queries = 0
        sum_val_loss = torch.zeros(1).to(self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in vali_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu: batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            val_loss = self.custom_loss_function(batch_preds, batch_std_labels)

            sum_val_loss += val_loss # due to batch processing

        avg_val_loss = val_loss / num_queries
        return avg_val_loss.cpu()

    def adhoc_performance_at_ks(self, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, max_label=None,
                                presort=False, device='cpu', need_per_q=False):
        '''
        Compute the performance using multiple metrics
        '''
        self.eval_mode()  # switch evaluation mode

        val_loss = self.validation(test_data)
        output = torch.zeros(len(ks))
        output[0] = val_loss
        output[1] = val_loss
        return output, output, output, output, output

###### Parameter of LambdaRank ######

class SimSiamParameter(ModelParameter):
    ''' Parameter class for SimSiam '''
    def __init__(self, debug=False, para_json=None):
        super(SimSiamParameter, self).__init__(model_id='SimSiam', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for SimSiam
        :return:
        """
        self.simsiam_para_dict = dict(model_id=self.model_id, aug_percent=0.7, dim=100, aug_type='qg')
        return self.simsiam_para_dict

    def to_para_string(self, log=False, given_para_dict=None):
        """
        String identifier of parameters
        :param log:
        :param given_para_dict: a given dict, which is used for maximum setting w.r.t. grid-search
        :return:
        """
        # using specified para-dict or inner para-dict
        simsiam_para_dict = given_para_dict if given_para_dict is not None else self.simsiam_para_dict

        s1, s2 = (':', '\n') if log else ('_', '_')
        simsiam_para_str = s1.join(['aug_percent', '{:,g}'.format(simsiam_para_dict['aug_percent']), 'embed_dim', '{:,g}'.format(simsiam_para_dict['dim']), 'aug_type', simsiam_para_dict['aug_type']])
        return simsiam_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for simsiam
        """
        if self.use_json:
            choice_aug = self.json_dict['aug_percent']
            choice_dim = self.json_dict['dim']
            choice_augtype = self.json_dict['aug_type']
        else:
            choice_aug = [0.3, 0.7] if self.debug else [0.7]  # 1.0, 10.0, 50.0, 100.0
            choice_dim = [50, 100] if self.debug else [100]  # 1.0, 10.0, 50.0, 100.0
            choice_augtype = ['zeroes', 'qg'] if self.debug else ['qg']  # 1.0, 10.0, 50.0, 100.0

        for aug_percent, dim, augtype in product(choice_aug, choice_dim, choice_augtype):
            self.simsiam_para_dict = dict(model_id=self.model_id, aug_percent=aug_percent, dim=dim, aug_type=augtype)
            yield self.simsiam_para_dict
