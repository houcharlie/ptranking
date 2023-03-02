"""
SimRank
"""

import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
import math

from itertools import product
from torch.optim.lr_scheduler import StepLR
from ptranking.base.utils import get_stacked_FFNet, LTRBatchNorm
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap, qg_and_zero
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from ptranking.metric.metric_utils import get_delta_ndcg
from absl import logging
class SimSiamRank(NeuralRanker):
    ''' SimRank '''

    def __init__(self, id='SimSiamRankPretrainer', sf_para_dict=None, model_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(SimSiamRank, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)
        self.aug_percent = model_para_dict['aug_percent']
        self.dim = model_para_dict['dim']
        self.aug_type = model_para_dict['aug_type']
        self.temperature = model_para_dict['temp']
        self.mix = model_para_dict['mix']
        if self.aug_type == 'zeroes':
            self.augmentation = zeroes
        elif self.aug_type == 'qg':
            self.augmentation = qgswap
        elif self.aug_type == 'qz':
            self.augmentation = qg_and_zero

    def init(self):
        self.point_sf = self.config_point_neural_scoring_function()
        # nr_hn = nn.Linear(136, 1)
        # self.point_sf.add_module('_'.join(['ff', 'scoring']), nr_hn)
        self.point_sf.to(self.device)
        self.projector, self.predictor, self.scorer = self.config_heads()
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.config_optimizer()
        print(self.point_sf, file=sys.stderr)
        print(self.projector, file=sys.stderr)
        print(self.predictor, file=sys.stderr)
        print(self.scorer, file=sys.stderr)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=1.)

    def config_point_neural_scoring_function(self):
        point_sf = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: point_sf = point_sf.to(self.device)
        return point_sf

    def config_heads(self):
        dim = self.dim
        prev_dim = -1
        for name, param in self.point_sf.named_parameters():
            if 'ff' in name and 'bias' not in name:
                prev_dim = param.shape[0]
        print('Input dimension', prev_dim, file=sys.stderr)
        print('Operating dimension', dim, file=sys.stderr)
        projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                  LTRBatchNorm(prev_dim),
                                  nn.ReLU(), # first layer
                                  nn.Linear(prev_dim, prev_dim, bias=False),
                                  LTRBatchNorm(prev_dim, affine=False),
                                  nn.ReLU(), # second layer
                                  nn.Linear(prev_dim, dim, bias=False),
                                  LTRBatchNorm(dim, affine=False),
                                  nn.ReLU())
        
        if self.gpu: projector = projector.to(self.device)

        scorer = nn.Linear(dim, 1)

        if self.gpu: scorer = scorer.to(self.device)

        pred_dim = int(dim // 4)
        
        predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                    LTRBatchNorm(pred_dim),
                                    nn.ReLU(), # hidden layer
                                    nn.Linear(pred_dim, dim),
                                    nn.ReLU()) # output layer
        if self.gpu: predictor = predictor.to(self.device)


        return projector, predictor, scorer

    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.projector.parameters():
            all_params.append(param)
        for param in self.predictor.parameters():
            all_params.append(param)
        for param in self.scorer.parameters():
            all_params.append(param)
        
        return nn.ParameterList(all_params)
        # return self.point_sf.parameters()

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
    
    
    def get_pairwise_comp_probs_soft(self, batch_preds, batch_std_labels, sigma=None):
        '''
        Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
        @param batch_preds:
        @param batch_std_labels:
        @param sigma:
        @return:
        '''
        # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
        batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

        # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        # batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
        # ensuring S_{ij} \in {-1, 0, 1}
        # batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
        # batch_std_p_ij = 0.5 * (1.0 + batch_Sij)
        batch_S_ij = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
        batch_P_ij = torch.sigmoid(sigma * batch_S_ij)

        return batch_p_ij, batch_P_ij
    
    def ranknet_loss(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        batch_p_ij, batch_std_p_ij = self.get_pairwise_comp_probs_soft(batch_preds=batch_preds, batch_std_labels=batch_std_labels,
                                                             sigma=1.)
        # import ipdb; ipdb.set_trace()
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        # _batch_loss = F.binary_cross_entropy(input=batch_p_ij,
        #                                      target=batch_std_p_ij, reduction='none')
        # import ipdb; ipdb.set_trace()
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

        return batch_loss

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        # batch_q_doc_vectors = batch_q_doc_vectors[:1, :3, :]
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        x1 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device, mix=self.mix)
        x2 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device, mix=self.mix)

        m1 = self.point_sf(x1)
        m2 = self.point_sf(x2)
    
        z1 = self.projector(m1)
        z2 = self.projector(m2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        z1_score = self.scorer(z1).view(-1, num_docs)
        z2_score = self.scorer(z2).view(-1, num_docs)

        p1_score = self.scorer(p1).view(-1, num_docs)
        p2_score = self.scorer(p2).view(-1, num_docs)
        # z1_score = self.scorer(z1.detach()).view(-1, num_docs) 
        # z2_score = self.scorer(z2.detach()).view(-1, num_docs) 

        # p1 = self.predictor(z1).view(-1, num_docs) 
        # p2 = self.predictor(z2).view(-1, num_docs) 

        return p1_score, p2_score, z1_score.detach(), z2_score.detach()

    def adjust_learning_rate(self, optimizer, init_lr, epoch):
        """Decay the learning rate based on schedule"""
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / 150.))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr


    def eval_mode(self):
        self.point_sf.eval()
        self.projector.eval()
        self.predictor.eval()
        self.scorer.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)
        self.projector.train(mode=True)
        self.predictor.train(mode=True)
        self.scorer.train(mode=True)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.point_sf.state_dict(), dir + name)

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.point_sf.load_state_dict(torch.load(file_model, map_location=device))

    def get_tl_af(self):
        return self.sf_para_dict[self.sf_para_dict['sf_id']]['TL_AF']

    def train(self, train_data, epoch_k=None, **kwargs):
        '''
        One epoch training using the entire training data
        '''
        self.train_mode()
        self.adjust_learning_rate(self.optimizer, self.lr, epoch_k)
        assert 'label_type' in kwargs and 'presort' in kwargs
        label_type, presort = kwargs['label_type'], kwargs['presort']
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        batches_processed = 0
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss, stop_training = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k, presort=presort, label_type=label_type)
            
            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()
            batches_processed += 1
        epoch_loss = epoch_loss/num_queries
        return epoch_loss, stop_training

    
    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        p1, p2, z1, z2 = batch_preds
        side1 = self.ranknet_loss(p1, z2)
        side2 = self.ranknet_loss(p2, z1)
        loss = 0.5 * side1 + 0.5 * side2
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

class SimSiamRankParameter(ModelParameter):
    ''' Parameter class for SimRank '''
    def __init__(self, debug=False, para_json=None):
        super(SimSiamRankParameter, self).__init__(model_id='SimSiamRank', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for SimRank
        :return:
        """
        self.para_dict = dict(model_id=self.model_id, aug_percent=0.7, dim=100, aug_type='qg', temp=0.07, mix=0.5)
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

        s1, s2 = (':', '\n') if log else ('_', '_')
        para_str = s1.join(['aug_percent', '{:,g}'.format(para_dict['aug_percent']), 'embed_dim', '{:,g}'.format(para_dict['dim']), 'aug_type', para_dict['aug_type'], 'temp', para_dict['temp'], 'mix', para_dict['mix']])
        return para_str

    def grid_search(self):
        """
        Iterator of parameter settings for simrank
        """
        if self.use_json:
            choice_aug = self.json_dict['aug_percent']
            choice_dim = self.json_dict['dim']
            choice_augtype = self.json_dict['aug_type']
            choice_temp = self.json_dict['temp']
            choice_mix = self.json_dict['mix']
        else:
            choice_aug = [0.3, 0.7] if self.debug else [0.7]  # 1.0, 10.0, 50.0, 100.0
            choice_dim = [50, 100] if self.debug else [100]  # 1.0, 10.0, 50.0, 100.0
            choice_augtype = ['zeroes', 'qg'] if self.debug else ['qg']  # 1.0, 10.0, 50.0, 100.0
            choice_temp = [0.07, 0.1] if self.debug else [0.07] 
            choice_mix = [1., 0.] if self.debug else [1.]


        for aug_percent, dim, augtype, temp, mix in product(choice_aug, choice_dim, choice_augtype, choice_temp, choice_mix):
            self.para_dict = dict(model_id=self.model_id, aug_percent=aug_percent, dim=dim, aug_type=augtype, temp=temp, mix=mix)
            yield self.para_dict
