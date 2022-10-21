"""
Row-wise simsiam pretraining
"""

import torch
import torch.nn as nn
import os
from ptranking.base.utils import get_stacked_FFNet
from ptranking.base.ranker import NeuralRanker
from augmentations import zeroes
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
    def __init__(self, id='SimSiamPretrainer', sf_para_dict=None, weight_decay=1e-3, gpu=False, device=None):
        super(SimSiam, self).__init__(id=id, sf_para_dict=sf_para_dict, weight_decay=weight_decay, gpu=gpu, device=device)

    def init(self):
        self.point_sf = self.config_point_neural_scoring_function()
        self.projector = self.config_projector()
        self.predictor = self.config_predictor()
        self.aug_percent = self.sf_para_dict[self.sf_para_dict['sf_id']]['aug_percent']
        self.config_optimizer()

    def config_point_neural_scoring_function(self):
        point_sf = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: point_sf = point_sf.to(self.device)
        return point_sf

    def config_projector(self):
        dim = self.sf_para_dict[self.sf_para_dict['sf_id']]['out_dim']
        prev_dim = self.sf_para_dict[self.sf_para_dict['sf_id']]['out_dim']
        projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim),
                                  nn.ReLU(inplace=True), # first layer
                                  nn.Linear(prev_dim, prev_dim, bias=False),
                                  nn.BatchNorm1d(prev_dim),
                                  nn.ReLU(inplace=True), # second layer
                                  nn.Linear(prev_dim, dim, bias=False),
                                  nn.BatchNorm1d(dim, affine=False))
        if self.gpu: projector = projector.to(self.device)
        return projector
    
    def config_predictor(self):
        dim = self.sf_para_dict[self.sf_para_dict['sf_id']]['out_dim']
        pred_dim = dim / 4
        predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                    nn.BatchNorm1d(pred_dim),
                                    nn.ReLU(inplace=True), # hidden layer
                                    nn.Linear(pred_dim, dim)) # output layer
        if self.gpu: predictor = predictor.to(self.device)
        return predictor

    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.projector.parameters():
            all_params.append(param)
        for param in self.predictor.parameters():
            all_params.append(param)
        
        return all_params

    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=1, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                    BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        ff_dims = [num_features]
        for i in range(num_layers):
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
        x1 = zeroes(batch_q_doc_vectors)
        x2 = zeroes(batch_q_doc_vectors)

        z1 = self.projector(self.point_sf(x1))
        z2 = self.projector(self.point_sf(x2))

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
