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
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap, qg_and_zero
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from absl import logging
import torch.nn.functional as F



class RankNeg(NeuralRanker):

    def __init__(self,
                 id='RankNegPretrainer',
                 sf_para_dict=None,
                 model_para_dict=None,
                 weight_decay=1e-3,
                 gpu=False,
                 device=None):
        super(RankNeg, self).__init__(id=id,
                                      sf_para_dict=sf_para_dict,
                                      weight_decay=weight_decay,
                                      gpu=gpu,
                                      device=device)
        self.aug_percent = model_para_dict['aug_percent']
        self.dim = model_para_dict['dim']
        self.aug_type = model_para_dict['aug_type']
        self.temperature = model_para_dict['temp']
        self.mix = model_para_dict['mix']
        self.blend = model_para_dict['blend']
        self.scale = model_para_dict['scale']
        self.gumbel = model_para_dict['gumbel']
        self.num_negatives = 100
        if self.aug_type == 'zeroes':
            self.augmentation = zeroes
        elif self.aug_type == 'qg':
            self.augmentation = qgswap
        elif self.aug_type == 'qz':
            self.augmentation = qg_and_zero

    def init(self):
        self.point_sf = self.config_point_neural_scoring_function()
        self.projector, self.predictor, self.rankneg_proj, self.rankneg_scorer = self.config_heads()
        self.xent_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.loss = nn.CosineSimilarity(dim=1).to(self.device)

        self.config_optimizer()

    def config_point_neural_scoring_function(self):
        point_sf = self.ini_pointsf(
            **self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: point_sf = point_sf.to(self.device)
        return point_sf

    def config_heads(self):
        dim = self.dim
        prev_dim = -1
        for name, param in self.point_sf.named_parameters():
            if 'ff' in name and 'bias' not in name:
                prev_dim = param.shape[0]
        projector = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim, affine=False),
            nn.ReLU(inplace=True),  # second layer
            nn.Linear(prev_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False))
        if self.gpu: projector = projector.to(self.device)

        pred_dim = int(dim // 4)

        predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim))  # output layer
        if self.gpu: predictor = predictor.to(self.device)

        simclr_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim),
                        nn.ReLU(), # first layer
                        nn.Linear(prev_dim, dim),
                        nn.ReLU())
        if self.gpu: simclr_projector = simclr_projector.to(self.device)

        scorer = nn.Linear(dim, 1)
        if self.gpu: scorer = scorer.to(self.device)

        return projector, predictor, simclr_projector, scorer

    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.projector.parameters():
            all_params.append(param)
        for param in self.predictor.parameters():
            all_params.append(param)
        for param in self.rankneg_proj.parameters():
            all_params.append(param)
        for param in self.rankneg_scorer.parameters():
            all_params.append(param)

        return nn.ParameterList(all_params)

    def ini_pointsf(self,
                    num_features=None,
                    h_dim=100,
                    out_dim=136,
                    num_layers=3,
                    AF='R',
                    TL_AF='S',
                    apply_tl_af=False,
                    BN=True,
                    bn_type=None,
                    bn_affine=False,
                    dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        encoder_layers = num_layers
        ff_dims = [num_features]
        for i in range(encoder_layers):
            ff_dims.append(h_dim)
        ff_dims.append(out_dim)

        point_sf = get_stacked_FFNet(ff_dims=ff_dims,
                                     AF=AF,
                                     TL_AF=TL_AF,
                                     apply_tl_af=apply_tl_af,
                                     dropout=dropout,
                                     BN=BN,
                                     bn_type=bn_type,
                                     bn_affine=bn_affine,
                                     device=self.device)
        return point_sf

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        batch_q_doc_vectors = batch_q_doc_vectors[:1,:2,:]
        p1, p2, z1, z2 = self.simsiam_forward(batch_q_doc_vectors)
        target_scores, ssldata_scores = self.rankneg_forward(batch_q_doc_vectors)
        return target_scores, ssldata_scores, p1, p2, z1, z2

    def simsiam_forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''

        x1 = self.augmentation(batch_q_doc_vectors, 0.95,
                               self.device)
        x2 = self.augmentation(batch_q_doc_vectors, 0.95,
                               self.device)

        data_dim = batch_q_doc_vectors.shape[2]
        x1_flat = x1.reshape((-1, data_dim))
        x2_flat = x2.reshape((-1, data_dim))
        mod1 = self.point_sf(x1_flat)
        mod2 = self.point_sf(x2_flat)
        z1 = self.projector(mod1)
        z2 = self.projector(mod2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()

    def rankneg_forward(self, batch_q_doc_vectors):
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        target = batch_q_doc_vectors
        ssldata = self.augmentation(batch_q_doc_vectors,
                               self.aug_percent,
                               self.device,
                               mix=self.mix,
                               scale=self.scale)
        # embeddings
        target_embed = self.point_sf(target)
        ssldata_embed = self.point_sf(ssldata)

        # rankneg forward
        target_z = self.rankneg_proj(target_embed)
        ssldata_z = self.rankneg_proj(ssldata_embed)
        _target_scores = self.rankneg_scorer(target_z)
        _ssldata_scores = self.rankneg_scorer(ssldata_z)

        target_scores = _target_scores.view(-1, num_docs)  # [batch_size, num_docs]
        ssldata_scores = _ssldata_scores.view(-1, num_docs)  # [batch_size, num_docs]

        return target_scores, ssldata_scores


    def eval_mode(self):
        self.point_sf.eval()
        self.projector.eval()
        self.predictor.eval()
        self.rankneg_proj.eval()
        self.rankneg_scorer.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)
        self.projector.train(mode=True)
        self.predictor.train(mode=True)
        self.rankneg_proj.train(mode=True)
        self.rankneg_scorer.train(mode=True)

    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.point_sf.state_dict(), dir + name)

    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.point_sf.load_state_dict(
            torch.load(file_model, map_location=device))

    def get_tl_af(self):
        return self.sf_para_dict[self.sf_para_dict['sf_id']]['TL_AF']

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        target_scores, source_scores, p1, p2, z1, z2 = batch_preds
        simsiam_loss = -(self.loss(p1, z2).mean() + self.loss(p2, z1).mean()) * 0.5
        rankneg_loss = self.rankneg_loss(target_scores, source_scores)
        loss = self.blend * rankneg_loss + (1. - self.blend) * simsiam_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def rankneg_loss(self, target_scores, source_scores):
        '''
        target scores: [batch_size, num_docs]
        source scores: [batch_size, num_docs]
        '''

        batch_size = target_scores.shape[0]
        num_scores = target_scores.shape[1]

        target_scores_expanded = target_scores[:, None, :].expand((batch_size, self.num_negatives, num_scores))
        gumbels = -torch.empty_like(target_scores_expanded).exponential_().log()
        sampled_negatives = target_scores_expanded + self.gumbel * gumbels


        # ranknet similarity scores
        preds = source_scores[:,None,:].expand(batch_size, self.num_negatives + 1, num_scores).to(self.device)
        all_targets = torch.cat([target_scores[:,None,:], sampled_negatives], dim=1).to(self.device)
        logits = -self.ranknet_loss_mat(preds, all_targets.detach())

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature

        loss = self.xent_loss(logits, labels)

        return loss

    def ranknet_loss_mat(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, batch, anking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        batch_p_ij, batch_std_p_ij = self.get_pairwise_comp_probs_hard_mat(batch_preds=batch_preds, batch_std_labels=batch_std_labels,
                                                             sigma=1.)

        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')

        batch_loss = torch.sum(_batch_loss, dim=(3, 2))

        return batch_loss        

    def get_pairwise_comp_probs_hard_mat(self, batch_preds, batch_std_labels, sigma=None):
        '''
        Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
        @param batch_preds:
        @param batch_std_labels:
        @param sigma:
        @return:
        '''
        # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        batch_s_ij = torch.unsqueeze(batch_preds, dim=3) - torch.unsqueeze(batch_preds, dim=2)
        batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

        # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=3) - torch.unsqueeze(batch_std_labels, dim=2)
        # ensuring S_{ij} \in {-1, 0, 1}
        # batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
        # batch_std_p_ij = 0.5 * (1.0 + batch_Sij)
        batch_std_p_ij_pos = torch.where(batch_std_diffs > 0., 1., 0.)
        batch_std_p_ij_neg = torch.where(batch_std_diffs < 0., -1., 0.)
        batch_std_p_ij = batch_std_p_ij_pos + batch_std_p_ij_neg
        return batch_p_ij, batch_std_p_ij



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

        return self.custom_loss_function(batch_preds, batch_std_labels,
                                         **kwargs), stop_training

    def validation(self,
                   vali_data=None,
                   vali_metric=None,
                   k=5,
                   presort=False,
                   max_label=None,
                   label_type=LABEL_TYPE.MultiLabel,
                   device='cpu'):
        self.eval_mode()  # switch evaluation mode

        num_queries = 0
        sum_val_loss = torch.zeros(1).to(self.device)
        for batch_ids, batch_q_doc_vectors, batch_std_labels in vali_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            if batch_std_labels.size(1) < k:
                continue  # skip if the number of documents is smaller than k
            else:
                num_queries += len(batch_ids)

            if self.gpu:
                batch_q_doc_vectors = batch_q_doc_vectors.to(self.device)
            batch_preds = self.predict(batch_q_doc_vectors)
            val_loss = self.custom_loss_function(batch_preds, batch_std_labels)

            sum_val_loss += val_loss  # due to batch processing

        avg_val_loss = val_loss / num_queries
        return avg_val_loss.cpu()

    def adhoc_performance_at_ks(self,
                                test_data=None,
                                ks=[1, 5, 10],
                                label_type=LABEL_TYPE.MultiLabel,
                                max_label=None,
                                presort=False,
                                device='cpu',
                                need_per_q=False):
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


class RankNegParameter(ModelParameter):
    ''' Parameter class for SimRank '''

    def __init__(self, debug=False, para_json=None):
        super(RankNegParameter, self).__init__(model_id='RankNeg',
                                               para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for SimRank
        :return:
        """
        self.para_dict = dict(model_id=self.model_id,
                              aug_percent=0.7,
                              dim=100,
                              aug_type='qg',
                              temp=0.07,
                              mix=0.5,
                              blend=0.5,
                              scale=0.01,
                              gumbel=1.0)
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
        para_str = s1.join([
            'aug_percent', '{:,g}'.format(para_dict['aug_percent']),
            'embed_dim', '{:,g}'.format(para_dict['dim']), 'aug_type',
            para_dict['aug_type'], 'temp', para_dict['temp'], 'mix',
            para_dict['mix'], 'blend', para_dict['blend'], 'scale',
            para_dict['scale'], 'gumbel', para_dict['gumbel']
        ])
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
            choice_blend = self.json_dict['blend']
            choice_scale = self.json_dict['scale']
            choice_gumbel = self.json_dict['gumbel']
        else:
            choice_aug = [0.3, 0.7
                          ] if self.debug else [0.7]  # 1.0, 10.0, 50.0, 100.0
            choice_dim = [50, 100
                          ] if self.debug else [100]  # 1.0, 10.0, 50.0, 100.0
            choice_augtype = ['zeroes', 'qg'] if self.debug else [
                'qg'
            ]  # 1.0, 10.0, 50.0, 100.0
            choice_temp = [0.07, 0.1] if self.debug else [0.07]
            choice_mix = [1., 0.] if self.debug else [1.]
            choice_blend = [1., 0.] if self.debug else [1.]
            choice_scale = [1., 0.] if self.debug else [1.]
            choice_gumbel = [1., 0.1] if self.debug else[1.]

        for aug_percent, dim, augtype, temp, mix, blend, scale, gumbel in product(
                choice_aug, choice_dim, choice_augtype, choice_temp,
                choice_mix, choice_blend, choice_scale, choice_gumbel):
            self.para_dict = dict(model_id=self.model_id,
                                  aug_percent=aug_percent,
                                  dim=dim,
                                  aug_type=augtype,
                                  temp=temp,
                                  mix=mix,
                                  blend=blend,
                                  scale=scale,
                                  gumbel=gumbel)
            yield self.para_dict