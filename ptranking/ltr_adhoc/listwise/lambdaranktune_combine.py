#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
"""

import torch
import torch.nn.functional as F
import os
import sys
import numpy as np

from torch.optim.lr_scheduler import StepLR
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.base.utils import get_stacked_FFNet, get_resnet, LTRBatchNorm, ResNetBlock, ResNetOutput
from ptranking.metric.metric_utils import get_delta_ndcg
from ptranking.base.adhoc_ranker import AdhocNeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from torch import nn
from torch.nn.init import xavier_normal_ as nr_init
from torch.nn.init import eye_ as eye_init
from torch.nn.init import zeros_ as zero_init
zeros_default_file = ('/scratch/charlieh/ptranking-results/'
                    'gpu_grid_SimSiam/SimSiam_SF_GE5GE_BN_Affine_Adam'
                    '_1e-06_MSLRWEB30K_MiD_10_MiR_1_TrBat_100_TrPresort'
                    '_EP_10_V_nDCG@5_QS_StandardScaler/aug_percent_0.7_embed_dim_100')

class LambdaRankTune(AdhocNeuralRanker):
    '''
    Christopher J.C. Burges, Robert Ragno, and Quoc Viet Le. 2006.
    Learning to Rank with Nonsmooth Cost Functions. In Proceedings of NIPS conference. 193–200.
    '''
    def __init__(self, sf_para_dict=None, model_para_dict=None, gpu=False, device=None):
        super(LambdaRankTune, self).__init__(id='LambdaRankTune', sf_para_dict=sf_para_dict, gpu=gpu, device=device, weight_decay=1e-2)
        self.sigma = model_para_dict['sigma']
        self.model_load_ckpt = model_para_dict['model_path']
        self.linear_path = model_para_dict['linear_path']
        self.epochs = 0
        self.freeze = model_para_dict['freeze']
        self.probe_layers = model_para_dict['probe_layers']
        self.dropout = model_para_dict['gumbel']
        self.shrink = model_para_dict['shrink']

    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)

        for param in self.simsiam_point_sf.parameters():
            all_params.append(param)
        for param in self.scoring_layer.parameters():
            all_params.append(param)

        
        return nn.ParameterList(all_params)

    def init(self):
        checkpoint_dir = self.model_load_ckpt
        simsiam_path = self.linear_path
        self.point_sf, self.scoring_layer, self.simsiam_point_sf  = self.config_point_neural_scoring_function()
        
        # self.point_sf = self.config_point_neural_scoring_function()
        # if not self.freeze:
        #     nr_hn = nn.Linear(136, 1)
        #     self.point_sf.add_module('_'.join(['ff', 'scoring']), nr_hn)
        #     self.point_sf.to(self.device)
        # else:
        #     for i in range(self.probe_layers - 1):
        #         nr_hn = nn.Linear(136, 136)
        #         self.point_sf.add_module('_'.join(['ff', 'scoring', str(i)]), nr_hn)
        #         self.point_sf.add_module('_'.join(['ff', 'scoring', str(i), 'RELU']), nn.ReLU())
        #     nr_hn = nn.Linear(136, 1)
        #     self.point_sf.add_module('_'.join(['ff', 'scoring', 'final']), nr_hn)
        #     self.point_sf.to(self.device)

        shrink = self.shrink
        self.config_optimizer()
        if len(checkpoint_dir) > 0:
            print('Loading checkpoint...', file=sys.stderr)
            print(os.path.join(checkpoint_dir, 'net_params_pretrain.pkl'), file=sys.stderr)
            checkpoint_file_name = os.path.join(checkpoint_dir, f'net_params_best_0_3_1_0.0_{shrink}.pkl')
            print(checkpoint_file_name)
            pretrained_dict = torch.load(checkpoint_file_name, map_location=self.device)
            curr_dict = self.point_sf.state_dict()
            for key in curr_dict.keys():
                curr_dict[key] = pretrained_dict[key]
            self.point_sf.load_state_dict(curr_dict)

        else:
            print('No checkpoint', file=sys.stderr)

        if len(simsiam_path) > 0:
            print('Loading simsiam...', file=sys.stderr)
            checkpoint_file_name = os.path.join(simsiam_path, f'net_params_best_0_3_1_0.0_{shrink}.pkl')
            print(checkpoint_file_name)
            pretrained_dict = torch.load(checkpoint_file_name, map_location=self.device)
            curr_dict = self.simsiam_point_sf.state_dict()
            for key in curr_dict.keys():
                curr_dict[key] = pretrained_dict[key]
            self.simsiam_point_sf.load_state_dict(curr_dict)
        else:
            print('No simsiam')
        # if len(checkpoint_dir) > 0:
        #     print('Loading checkpoint...', file=sys.stderr)
        #     print(os.path.join(checkpoint_dir, 'net_params_pretrain'), file=sys.stderr)
        #     checkpoint_file_name = os.path.join(checkpoint_dir, 'net_params_pretrain.pkl')
        #     pretrained_dict = torch.load(checkpoint_file_name, map_location=self.device)
        #     pointsf_dict = self.point_sf.state_dict()
        #     pointsf_dict.update(pretrained_dict)
        #     self.point_sf.load_state_dict(pointsf_dict)
        # else:
        #     print('No checkpoint', file=sys.stderr)
        # self.embedder.train(mode=True)
        # if self.freeze:
        #     print('FROZEN')
        #     self.embedder.eval()
        #     for name, param in self.embedder.named_parameters():
        #         param.requires_grad = False
        # if self.freeze:
        #     print('FROZEN with ', self.probe_layers)
        #     for name, param in self.point_sf.named_parameters():
        #         if 'ff_scoring' not in name:
        #             param.requires_grad = False
        print(self.point_sf)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=40, gamma=1.)

    def config_point_neural_scoring_function(self):
        point_sf, scoring_layer, simsiam_point_sf = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: 
            point_sf = point_sf.to(self.device)
            simsiam_point_sf = simsiam_point_sf.to(self.device)
            scoring_layer = scoring_layer.to(self.device)

        return point_sf, scoring_layer, simsiam_point_sf

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
        h_dim = 136
        point_sf = get_resnet(num_features, h_dim)

        # point_sf = get_stacked_FFNet(ff_dims=ff_dims, AF=AF, TL_AF=TL_AF, apply_tl_af=apply_tl_af, dropout=dropout,
        #                              BN=BN, bn_type=bn_type, bn_affine=bn_affine, device=self.device)
        if 'SimCLR' in self.model_load_ckpt:
            point_sf.add_module('_'.join(['project', 'linear', str(0)]), nn.Linear(h_dim, h_dim))
            point_sf.add_module('_'.join(['project', 'relu', str(0)]), nn.ReLU())
        
        scoring_adapter = nn.Sequential()
        for i in range(self.probe_layers - 1):
            nr_block =  nn.Linear(h_dim, h_dim)
            scoring_adapter.add_module('_'.join(['ff', 'scoring', str(i + 1)]), nr_block)
            scoring_adapter.add_module('_'.join(['ff', 'scoring', 'act', str(i+1)]), nn.ReLU())
        # scoring_layer = nn.Linear(h_dim, 1)
        # scoring_adapter.add_module('scoring_layer', scoring_layer)

        point_sf.add_module('scoring_adapter', scoring_adapter)



        simsiam_point_sf = get_resnet(num_features, h_dim)

        scoring_adapter = nn.Sequential()
        for i in range(self.probe_layers - 1):
            nr_block =  nn.Linear(h_dim, h_dim)
            scoring_adapter.add_module('_'.join(['ff', 'scoring', str(i + 1)]), nr_block)
            scoring_adapter.add_module('_'.join(['ff', 'scoring', 'act', str(i+1)]), nn.ReLU())
        # scoring_adapter.add_module('scoring_layer', scoring_layer)
        
        # scoring_adapter.add_module('scoring_layer', scoring_layer)
        simsiam_point_sf.add_module('scoring_adapter', scoring_adapter)
        
        # for simulated binary labels--do not linear probe
        scoring_layer = nn.Sequential(
            LTRBatchNorm(2*h_dim, affine=False),
            nn.Dropout(self.dropout),
            nn.Linear(2 * h_dim, 1),
            # nn.ReLU(),
            # nn.Linear(h_dim, 1)
        )

        # for simulated crowdsourcing--linear probe for 100 epochs
        # scoring_layer = nn.Sequential(
        #     LTRBatchNorm(2 * h_dim, affine=False),
        #     nn.Dropout(0.2),
        #     nn.Linear(2 * h_dim, 1)
        # )
        print(scoring_layer)
        # point_sf.add_module('scoring_adapter', scoring_adapter)
        point_sf.to(self.device)
        scoring_adapter.to(self.device) 
        simsiam_point_sf.to(self.device)

        return point_sf, scoring_layer, simsiam_point_sf, 
    def forward(self, batch_q_doc_vectors):
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        simclr_output = self.point_sf(batch_q_doc_vectors)
        # simclr_output = nn.functional.normalize(simclr_output, dim=2)
        simsiam_output = self.simsiam_point_sf(batch_q_doc_vectors)
        # simsiam_output = nn.functional.normalize(simsiam_output, dim=2)
        input_to_score = torch.cat([simclr_output, simsiam_output], dim=2)
        _batch_preds = self.scoring_layer(input_to_score)
        batch_preds = _batch_preds.view(-1, num_docs) 
        return batch_preds

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
        # torch.nn.utils.clip_grad_norm_(self.point_sf.parameters(), 1.0)
        self.optimizer.step()
        

        return batch_loss
    # def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
    #     '''
    #     @param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
    #     @param batch_std_labels: [batch, ranking_size] each row represents the standard relevance grades for documents associated with the same query
    #     @param kwargs:
    #     @return:
    #     '''
    #     batch_p_ij, batch_std_p_ij = get_pairwise_comp_probs(batch_preds=batch_preds, batch_std_labels=batch_std_labels,
    #                                                          sigma=self.sigma)
    #     _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
    #                                          target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
    #     batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))

    #     self.optimizer.zero_grad()
    #     batch_loss.backward()
    #     self.optimizer.step()

    #     return batch_loss
    def train_mode(self):
        self.point_sf.train(mode=True)
        self.simsiam_point_sf.train(mode=True)
        self.scoring_layer.train(mode=True)
    
    def eval_mode(self):
        self.point_sf.eval()
        self.simsiam_point_sf.eval()
        self.scoring_layer.eval()


    def save(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.point_sf.state_dict(), dir + name + '.pkl')
        torch.save(self.simsiam_point_sf.state_dict(), dir + name + 'simsiam' + '.pkl')
        torch.save(self.scoring_layer.state_dict(), dir + name + 'scoring_layer' + '.pkl')


    def load(self, file_model, **kwargs):
        device = kwargs['device']
        self.point_sf.load_state_dict(torch.load(file_model + '.pkl', map_location=device))
        self.simsiam_point_sf.load_state_dict(torch.load(file_model + 'simsiam' + '.pkl', map_location=device))
        self.scoring_layer.load_state_dict(torch.load(file_model + 'scoring_layer' +'.pkl', map_location=device))
    
    def train(self, train_data, epoch_k=None, **kwargs):
        '''
        One epoch training using the entire training data
        '''
        self.train_mode()
        if not self.freeze:
            if self.epochs < 5:
                # self.point_sf.scoring_adapter.train(mode=True)
                for name, param in self.point_sf.named_parameters():
                    param.requires_grad = False
                for name, param in self.simsiam_point_sf.named_parameters():
                    param.requires_grad = False

                for name, param in self.scoring_layer.named_parameters():
                    param.requires_grad = True
            else:
                for name, param in self.point_sf.named_parameters():
                    param.requires_grad = True
                for name, param in self.simsiam_point_sf.named_parameters():
                    param.requires_grad = True
                for name, param in self.scoring_layer.named_parameters():
                    param.requires_grad = True
        else:
            self.point_sf.eval()
            self.point_sf.scoring_adapter.train(mode=True)
            for name, param in self.point_sf.named_parameters():
                param.requires_grad = False

                
        assert 'label_type' in kwargs and 'presort' in kwargs
        label_type, presort = kwargs['label_type'], kwargs['presort']
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        batches_processed = 0
        # self.optimizer.zero_grad()
        
        
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss, stop_training = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k, presort=presort, label_type=label_type)
            # loss = batch_loss / float(size_of_train)
            # loss.backward()
            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()
            batches_processed += 1

        # torch.nn.utils.clip_grad_norm_(self.point_sf.parameters(), 1.0)
        # self.optimizer.step()
        epoch_loss = epoch_loss/num_queries
        self.epochs += 1
        return epoch_loss, stop_training

###### Parameter of LambdaRank ######

class LambdaRankTuneParameter(ModelParameter):
    ''' Parameter class for LambdaRank '''
    def __init__(self, debug=False, para_json=None):
        super(LambdaRankTuneParameter, self).__init__(model_id='LambdaRankTune', para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for LambdaRank
        :return:
        """
        self.lambda_para_dict = dict(model_id=self.model_id, sigma=1.0, model_path=zeros_default_file, linear_path='', freeze=False, probe_layers=1)
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
        pretrain_set = lambda_para_dict['model_path'].split('/')[-1]
        s1, s2 = (':', '\n') if log else ('_', '_')
        lambdarank_para_str = s1.join(['Sigma', '{:,g}'.format(lambda_para_dict['sigma']), 'pretrain_set', pretrain_set])
        return lambdarank_para_str

    def grid_search(self):
        """
        Iterator of parameter settings for LambdaRank
        """
        if self.use_json:
            choice_sigma = self.json_dict['sigma']
            choice_pretrains = self.json_dict['model_path']
            choice_linear = self.json_dict['linear_path']
            choice_freeze = self.json_dict['freeze']
            choice_probe_layers = self.json_dict['probe_layers']
        else:
            choice_sigma = [5.0, 1.0] if self.debug else [1.0]  # 1.0, 10.0, 50.0, 100.0

        for sigma in choice_sigma:
            self.lambda_para_dict = dict(model_id=self.model_id, sigma=sigma, model_path=choice_pretrains[0], linear_path=choice_linear[0], freeze=choice_freeze[0], probe_layers = choice_probe_layers[0])
            yield self.lambda_para_dict
