"""
Row-wise simsiam pretraining
"""

import torch
import torch.nn as nn
import os
import sys
import math
import time
from itertools import product
from ptranking.base.utils import get_stacked_FFNet, get_resnet, LTRBatchNorm
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap, gaussian
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from absl import logging
import torch.optim as optim

import torch.nn.functional as F
from ptranking.metric.metric_utils import get_delta_ndcg
from torch.nn.init import eye_ as eye_init
from torch.nn.init import zeros_ as zero_init

class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = LTRBatchNorm(in_dim, momentum=0.1, affine=True, track_running_stats=False)
        self.ff1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.ff2 = nn.Linear(in_dim, in_dim)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.ff1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.ff2(out)
        out = self.dropout2(out)

        out += identity
        return out
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
        self.sigma = model_para_dict['blend']
        self.scale = model_para_dict['scale']
        self.gumbel = model_para_dict['gumbel']
        self.num_negatives = model_para_dict['num_negatives']
        self.epochs= 0
        self.loss = nn.CosineSimilarity(dim=1).to(self.device)

        if self.aug_type == 'zeroes':
            self.augmentation = zeroes
        elif self.aug_type == 'qg':
            self.augmentation = qgswap
        elif self.aug_type == 'gaussian':
            self.augmentation = gaussian

    def get_resnet(self, data_dim, hidden_dim=130, dropout=0.1):
        ff_net = nn.Sequential()
        n_init = nn.Linear(data_dim, hidden_dim, bias=False)
        ff_net.add_module('_'.join(['input_mapping']), n_init)

        num_layers = 3
        for i in range(num_layers):
            nr_block = ResNetBlock(hidden_dim)
            ff_net.add_module('_'.join(['resnet', str(i + 1)]), nr_block)
        ff_net.add_module('_'.join(['bn_resnet']), LTRBatchNorm(hidden_dim, momentum=0.1, affine=True, track_running_stats=False))
        ff_net.add_module('_'.join(['bn_act']), nn.ReLU())
        return ff_net
    

    def init(self):
        self.point_sf, self.projector, self.anti_score = self.config_point_neural_scoring_function()
        self.loss_no_reduction = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.config_optimizer()
        self.anti_optimizer = optim.Adam(self.anti_score.parameters(), lr = 5.0 * self.lr, weight_decay = self.weight_decay)



    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.projector.parameters():
            all_params.append(param)
        return nn.ParameterList(all_params)

    def config_point_neural_scoring_function(self):
        point_sf, projector, anti_score = self.ini_pointsf(
            **self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: 
            point_sf = point_sf.to(self.device)
            projector = projector.to(self.device)
            anti_score = anti_score.to(self.device)
        return point_sf, projector, anti_score

    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=136, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                    BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        h_dim = 136
        point_sf = self.get_resnet(num_features, h_dim)
        projector = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )
        print('Running PAIRCON.......')
        anti_score = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        return point_sf, projector, anti_score
    def get_pairwise_comp_probs(self, batch_preds, batch_std_labels, sigma=None):
        '''
        Get the predicted and standard probabilities p_ij which denotes d_i beats d_j
        @param batch_preds:
        @param batch_std_labels:
        @param sigma:
        @return:
        '''
        # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        # [batch_size, num_docs, 1], [batch_size, 1, num_docs]
        batch_s_ij = torch.unsqueeze(batch_preds, dim=2) - torch.unsqueeze(batch_preds, dim=1)
        batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

        # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        batch_std_diffs = torch.unsqueeze(batch_std_labels, dim=2) - torch.unsqueeze(batch_std_labels, dim=1)
        batch_std_p_ij = torch.sigmoid(sigma * batch_std_diffs)
        probs_reshaped = torch.stack([1-batch_std_p_ij, batch_std_p_ij], dim=-1)
        tau = 0.1  # Temperature parameter; adjust as needed
        hard_samples = F.gumbel_softmax(probs_reshaped, tau=tau, hard=True, dim=-1)
        hard_labels = hard_samples[..., 1]
        return batch_p_ij, hard_labels

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        data_dim, num_docs, num_features = batch_q_doc_vectors.shape

        scores = self.projector(self.point_sf(batch_q_doc_vectors)).reshape((data_dim, num_docs))
        labels = self.anti_score(batch_q_doc_vectors).reshape((data_dim, num_docs))

        return scores, labels


    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        
        '''

        scores, labels = batch_preds
        batch_p_ij, batch_std_p_ij = self.get_pairwise_comp_probs(batch_preds=scores, batch_std_labels=labels, sigma=1.0)
        print(batch_std_p_ij[0,0,:])
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                        target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        batch_loss = torch.sum(torch.sum(_batch_loss, dim=(2, 1)))
        self.optimizer.zero_grad()
        batch_loss.backward(retain_graph=True)
        

        self.anti_optimizer.zero_grad()
        (-batch_loss).backward()
        self.optimizer.step()
        self.anti_optimizer.step()
        return batch_loss

    def eval_mode(self):
        self.point_sf.eval()
        self.projector.eval()
        self.anti_score.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)
        self.projector.train(mode=True)
        self.anti_score.train(mode=True)


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

    def train(self, train_data, epoch_k=None, **kwargs):
        '''
        One epoch training using the entire training data
        '''
        self.train_mode()
        assert 'label_type' in kwargs and 'presort' in kwargs
        label_type, presort = kwargs['label_type'], kwargs['presort']
        num_queries = 0
        epoch_loss = torch.tensor([0.0], device=self.device)
        batches_processed = 0
        # if self.epochs < 10:
        #     for name, param in self.point_sf.named_parameters():
        #         param.requires_grad = False
        #     for name, param in self.projector.named_parameters():
        #         param.requires_grad = False
        #     for name, param in self.anti_score.named_parameters():
        #         param.requires_grad = True
        # else:
        #     for name, param in self.point_sf.named_parameters():
        #         param.requires_grad = True
        #     for name, param in self.projector.named_parameters():
        #         param.requires_grad = True
        #     for name, param in self.anti_score.named_parameters():
        #         param.requires_grad = False
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss, stop_training = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k, presort=presort, label_type=label_type)

            
            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()
            batches_processed += 1

        self.epochs += 1
        epoch_loss = epoch_loss/batches_processed
        return epoch_loss, stop_training
    
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
                              gumbel=1.0,
                              num_negatives=100)
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
            para_dict['scale'], 'gumbel', para_dict['gumbel'],
            para_dict['num_negatives'], 'num_negatives'
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
            choice_negatives = self.json_dict['num_negatives']
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
            choice_gumbel = [1., 0.1] if self.debug else [1.]
            choice_negatives = [100, 10] if self.debug else [1]

        for aug_percent, dim, augtype, temp, mix, blend, scale, gumbel, num_negatives in product(
                choice_aug, choice_dim, choice_augtype, choice_temp,
                choice_mix, choice_blend, choice_scale, choice_gumbel, choice_negatives):
            self.para_dict = dict(model_id=self.model_id,
                                  aug_percent=aug_percent,
                                  dim=dim,
                                  aug_type=augtype,
                                  temp=temp,
                                  mix=mix,
                                  blend=blend,
                                  scale=scale,
                                  gumbel=gumbel,
                                  num_negatives=num_negatives)
            yield self.para_dict