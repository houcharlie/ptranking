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
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap, gaussian, categorical_augment
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from absl import logging
import torch.nn.functional as F
from ptranking.metric.metric_utils import get_delta_ndcg
from torch.nn.init import eye_ as eye_init
from torch.nn.init import zeros_ as zero_init
from ptranking.data.binary_features import mslr_binary_features, yahoo_binary_features, istella_binary_features


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
        self.epochs_done = 0
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
        self.point_sf, self.projector = self.config_point_neural_scoring_function()
        self.loss_no_reduction = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.config_optimizer()



    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.projector.parameters():
            all_params.append(param)
        return nn.ParameterList(all_params)

    def config_point_neural_scoring_function(self):
        point_sf, projector = self.ini_pointsf(
            **self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: 
            point_sf = point_sf.to(self.device)
            projector = projector.to(self.device)
        return point_sf, projector

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
        self.categorical_features = categorical_features
        h_dim = 136
        point_sf = self.get_resnet(num_features, 136)
        projector = nn.Sequential(
            nn.Linear(h_dim, 1)
        )
        print('Running PAIRCON.......')

        return point_sf, projector
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

        return batch_p_ij, batch_std_p_ij
    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        data_dim, num_docs, num_features = batch_q_doc_vectors.shape

        x1 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device, self.categorical_features)
        x1 = categorical_augment(x1, self.aug_percent, self.device, self.categorical_features)
        x2 = self.augmentation(batch_q_doc_vectors, self.aug_percent, self.device, self.categorical_features)
        x2 = categorical_augment(x2, self.aug_percent, self.device, self.categorical_features)

        embed1 = self.point_sf(x1)
        embed2 = self.point_sf(x2)
        z1 = self.projector(embed1)
        z2 = self.projector(embed2)

        s1 = z1.view(data_dim, num_docs)
        s2 = z2.view(data_dim, num_docs)

        s_concat = torch.cat((s1, s2), dim=1)

        logits_qg, labels_qg = self.paircon_loss(s_concat)

        return logits_qg, labels_qg
    
    def paircon_loss(self, s_concat):
        '''
        s_concat: [batch_size, 2 * qgsize]
        '''
        batch_size = s_concat.shape[0]
        qg_size = s_concat.shape[1]//2
        # [batch_size, 2 * qg_size, 2 * qg_size]
        batch_s_ij = torch.unsqueeze(s_concat, dim=2) - torch.unsqueeze(s_concat, dim=1)
        # [batch_size, 2 * qg_size, 2 * qg_size]
        batch_p_ij = torch.sigmoid(self.sigma * batch_s_ij)


        # i1, i2 => f(i1), f(i2) (dim d) => scorer(f(i1) concat f(i2)) (scorer is like 3 layers) => 1 dimensional score 
        # scorer(f(i1) concat f(i2)) (end of second layer output): embed_i1i2
        # 
        # Expanding dimensions for broadcasting
        # batch_p_ij_expanded_ij repeats along the 'k' axis
        # batch_p_ij_expanded_ik repeats along the 'j' axis
        # This prepares both tensors for element-wise BCE computation

        batch_p_ij_expanded_ij = batch_p_ij.unsqueeze(3).expand(-1, -1, -1, 2 * qg_size)
        batch_p_ij_expanded_ik = batch_p_ij.unsqueeze(2).expand(-1, -1, 2 * qg_size, -1)
        
        # Asymmetry could avoid mode collapse
        # s, s', plug into ranking loss, minimize only one end

        # Now compute the binary cross entropy
        # Note: F.binary_cross_entropy expects input in the form of (input, target),
        # so we treat one of the expanded tensors as input and the other as target.
        # The `reduction='none'` argument ensures we keep the full output dimensionality.
        # The higher, the more similar.
        # if self.num_negatives == 0:
        #     epsilon = 1e-12
        #     p1 = batch_p_ij_expanded_ij
        #     p2 = batch_p_ij_expanded_ik
        #     p1 = p1.clamp(min=epsilon, max=1-epsilon)
        #     p2 = p2.clamp(min=epsilon, max=1-epsilon)
        #     kl_divergence_p1_p2 = p1 * torch.log(p1 / p2) + (1 - p1) * torch.log((1 - p1) / (1 - p2))

        #     # Compute KL divergence from p2 to p1
        #     kl_divergence_p2_p1 = p2 * torch.log(p2 / p1) + (1 - p2) * torch.log((1 - p2) / (1 - p1))

        #     # Compute symmetric KL divergence by averaging the two KL divergences
        #     symmetric_kl_divergence = (kl_divergence_p1_p2 + kl_divergence_p2_p1) / 2
        #     similarity_matrix = -symmetric_kl_divergence
        # elif self.num_negatives == 1:
        similarity_matrix = -F.mse_loss(batch_p_ij_expanded_ij, batch_p_ij_expanded_ik, reduction='none')

        labels = torch.cat([torch.arange(qg_size) for i in range(2)], dim=0)
        # [2 x qgsize, 2 x qgsize] 4 identity matrices in each quadrant
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        # [batchsize, 2 x qgsize, 2 x qgsize]
        labels = labels[None, None, :, :].expand(batch_size, 2 * qg_size, -1, -1)

        mask = torch.eye(labels.shape[2], dtype=torch.bool).to(self.device)[None, None, :].expand(batch_size, 2 * qg_size, -1, -1)
        # erase self similarities
        labels = labels[~mask].view(batch_size, 2 * qg_size, 2 * qg_size, -1).to(self.device)
        similarity_matrix = similarity_matrix[~mask].view(batch_size, 2 * qg_size, 2 * qg_size, -1).to(self.device)

        # get augmented pair similarities
        positives = similarity_matrix[labels.bool()].view(batch_size, 2 * qg_size, 2 * qg_size, -1).to(self.device)
        # get the non augmented pair similarities
        negatives = similarity_matrix[~labels.bool()].view(batch_size, 2 * qg_size, 2 * qg_size, -1).to(self.device)
        
        logits = torch.cat([positives, negatives], dim=3)
        labels = torch.zeros((logits.shape[0], logits.shape[1], logits.shape[2]), dtype=torch.long).to(self.device)

        logits = logits / self.temperature


        return logits, labels


    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        logits_qg, labels_qg = batch_preds
        loss = self.loss_no_reduction(logits_qg.permute(0, 3, 2, 1), labels_qg)
        loss = torch.mean(loss)
        pred = torch.argmax(logits_qg, dim=3)
        # [batchsize, 2 x qgsize]
        correct = torch.sum(pred == labels_qg)
        total_num = pred.shape[0] * pred.shape[1] * pred.shape[2]
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.point_sf.parameters(), 2.0)
        self.optimizer.step()
        return loss, correct, total_num

    def eval_mode(self):
        self.point_sf.eval()
        self.projector.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)
        self.projector.train(mode=True)


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
        # self.optimizer.zero_grad()
        all_correct = torch.tensor([0.0], device=self.device)
        all_attempts = torch.tensor([0.0], device=self.device)
        start_time = time.time()
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            (batch_loss, correct, total_num), stop_training = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k, presort=presort, label_type=label_type)
            # loss = batch_loss/float(size_of_train_data)
            # loss.backward()
            
            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()
            batches_processed += 1
            all_correct += correct
            all_attempts += total_num
        # self.optimizer.step()
        print('Epoch accuracy', all_correct/all_attempts, 'out of', all_attempts, file=sys.stderr)

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