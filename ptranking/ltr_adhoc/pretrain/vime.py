import torch
import torch.nn as nn
import os
import sys
import math
import time
from itertools import product
from ptranking.base.utils import get_stacked_FFNet, get_resnet
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap, gaussian
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from absl import logging
import torch.nn.functional as F
from ptranking.metric.metric_utils import get_delta_ndcg
from torch.nn.init import eye_ as eye_init
from torch.nn.init import zeros_ as zero_init

class VIME(NeuralRanker):

    def __init__(self,
                 id='VIME',
                 sf_para_dict=None,
                 model_para_dict=None,
                 weight_decay=1e-3,
                 gpu=False,
                 device=None):
        super(VIME, self).__init__(id=id,
                                      sf_para_dict=sf_para_dict,
                                      weight_decay=weight_decay,
                                      gpu=gpu,
                                      device=device)
        self.aug_percent = model_para_dict['aug_percent']

    def init(self):
        self.point_sf, self.decoder1, self.decoder2 = self.config_point_neural_scoring_function()
        self.xent_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.mse_loss = torch.nn.MSELoss().to(self.device)
        self.config_optimizer()


    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.decoder1.parameters():
            all_params.append(param)
        for param in self.decoder2.parameters():
            all_params.append(param)
        return nn.ParameterList(all_params)
    
    
    def config_point_neural_scoring_function(self):
        point_sf, decoder1, decoder2 = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: 
            point_sf = point_sf.to(self.device)
            decoder1 = decoder1.to(self.device)
            decoder2 = decoder2.to(self.device)
        return point_sf, decoder1, decoder2
        
    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=136, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        h_dim = 136
        point_sf = get_resnet(num_features, h_dim)
        decoder1 = nn.Sequential(
            nn.Linear(h_dim, num_features)
        )
        decoder2 = nn.Sequential(
            nn.Linear(h_dim, num_features),
            nn.Sigmoid()
        )
        return point_sf, decoder1, decoder2

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        orig_shape = batch_q_doc_vectors.shape
        data_dim = batch_q_doc_vectors.shape[2]
        x_flat = batch_q_doc_vectors.reshape(-1, data_dim)
        corrupted_indices_cont = torch.rand(x_flat.shape).to(self.device)
        corrupted_indices_indicator = (corrupted_indices_cont < self.aug_percent).to(self.device)
        dim0_target, dim1_target = torch.where(corrupted_indices_indicator)
        dim0_source = torch.randint(0, x_flat.shape[0], size=dim0_target.shape).to(self.device)
        aug_x = x_flat.detach().clone().to(self.device)
        aug_x[dim0_target, dim1_target] = x_flat[dim0_source, dim1_target].detach().clone().to(self.device)

        z = self.point_sf(aug_x.reshape(orig_shape))
        out_dim = z.shape[2]
        z = z.reshape(-1, out_dim)
        x_reconstruct = self.decoder1(z)
        m_reconstruct = self.decoder2(z)

        m = corrupted_indices_indicator.float()
        

        return x_flat, x_reconstruct, m, m_reconstruct

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        x, x_, m, m_ = batch_preds
        loss = 1.0 * F.binary_cross_entropy(m_, m) + 2.0 * self.mse_loss(x, x_)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def eval_mode(self):
        self.point_sf.eval()
        self.decoder1.eval()
        self.decoder2.eval()
    def train_mode(self):
        self.point_sf.train(mode=True)
        self.decoder1.train(mode=True)
        self.decoder2.train(mode=True)

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
        
        start_time = time.time()
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data: # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            num_queries += len(batch_ids)
            if self.gpu: batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(self.device), batch_std_labels.to(self.device)

            batch_loss, stop_training = self.train_op(batch_q_doc_vectors, batch_std_labels, batch_ids=batch_ids, epoch_k=epoch_k, presort=presort, label_type=label_type)
            # loss = batch_loss/float(size_of_train_data)
            # loss.backward()
            
            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()
            batches_processed += 1
        print("---One epoch time %s seconds ---" % (time.time() - start_time), file=sys.stderr)
        # self.optimizer.step()
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

class VIMEParameter(ModelParameter):
    ''' Parameter class for SimRank '''

    def __init__(self, debug=False, para_json=None):
        super(VIMEParameter, self).__init__(model_id='VIME',
                                               para_json=para_json)
        self.debug = debug

    def default_para_dict(self):
        """
        Default parameter setting for SimRank
        :return:
        """
        self.para_dict = dict(model_id=self.model_id,
                              aug_percent=0.7)
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
            'aug_percent', '{:,g}'.format(para_dict['aug_percent'])
        ])
        return para_str

    def grid_search(self):
        """
        Iterator of parameter settings for simrank
        """
        if self.use_json:
            choice_aug = self.json_dict['aug_percent']
        else:
            choice_aug = [0.3, 0.7
                          ] if self.debug else [0.7]  # 1.0, 10.0, 50.0, 100.0

        for aug_percent in choice_aug:
            self.para_dict = dict(model_id=self.model_id,
                                  aug_percent=aug_percent)
            yield self.para_dict