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

class SubTab(NeuralRanker):

    def __init__(self,
                 id='SubTab',
                 sf_para_dict=None,
                 model_para_dict=None,
                 weight_decay=1e-3,
                 gpu=False,
                 device=None):
        super(SubTab, self).__init__(id=id,
                                      sf_para_dict=sf_para_dict,
                                      weight_decay=weight_decay,
                                      gpu=gpu,
                                      device=device)
        self.gaussian_noise = 0.1
        self.subsets = 4
        
        self.aug_percent = model_para_dict['aug_percent']

        self.mse_loss = torch.nn.MSELoss().to(self.device)

    def init(self):
        self.point_sf, self.decoder = self.config_point_neural_scoring_function()
        self.config_optimizer()


    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.decoder.parameters():
            all_params.append(param)
        return nn.ParameterList(all_params)
    
    
    def config_point_neural_scoring_function(self):
        point_sf, decoder = self.ini_pointsf(**self.sf_para_dict[self.sf_para_dict['sf_id']])
        if self.gpu: 
            point_sf = point_sf.to(self.device)
            decoder = decoder.to(self.device)
        return point_sf, decoder
        
    def ini_pointsf(self, num_features=None, h_dim=100, out_dim=136, num_layers=3, AF='R', TL_AF='S', apply_tl_af=False,
                BN=True, bn_type=None, bn_affine=False, dropout=0.1):
        '''
        Initialization of a feed-forward neural network
        '''
        h_dim = 136
        self.subset_size = int(num_features * 0.75)
        self.increment = (num_features - self.subset_size)//self.subsets
        point_sf = get_resnet(self.subset_size, h_dim)
        decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_features)
        )
        return point_sf, decoder

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''
        num_features = batch_q_doc_vectors.shape[2]
        reconstructions = []
        for i in range(self.subsets):
            if i == self.subsets - 1:
                start = num_features - self.subset_size
                end = num_features
            else:
                start = i * self.increment
                end = start + self.subset_size
                
            augmented_data = gaussian(zeroes(batch_q_doc_vectors[:,:,start:end], self.aug_percent, self.device), self.gaussian_noise, self.device)
            encoded = self.point_sf(augmented_data)
            decoded = self.decoder(encoded)
            reconstructions.append(decoded)
        
        return reconstructions, batch_q_doc_vectors

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        reconstructions, orig = batch_preds
        loss = torch.tensor([0.]).to(self.device)
        normalization = float(len(reconstructions))
        for reconstruction in reconstructions:
            loss += self.mse_loss(reconstruction, orig) / normalization
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def eval_mode(self):
        self.point_sf.eval()
    def train_mode(self):
        self.point_sf.train(mode=True)
        self.decoder.train(mode=True)

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

class SubTabParameter(ModelParameter):
    ''' Parameter class for SimRank '''

    def __init__(self, debug=False, para_json=None):
        super(SubTabParameter, self).__init__(model_id='SubTab',
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