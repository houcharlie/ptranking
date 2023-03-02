import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F

from itertools import product
from torch.optim.lr_scheduler import StepLR
from ptranking.base.utils import get_stacked_FFNet, LTRBatchNorm
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.pretrain.augmentations import zeroes, qgswap, qg_and_zero
from ptranking.data.data_utils import LABEL_TYPE
from ptranking.ltr_adhoc.eval.parameter import ModelParameter
from ptranking.ltr_adhoc.util.lambda_utils import get_pairwise_comp_probs
from absl import logging


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
        self.simsiam_scale = 1.
        self.num_negatives = 300
        self.filtered_negatives = 100
        self.min_swap = int(self.gumbel)
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
        self.simclr_projector, self.simsiam_projector, self.scorer, self.predictor = self.config_heads()
        self.config_optimizer()
        self.xent_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.cosine_loss = nn.CosineSimilarity(dim=1).to(self.device)
        print(self.point_sf, file=sys.stderr)
        print(self.simsiam_projector, file=sys.stderr)
        print(self.simclr_projector, file=sys.stderr)
        print(self.scorer, file=sys.stderr)
        print(self.predictor, file=sys.stderr)
        # self.scheduler = StepLR(self.optimizer, step_size=20, gamma=1.)

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

        simsiam_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                            nn.BatchNorm1d(prev_dim),
                            nn.ReLU(), # first layer
                            nn.Linear(prev_dim, prev_dim, bias=False),
                            nn.BatchNorm1d(prev_dim, affine=False),
                            nn.ReLU(inplace=True), # second layer
                            nn.Linear(prev_dim, dim, bias=False),
                            nn.BatchNorm1d(dim, affine=False))
        
        if self.gpu: simsiam_projector = simsiam_projector.to(self.device)

        pred_dim = int(dim // 4)
        
        predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                    nn.BatchNorm1d(pred_dim),
                                    nn.ReLU(inplace=True), # hidden layer
                                    nn.Linear(pred_dim, dim)) # output layer
        if self.gpu: predictor = predictor.to(self.device)

        simclr_projector = nn.Sequential(nn.Linear(prev_dim, prev_dim),
                                nn.ReLU(), # first layer
                                nn.Linear(prev_dim, dim),
                                nn.ReLU())
        if self.gpu: simclr_projector = simclr_projector.to(self.device)

        # projector = nn.Sequential(
        #     nn.Linear(prev_dim, prev_dim),
        #     nn.ReLU(),  # first layer
        #     nn.Linear(prev_dim, prev_dim),
        #     nn.ReLU())
        scorer = nn.Linear(dim, 1)
        if self.gpu: scorer = scorer.to(self.device)

        return simclr_projector, simsiam_projector, scorer, predictor

    def get_parameters(self):
        all_params = []
        for param in self.point_sf.parameters():
            all_params.append(param)
        for param in self.simsiam_projector.parameters():
            all_params.append(param)
        for param in self.simclr_projector.parameters():
            all_params.append(param)
        for param in self.scorer.parameters():
            all_params.append(param)
        for param in self.predictor.parameters():
            all_params.append(param)

        # for param in self.scorer.parameters():
        #     all_params.append(param)

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

    def simsiam_forward(self, batch_q_doc_vectors):
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
        mod1 = self.point_sf(x1_flat)
        mod2 = self.point_sf(x2_flat)
        z1 = self.simsiam_projector(mod1)
        z2 = self.simsiam_projector(mod2)
        # for param in self.predictor.parameters():
        #     print(param)


        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        # print('data sum', torch.sum(batch_q_doc_vectors))
        # print('x1 sum', torch.sum(x1))
        # print('x2 sum', torch.sum(x2))
        # print('mod1 sum', torch.sum(mod1))
        # print('mod2 sum', torch.sum(mod2))
        # print('z1 sum', torch.sum(z1))
        # print('z2 sum', torch.sum(z2))
        # print('p1 sum', torch.sum(p1))
        # print('p2 sum', torch.sum(p2))
        return p1, p2, z1.detach(), z2.detach()
    
    def rankneg_forward(self, batch_q_doc_vectors):
        batch_size, num_docs, num_features = batch_q_doc_vectors.size()
        target = batch_q_doc_vectors
        ssldata = self.augmentation(batch_q_doc_vectors,
                               self.aug_percent,
                               self.device,
                               mix=self.mix,
                               scale=self.scale)
        # x1_flat = x1.reshape((-1, data_dim))
        # x2_flat = x2.reshape((-1, data_dim))
        # embeddings
        target_embed = self.point_sf(target)
        ssldata_embed = self.point_sf(ssldata)

        # rankneg forward
        target_z = self.simclr_projector(target_embed)
        ssldata_z = self.simclr_projector(ssldata_embed)
        _target_scores = self.scorer(target_z)
        _ssldata_scores = self.scorer(ssldata_z)

        target_scores = _target_scores.view(-1, num_docs)  # [batch_size, num_docs]
        ssldata_scores = _ssldata_scores.view(-1, num_docs)  # [batch_size, num_docs]

        return target_scores, ssldata_scores

    def forward(self, batch_q_doc_vectors):
        '''
        Forward pass through the scoring function, where each document is scored independently.
        @param batch_q_doc_vectors: [batch_size, num_docs, num_features], the latter two dimensions {num_docs, num_features} denote feature vectors associated with the same query.
        @return:
        '''

        p1, p2, z1, z2 = self.simsiam_forward(batch_q_doc_vectors)
        # target_scores, ssldata_scores = self.rankneg_forward(batch_q_doc_vectors)
        # return target_scores, ssldata_scores, p1, p2, z1, z2
        return p1, p2, z1, z2

    def eval_mode(self):
        self.point_sf.eval()
        self.simsiam_projector.eval()
        self.simclr_projector.eval()
        self.scorer.eval()
        self.predictor.eval()

    def train_mode(self):
        self.point_sf.train(mode=True)
        self.simsiam_projector.train(mode=True)
        self.simclr_projector.train(mode=True)
        self.scorer.train(mode=True)
        self.predictor.train(mode=True)

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
    
    # def downsample(self, batch_q_doc_vectors):
    #     batch_size, num_docs, num_features = batch_q_doc_vectors.size()
    #     indices = torch.argsort(torch.rand((batch_size, num_docs)).to(self.device), dim=1).to(self.device)[:,:,None]
    #     indices = indices.expand(batch_size, num_docs, num_features)
    #     row_indices = torch.arange(batch_size)[:, None, None].expand(batch_size, num_docs, num_features)
    #     feature_indices = torch.arange(num_features)[None, None, :].expand(batch_size, num_docs, num_features)
    #     shuffled_q_doc_vectors = batch_q_doc_vectors.clone()[row_indices, indices, feature_indices]
    #     batch_q_doc_vectors = shuffled_q_doc_vectors[:,:self.qg_limit,:]
    #     return batch_q_doc_vectors

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
        all_correct = torch.tensor([0.0], device=self.device)
        all_attempts = torch.tensor([0.0], device=self.device)
        acc_batch_size = torch.tensor([0.0], device=self.device)
        acc_batch = []
        stop_training = False
        for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:  # batch_size, [batch_size, num_docs, num_features], [batch_size, num_docs]
            # if batch_q_doc_vectors.shape[1] < self.qg_limit:
            #     continue
            # acc_batch.append(self.downsample(batch_q_doc_vectors))
            # acc_batch_size += batch_q_doc_vectors.shape[0]
            # if acc_batch_size > self.acc_batch_size:
                # all_batch = torch.cat(acc_batch, axis = 0)
                # acc_batch = []
                # acc_batch_size = torch.tensor([0.0], device=self.device)
            num_queries += len(batch_ids)
            if self.gpu:
                batch_q_doc_vectors, batch_std_labels = batch_q_doc_vectors.to(
                    self.device), batch_std_labels.to(self.device)

            (batch_loss, correct,
            attempts), _ = self.train_op(batch_q_doc_vectors,
                                            batch_std_labels,
                                            batch_ids=batch_ids,
                                            epoch_k=epoch_k,
                                            presort=presort,
                                            label_type=label_type)
            epoch_loss += batch_loss.item()
            all_correct += correct
            all_attempts += attempts
            batches_processed += 1
            
        print('Epoch accuracy',
              'qg_correct',
              all_correct / all_attempts,
              'out of',
              all_attempts,
              file=sys.stderr)
        epoch_loss = epoch_loss / num_queries
        return epoch_loss, stop_training

    def custom_loss_function(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch_size, num_docs, num_features]
        @param batch_std_labels: not used
        @param kwargs:
        @return:
        '''
        # target_scores, source_scores, p1, p2, z1, z2 = batch_preds
        p1, p2, z1, z2 = batch_preds
        simsiam_loss = -(self.cosine_loss(p1, z2).mean() + self.cosine_loss(p2, z1).mean()) * 0.5
        # rankneg_loss, correct, total_num = self.rankneg_loss(target_scores, source_scores)
        # loss = self.blend * rankneg_loss + (1. - self.blend) * simsiam_loss
        loss = simsiam_loss
        # print_dict = [
            # ('simclr_proj', self.simclr_projector),
            # ('scorer', self.scorer),
        #     ('simsiam_proj', self.simsiam_projector),
        #     ('predictor', self.predictor),
        #     ('pointsf', self.point_sf)
        # ]
        # for module in print_dict:
        #     currsum = torch.Tensor([0.]).to(self.device)
        #     for param in module[1].parameters():
        #         currsum += torch.sum(param)
        #     print(module[0], currsum)
        # print('')
        # print('overall loss', loss)
        # print('rankneg loss', rankneg_loss)
        # print('simsiam loss', simsiam_loss)
        # import ipdb; ipdb.set_trace()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return loss, correct, total_num
        return loss, 0, 1

    

    def rankneg_loss(self, target_scores, source_scores):
        '''
        target scores: [batch_size, num_docs]
        source scores: [batch_size, num_docs]
        '''

        batch_size = target_scores.shape[0]
        num_scores = target_scores.shape[1]

        # close_negs = target_scores[:, None, :].expand((batch_size, self.num_negatives, num_scores)).clone().to(self.device) # targets
        # target_sorted_idx = torch.argsort(target_scores, dim=1, descending=True)[:, None, :].to(self.device).expand((batch_size, self.num_negatives, num_scores))
        # swap_num = torch.randint(low=self.min_swap, high=num_scores, size=(batch_size, self.num_negatives, 1)).to(self.device) # the rank, above which we will
        # row_indices = torch.arange(0, batch_size)[:, None, None].to(self.device).expand((batch_size, self.num_negatives, 1))
        # negatives_indices = torch.arange(self.num_negatives, device = self.device)[None, :, None].expand((batch_size, self.num_negatives, 1))
        # shift_val = close_negs[row_indices, negatives_indices, target_sorted_idx[row_indices, negatives_indices, swap_num]] # batch_size x self.num_negatives x 1
        # close_negs_shift = close_negs - shift_val
        # random_nums = torch.rand_like(close_negs_shift, device = self.device)
        # close_negs_shift[torch.nonzero(close_negs_shift > 0., as_tuple=True)] = random_nums[torch.nonzero(close_negs_shift > 0., as_tuple=True)]
        # close_negs = target_scores.clone().to(self.device)
        # close_negs_abs = torch.abs(close_negs).to(self.device)
        # close_negs_max = torch.max(close_negs_abs).to(self.device)
        # close_negs_normalized = close_negs / close_negs_max 
        # close_negs = close_negs.expand((batch_size, self.num_negatives, num_scores)).to(self.device)
        # close_negs_shift = 


        # completely random scores
        # sampled_neg_probs = torch.randn((batch_size, self.num_negatives - self.num_close_negatives, num_scores), device=self.device)
        
        # close_negs = target_scores[:, None, :].expand((batch_size, self.num_close_negatives, num_scores)).clone().to(self.device)
        # swap_product_idx_more = torch.randint(num_scores, size=(batch_size * self.num_close_negatives * 2,  2), device=self.device)
        # swap_product_idx_mask = swap_product_idx_more[(swap_product_idx_more[:,0] != swap_product_idx_more[:,1]).nonzero().squeeze(),:]
        # swap_product_idx_not_shape = swap_product_idx_mask[:batch_size * self.num_close_negatives, :]
        # swap_product_idx = swap_product_idx_not_shape.reshape((batch_size, self.num_close_negatives, 2))
        # swap_product_idx_opposite = swap_product_idx.clone()
        # swap_product_idx_opposite[:,:,0] = swap_product_idx[:,:,1]
        # swap_product_idx_opposite[:,:,1] = swap_product_idx[:,:,0]
        # row_indices = torch.arange(0, batch_size)[:, None, None].to(self.device).expand((batch_size, self.num_close_negatives, 2))
        # negatives_indices = torch.arange(self.num_close_negatives)[None, :, None].expand(batch_size, self.num_close_negatives, 2)
        # close_negs[row_indices, negatives_indices, swap_product_idx] = close_negs.clone()[row_indices, negatives_indices, swap_product_idx_opposite]


        sampled_neg_scores = target_scores
        sampled_neg_scores = sampled_neg_scores[:, None, :].expand((batch_size, self.num_negatives, num_scores))
        # soft, perturbed rankings
        # sampled_neg_probs = F.gumbel_softmax(sampled_neg_scores, tau=self.gumbel, hard=False, dim=2)

        # indices = torch.argsort(torch.rand((batch_size, self.num_negatives, num_scores)).to(self.device), dim=1).to(self.device)
        # row_indices = torch.arange(batch_size)[:, None, None].expand(batch_size, self.num_negatives, num_scores)
        # negatives_indices = torch.arange(self.num_negatives)[None, None, :].expand(batch_size, self.num_negatives, )
        # shuffled_q_doc_vectors = batch_q_doc_vectors.clone()[row_indices, indices, feature_indices]

        gumbels = -torch.empty_like(sampled_neg_scores).exponential_().log()
        sampled_neg_probs = sampled_neg_scores + self.gumbel * gumbels



        # sampled_neg

        # calculate hard negatives with hard labels [batch_size, num negatives, scores]
        argsort_indices = torch.argsort(sampled_neg_probs, dim=2).to(self.device)
        row_indices = torch.arange(0, batch_size)[:, None, None].to(self.device).expand((batch_size, self.num_negatives, num_scores))
        negative_indices = torch.arange(0, self.num_negatives)[None, :, None].to(self.device).expand((batch_size, self.num_negatives, num_scores))
        fill_in_vals = torch.arange(0, num_scores, dtype=torch.float32)[None, None, :].to(self.device).expand((batch_size, self.num_negatives, num_scores))
        hard_rank_negs = torch.zeros_like(sampled_neg_probs).to(self.device)
        hard_rank_negs[row_indices, negative_indices, argsort_indices] = fill_in_vals

        # real labels [batch_size, scores]
        argsort_indices = torch.argsort(target_scores, dim=1).to(self.device)
        row_indices = torch.arange(0, batch_size)[:, None].to(self.device).expand((batch_size, num_scores))
        fill_in_vals = torch.arange(0, num_scores, dtype=torch.float32)[None, :].to(self.device).expand((batch_size, num_scores))
        hard_rank_target = torch.zeros_like(target_scores).to(self.device)
        hard_rank_target[row_indices, argsort_indices] = fill_in_vals

        # filter out all the negatives that match the target
        hard_rank_neg_diff = hard_rank_negs - hard_rank_target[:, None, :]
        hard_rank_neg_diff_abs = torch.abs(hard_rank_neg_diff).to(self.device)
        hard_rank_neg_diff_condensed = torch.sum(hard_rank_neg_diff_abs, dim=2).to(self.device)
        non_match_mask = (hard_rank_neg_diff_condensed != 0.).to(self.device)
        row_wise_mask = torch.prod(non_match_mask, axis=0, keepdim=True).to(self.device)
        mask_uniform = non_match_mask * row_wise_mask

        cumsum_mask = torch.cumsum(mask_uniform[0,:], dim=0).to(self.device)
        idx_filtered = torch.nonzero(cumsum_mask == self.filtered_negatives).to(self.device)[0,0] + 1
        mask_uniform_down_sample = mask_uniform[:, :idx_filtered]
        matching_idx = torch.nonzero(mask_uniform_down_sample, as_tuple=True)
        row_idx, neg_idx = matching_idx
        hard_rank_negs_clean = hard_rank_negs[row_idx, neg_idx, :].reshape((batch_size, self.filtered_negatives, num_scores)).to(self.device)

        # ranknet similarity scores
        preds = source_scores[:,None,:].expand(batch_size, self.filtered_negatives + 1, num_scores).to(self.device)
        all_targets = torch.cat([hard_rank_target[:,None,:], hard_rank_negs_clean], dim=1).to(self.device)
        logits = -self.ranknet_loss_mat(preds, all_targets.detach())

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        logits = logits / self.temperature

        loss = self.xent_loss(logits, labels)
        match_pred = torch.argmax(logits, dim=1)
        match_correct = torch.sum(match_pred == labels)
        total_num = logits.shape[0]

        return loss, match_correct, total_num


    def ranknet_loss_mat(self, batch_preds, batch_std_labels, **kwargs):
        '''
        @param batch_preds: [batch, batch, ranking_size] each row represents the relevance predictions for documents associated with the same query
        @param batch_std_labels: [batch, batch, anking_size] each row represents the standard relevance grades for documents associated with the same query
        @param kwargs:
        @return:
        '''
        batch_p_ij, batch_std_p_ij = self.get_pairwise_comp_probs_hard_mat(batch_preds=batch_preds, batch_std_labels=batch_std_labels,
                                                             sigma=1.)

        # import ipdb; ipdb.set_trace()
        _batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1),
                                             target=torch.triu(batch_std_p_ij, diagonal=1), reduction='none')
        # _batch_loss = F.binary_cross_entropy(input=batch_p_ij,
        #                                      target=batch_std_p_ij, reduction='none')
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
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)
        # batch_std_p_ij = torch.where(batch_std_diffs > 0., 1., 0.)
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