import torch
import torch.nn as nn
import torch.nn.functional as F

from MHCN import MHCN


def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2


class GBSR(nn.Module):
    def __init__(self, args, dataset):
        super(GBSR, self).__init__()
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.latent_dim = args.latent_dim
        self.init_type = args.init_type
        self.l2_reg = args.l2_reg
        self.beta = args.beta
        self.sigma = args.sigma
        self.edge_bias = args.edge_bias
        self.batch_size = args.batch_size

        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

        args.device = self.device
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.latent_dim = args.latent_dim

        self.backbone = MHCN(dataset, args).to(self.device)
        self.base_H_s = self.backbone.H_s
        self.base_H_j = self.backbone.H_j
        self.base_H_p = self.backbone.H_p

        if hasattr(dataset, "social_i") and len(dataset.social_i) > 0:
            social_u = torch.as_tensor(dataset.social_i, dtype=torch.long, device=self.device)
            social_v = torch.as_tensor(dataset.social_j, dtype=torch.long, device=self.device)
        else:
            social_u = torch.empty(0, dtype=torch.long, device=self.device)
            social_v = torch.empty(0, dtype=torch.long, device=self.device)

        if social_u.numel() > 0:
            self.social_indices = torch.stack([social_u, social_v], dim=0)
            self.social_init_values = torch.ones_like(social_u, dtype=torch.float)
        else:
            self.social_indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.social_init_values = torch.empty((0,), dtype=torch.float, device=self.device)

        self.social_u = social_u
        self.social_v = social_v

        self._init_weights()

    def _init_weights(self):
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=2*self.latent_dim, out_features=self.latent_dim, bias=True)
        self.linear_2 = nn.Linear(in_features=self.latent_dim, out_features=1, bias=True)
        return None

    def _row_normalize(self, sparse_mat):
        if sparse_mat._nnz() == 0:
            return sparse_mat
        indices = sparse_mat.indices()
        values = sparse_mat.values()
        row = indices[0]
        row_sum = torch.zeros(sparse_mat.size(0), device=values.device)
        row_sum.index_add_(0, row, values)
        inv = row_sum.pow(-1)
        inv[inv == float('inf')] = 0
        values = values * inv[row]
        return torch.sparse.FloatTensor(indices, values, sparse_mat.size()).coalesce()

    def graph_learner(self):
        if self.social_u.numel() == 0:
            return self.base_H_s
        row_emb = self.backbone.user_embeddings.weight[self.social_u]
        col_emb = self.backbone.user_embeddings.weight[self.social_v]
        cat_emb = torch.cat([row_emb, col_emb], dim=1)

        out_layer1 = self.activate(self.linear_1(cat_emb))
        logit = self.linear_2(out_layer1).view(-1)

        if self.training:
            eps = torch.rand(logit.shape, device=self.device)
            mask_gate_input = torch.log(eps) - torch.log(1 - eps)
            mask_gate_input = (logit + mask_gate_input) / 0.2
            mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias
        else:
            mask_gate_input = torch.sigmoid(logit / 0.2) + self.edge_bias

        learned_values = self.social_init_values * mask_gate_input
        social_graph = torch.sparse.FloatTensor(
            self.social_indices,
            learned_values,
            torch.Size([self.num_user, self.num_user]),
        ).coalesce().to(self.device)
        return self._row_normalize(social_graph)

    def _apply_masked_social_graph(self, social_graph):
        if social_graph is self.base_H_s or social_graph._nnz() == 0:
            self.backbone.H_s = self.base_H_s
            self.backbone.H_j = self.base_H_j
            self.backbone.H_p = self.base_H_p
            return
        coalesced = social_graph.coalesce()
        idx = coalesced.indices().detach().cpu().numpy()
        val = coalesced.values().detach().cpu().numpy()
        data = self.backbone.data
        data.train_social_h_list = idx[0].tolist()
        data.train_social_t_list = idx[1].tolist()
        data.train_social_w_list = val.tolist()
        self.backbone.update_channels_from_S_fast(rebuild_R=False)

    def infer_embedding(self):
        masked_social_adj = self.graph_learner()
        self._apply_masked_social_graph(masked_social_adj)
        return self.backbone.infer_embedding()

    def forward(self, *unused):
        return self.infer_embedding()

    def hsic_graph(self, users, pos_items):
        ### user part ###
        users = torch.unique(users)
        items = torch.unique(pos_items)
        
        input_x = self.user_emb_old[users]
        input_y = self.user_emb[users]
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        
        # 정확한 샘플 수 사용
        loss_user = hsic(Kx, Ky, input_x.size(0))
        
        ### item part ###
        input_i = self.item_emb_old[items]
        input_j = self.item_emb[items]
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        
        loss_item = hsic(Ki, Kj, input_i.size(0))
        
        loss = loss_user + loss_item
        return loss


    def calculate_all_loss(self, users, pos_items, neg_items):
        masked_social_adj = self.graph_learner()

        self.backbone.H_s = self.base_H_s
        self.backbone.H_j = self.base_H_j
        self.backbone.H_p = self.base_H_p
        self.user_emb_old, self.item_emb_old = self.backbone.infer_embedding()

        self._apply_masked_social_graph(masked_social_adj)
        self.user_emb, self.item_emb = self.backbone.infer_embedding()

        mhcn_loss_term, reg_loss, ss_loss = self.backbone.bpr_loss(users, pos_items, neg_items)

        user_emb = self.user_emb[users]
        pos_emb = self.item_emb[pos_items]
        neg_emb = self.item_emb[neg_items]
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        auc = torch.mean((pos_scores > neg_scores).float())

        reg_term = reg_loss * self.lambda1 + ss_loss * self.lambda2
        ib_loss = self.hsic_graph(users, pos_items) * self.beta
        loss = mhcn_loss_term + reg_term + ib_loss
        return auc, mhcn_loss_term, reg_term, ib_loss, loss
