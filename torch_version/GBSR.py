import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.gcn_layer = args.gcn_layer
        self.latent_dim = args.latent_dim
        self.init_type = args.init_type
        self.l2_reg = args.l2_reg
        self.beta = args.beta
        self.sigma = args.sigma
        self.edge_bias = args.edge_bias
        self.batch_size = args.batch_size
        
        # 1. Device 설정 (가장 먼저 수행)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

        # 2. 전체 행렬 가져오기 (social_index는 부정확하므로 무시)
        self.adj_matrix, _ = dataset.get_uu_i_matrix()
        
        # 3. [핵심 수정] OOM 및 인덱스 에러 방지
        # Coalesce된 행렬을 기준으로 직접 Social Edge 판별
        all_indices = self.adj_matrix.indices()
        all_values = self.adj_matrix.values()
        
        # Social Edge 판별 조건: Row와 Col 인덱스 모두 num_user보다 작으면 User-User 엣지임
        row_indices = all_indices[0]
        col_indices = all_indices[1]
        
        # 마스크 생성 (GPU 연산)
        is_social = (row_indices < self.num_user) & (col_indices < self.num_user)
        
        # (1) UI Graph: Social이 아닌 엣지들 (User-Item) -> 고정 (Gradient X)
        self.ui_indices = all_indices[:, ~is_social]
        self.ui_values = all_values[~is_social].detach()
        self.ui_graph = torch.sparse.FloatTensor(
            self.ui_indices, self.ui_values, self.adj_matrix.shape
        ).coalesce().to(self.device)
        
        # (2) Social Graph: Social인 엣지들 -> 학습 대상
        self.social_indices = all_indices[:, is_social]
        self.social_init_values = all_values[is_social]
        
        # Graph Learner용 인덱스
        self.social_u = self.social_indices[0]
        self.social_v = self.social_indices[1]
        
        self._init_weights()


    def _init_weights(self):
        self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
        self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)
        if self.init_type == 'norm':
            nn.init.normal_(self.user_embeddings.weight, std=0.01)
            nn.init.normal_(self.item_embeddings.weight, std=0.01)
        elif self.init_type == 'xa_norm':
            nn.init.xavier_normal_(self.user_embeddings.weight)
            nn.init.xavier_normal_(self.item_embeddings.weight)
        else:
            raise NotImplementedError
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(in_features=2*self.latent_dim, out_features=self.latent_dim, bias=True)
        self.linear_2 = nn.Linear(in_features=self.latent_dim, out_features=1, bias=True)
        return None


    def graph_learner(self, user_emb):
        # 학습 대상인 Social Edge에 대해서만 연산 수행
        row, col = self.social_u, self.social_v
        row_emb = user_emb.weight[row]
        col_emb = user_emb.weight[col]
        cat_emb = torch.cat([row_emb, col_emb], dim=1)
        
        out_layer1 = self.activate(self.linear_1(cat_emb))
        logit = self.linear_2(out_layer1)
        logit = logit.view(-1)
        
        # [수정] 학습(Training) 중일 때만 노이즈 추가, 평가(Eval) 때는 노이즈 제거
        if self.training:
            eps = torch.rand(logit.shape).to(self.device)
            mask_gate_input = torch.log(eps) - torch.log(1 - eps)
            mask_gate_input = (logit + mask_gate_input) / 0.2
            mask_gate_input = torch.sigmoid(mask_gate_input) + self.edge_bias
        else:
            # 평가 시에는 노이즈 없이 Deterministic하게 계산
            # (방법 1) 단순히 Sigmoid 통과 (논문에 따라 다를 수 있으나 가장 일반적)
            # mask_gate_input = torch.sigmoid(logit) + self.edge_bias 
            
            # (방법 2) GBSR 원본 구현을 따른다면, 노이즈 부분(log eps)을 0으로 처리하거나 
            # 단순히 bias만 더해서 sigmoid를 취하기도 함. 
            # 여기서는 노이즈 없이 로짓 그대로 사용하여 가장 강력한 신호만 남깁니다.
            mask_gate_input = torch.sigmoid(logit / 0.2) + self.edge_bias

        # Social 부분만 업데이트 (메모리 절약)
        learned_values = self.social_init_values * mask_gate_input
        
        # Social Graph 생성
        social_graph = torch.sparse.FloatTensor(
            self.social_indices, 
            learned_values, 
            self.adj_matrix.shape
        ).coalesce().to(self.device)
        
        return social_graph


    def forward(self, social_adj):
        '''
        LightGCN-S Encoders
        Fixed UI Graph + Learned Social Graph 합산 전파
        '''
        ego_emb = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb = [ego_emb]
        
        for _ in range(self.gcn_layer):
            # 1. 고정된 UI 그래프 (메모리 효율적)
            out_ui = torch.sparse.mm(self.ui_graph, all_emb[-1])
            
            # 2. 학습된 Social 그래프
            out_social = torch.sparse.mm(social_adj, all_emb[-1])
            
            # 합산
            tmp_emb = out_ui + out_social
            all_emb.append(tmp_emb)
            
        all_emb = torch.stack(all_emb, dim=1)
        mean_emb = torch.mean(all_emb, dim=1)
        user_emb, item_emb = torch.split(mean_emb, [self.num_user, self.num_item])
        return user_emb, item_emb


    def getEmbedding(self, users, pos_items, neg_items):
        users_emb = self.user_emb[users]
        pos_emb = self.item_emb[pos_items]
        neg_emb = self.item_emb[neg_items]
        users_emb_ego = self.user_embeddings(users)
        pos_emb_ego = self.item_embeddings(pos_items)
        neg_emb_ego = self.item_embeddings(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self, users, pos_items, neg_items):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        reg_loss = 1/2 * (userEmb0.norm(2).pow(2) +
                    posEmb0.norm(2).pow(2) +
                    negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        auc = torch.mean((pos_scores > neg_scores).float())
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return auc, bpr_loss, reg_loss*self.l2_reg


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
        # 1. learning denoised social graph (Social Graph만 반환됨)
        self.masked_social_adj = self.graph_learner(self.user_embeddings)
        
        # 2. learning embeddings from lightgcn-s
        # Original Social Graph (No Masking) 생성 for 'old' embeddings
        original_social_graph = torch.sparse.FloatTensor(
            self.social_indices, self.social_init_values, self.adj_matrix.shape
        ).coalesce().to(self.device)
        
        self.user_emb_old, self.item_emb_old = self.forward(original_social_graph)
        self.user_emb, self.item_emb = self.forward(self.masked_social_adj)
        
        # 3. Max mutual information
        auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items)
        # 4. Min mutual information
        ib_loss = self.hsic_graph(users, pos_items) * self.beta
        loss = bpr_loss + reg_loss + ib_loss
        return auc, bpr_loss, reg_loss, ib_loss, loss