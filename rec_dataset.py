import torch
import numpy as np
import random
import scipy.sparse as sp
from time import time
from collections import defaultdict

class Dataset(object):
    def __init__(self, args):
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.args = args
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.social_noise_ratio = args.social_noise_ratio
        
        # 1. 데이터 로드 (형상 자동 감지 및 변환)
        self.load_data()
        
        # 2. 유저/아이템 수 자동 보정
        self.num_user = args.num_user
        self.num_item = args.num_item
        
        max_u = 0
        max_i = 0
        
        # traindata는 이제 무조건 {user_id: [item_id, ...]} 형태임
        for u, items in self.traindata.items():
            if u > max_u: max_u = u
            if items:
                max_i_local = max(items)
                if max_i_local > max_i: max_i = max_i_local

        for u, items in self.testdata.items():
            if u > max_u: max_u = u
            if items:
                max_i_local = max(items)
                if max_i_local > max_i: max_i = max_i_local

        if max_u >= self.num_user:
            self.num_user = max_u + 1
        if max_i >= self.num_item:
            self.num_item = max_i + 1
            
        self.num_node = self.num_user + self.num_item
        print(f"Final Dataset Info: Users={self.num_user}, Items={self.num_item}, Total Nodes={self.num_node}")

        # 3. 학습용 리스트 생성
        if not hasattr(self, 'social_i'):
            self.social_i, self.social_j = [], []
        self.training_user, self.training_item = [], []
        for u, items in self.traindata.items():
            for i in items:
                self.training_user.append(u)
                self.training_item.append(i)
        
        print(f"Total Training Interactions: {len(self.training_user)}")

        # MHCN 호환 필드 구성
        self.n_users = self.num_user
        self.n_items = self.num_item
        self.train_h_list = list(self.training_user)
        self.train_t_list = list(self.training_item)
        self.train_user_dict = {u: list(items) for u, items in self.traindata.items()}
        self.train_item_dict = {}
        for u, items in self.traindata.items():
            for i in items:
                self.train_item_dict.setdefault(i, []).append(u)
        self.train_social_h_list = list(self.social_i)
        self.train_social_t_list = list(self.social_j)
        self.train_social_w_list = None

        # 4. 블랙리스트 네거티브 샘플러 구축
        print("Building negative sampler...")
        self.train_user_pos_set = {u: set(items) for u, items in self.traindata.items()}
        
        self.neg_upper, self.neg_map = self._build_neg_sampler(
            n_items=self.num_item,
            n_users=self.num_user,
            user_pos_set=self.train_user_pos_set
        )
        print("Negative sampler built.")

        # 5. 그래프 생성 (원본 로직: 대칭화 없음, Self-loop 없음)
        self.adj_matrix = self.lightgcn_adj_matrix()
        try:
            self.uu_i_matrix = self.social_lightgcn_adj_matrix()
        except:
            pass

    # --- [데이터 포맷 정규화 핵심 로직] ---
    def _convert_to_dict(self, data):
        """
        입력 데이터가 [User, Item] 쌍의 리스트인지, 
        User별 Item 리스트인지 판별하여 무조건 {User: [Items]} 딕셔너리로 변환
        """
        normalized = defaultdict(list)
        
        # Numpy Array인 경우
        if isinstance(data, np.ndarray):
            # Case A: N x 2 행렬 (User, Item) -> Interaction List
            if data.ndim == 2 and data.shape[1] == 2:
                print("Detected format: N x 2 Interaction Matrix")
                for row in data:
                    u, i = int(row[0]), int(row[1])
                    normalized[u].append(i)
            # Case B: Object Array (각 행이 리스트) -> Adjacency List
            else:
                print("Detected format: Adjacency List (Array)")
                for u, items in enumerate(data):
                    if items is not None:
                        # items가 단일 값인지 리스트인지 확인
                        if isinstance(items, (list, np.ndarray)):
                             normalized[int(u)].extend([int(i) for i in items])
                        else: # 단일 값일 경우
                             normalized[int(u)].append(int(items))
                             
        # 리스트인 경우
        elif isinstance(data, list):
            # 첫 번째 요소로 구조 추론
            if len(data) > 0 and isinstance(data[0], (list, np.ndarray)) and len(data[0]) == 2 and isinstance(data[0][0], int):
                 # [ [u, i], [u, i] ... ] 형태일 가능성 높음 (단, Adjacency List일 수도 있음. 
                 # 하지만 보통 Adjacency List는 길이가 가변적임. 길이가 모두 2라면 의심해봐야 함)
                 # 여기서는 안전하게 Adjacency List로 가정하되, 위쪽 np.load에서 처리됨.
                 # 만약 리스트의 리스트라면 Enumerate 방식 사용
                 for u, items in enumerate(data):
                    if items is not None:
                        normalized[int(u)].extend([int(i) for i in items])
            else:
                for u, items in enumerate(data):
                    if items is not None:
                        normalized[int(u)].extend([int(i) for i in items])
                        
        # 딕셔너리인 경우
        elif isinstance(data, dict):
            for u, items in data.items():
                if items is not None:
                    normalized[int(u)].extend([int(i) for i in items])
                    
        return dict(normalized)

    def load_data(self):
        print("Loading data from:", self.data_path)
        
        # .npy 파일 로드
        raw_train = np.load(self.data_path + 'traindata.npy', allow_pickle=True)
        raw_test = np.load(self.data_path + 'testdata.npy', allow_pickle=True)
        
        # [수정] tolist()를 먼저 하지 않고, Numpy Array 상태에서 형상(Shape) 검사 수행
        self.traindata = self._convert_to_dict(raw_train)
        self.testdata = self._convert_to_dict(raw_test)
        self.valdata = self.testdata 
        
        try:
            # 소셜 데이터 로드
            if self.social_noise_ratio == 0:
                raw_social = np.load(self.data_path + 'user_users_d.npy', allow_pickle=True)
            elif self.social_noise_ratio == 0.2:
                raw_social = np.load(self.data_path + 'attacked_user_users_0.2.npy', allow_pickle=True)
            # ... (나머지 케이스 생략) ...
            else:
                # Fallback
                raw_social = np.load(self.data_path + 'user_users_d.npy', allow_pickle=True)

            self.user_users = self._convert_to_dict(raw_social)
            
            self.social_i, self.social_j = [], []
            for u, users in self.user_users.items():
                if users:
                    self.social_i.extend([u] * len(users))
                    self.social_j.extend(users)
            print('Successfully loaded social networks')
        except Exception as e:
            print(f"Social data loading skipped or failed: {e}")
            pass

    # --- [네거티브 샘플링 로직] ---
    def _build_neg_sampler(self, n_items, n_users, user_pos_set):
        neg_upper = np.zeros(n_users, dtype=np.int64)
        neg_map = [dict() for _ in range(n_users)]
        for u in range(n_users):
            pos = user_pos_set.get(u, set()) # set
            L = n_items - len(pos)
            neg_upper[u] = L
            if L <= 0: continue
            
            head_black = [x for x in pos if x < L]
            tail_white = iter(set(range(L, n_items)) - pos)
            for b in head_black:
                neg_map[u][b] = next(tail_white)
        return neg_upper, neg_map

    def ng_sample(self, u: int) -> int:
        L = int(self.neg_upper[u])
        if L <= 0: return random.randint(0, self.num_item - 1)
        r = random.randint(0, L - 1)
        return self.neg_map[u].get(r, r)

    # --- [그래프 생성 로직 (원본 준수: Self-loop X, Social 대칭 X)] ---
    def social_lightgcn_adj_matrix(self):
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        
        social_i = np.array(self.social_i)
        social_j = np.array(self.social_j)
        social_r = np.ones_like(social_i, dtype=np.float32)
        social_adj = sp.csr_matrix((social_r, (social_i, social_j)), shape=(self.num_node, self.num_node))
        
        adj_mat = adj_mat + social_adj
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def get_uu_i_matrix(self):
        # 1. User-Item / Item-User
        user_dim = torch.LongTensor(self.training_user)
        item_dim = torch.LongTensor(self.training_item) + self.num_user
        
        first_sub = torch.stack([user_dim, item_dim])
        second_sub = torch.stack([item_dim, user_dim])
        
        # 2. Social Graph
        if hasattr(self, 'social_i') and len(self.social_i) > 0:
            s_i = torch.LongTensor(self.social_i)
            s_j = torch.LongTensor(self.social_j)
            third_sub = torch.stack([s_i, s_j])
            index = torch.cat([first_sub, second_sub, third_sub], dim=1)
        else:
            index = torch.cat([first_sub, second_sub], dim=1)

        # 3. Sparse Tensor
        data = torch.ones(index.size(-1)).float()
        
        # CPU Coalesce
        temp_graph = torch.sparse.FloatTensor(index, data, 
                                            torch.Size([self.num_node, self.num_node])).coalesce()
        
        indices = temp_graph.indices()
        values = temp_graph.values() 

        # Degree 계산 (GPU)
        row_indices = indices[0]
        deg = torch.zeros(self.num_node).float().to(self.device)
        deg.index_add_(0, row_indices.to(self.device), values.to(self.device))

        # 정규화
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        col_indices = indices[1]
        norm_values = values.to(self.device) * deg_inv_sqrt[row_indices.to(self.device)] * deg_inv_sqrt[col_indices.to(self.device)]
        
        # GPU Graph 생성 및 Coalesce 마킹
        Graph = torch.sparse.FloatTensor(indices.to(self.device), norm_values, 
            torch.Size([self.num_node, self.num_node]))
        
        Graph = Graph.coalesce()

        return Graph, None

    def lightgcn_adj_matrix(self):
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def _batch_sampling(self, num_negative):
        indices = np.arange(len(self.training_user))
        np.random.shuffle(indices)
        
        batch_num = int(len(indices) / self.batch_size) + 1
        
        training_user = np.array(self.training_user)
        training_item = np.array(self.training_item)

        for k in range(batch_num):
            index_start = k * self.batch_size
            index_end = min((k + 1) * self.batch_size, len(indices))
            if index_start >= len(indices): break
            
            current_batch_indices = indices[index_start:index_end]
            if len(current_batch_indices) < self.batch_size:
                 extra = indices[:self.batch_size - len(current_batch_indices)]
                 current_batch_indices = np.concatenate([current_batch_indices, extra])

            users = training_user[current_batch_indices]
            pos_items = training_item[current_batch_indices]
            neg_items = []
            
            for u in users:
                neg_items.append(self.ng_sample(u))
                
            yield users, pos_items, np.array(neg_items)
