# -*- coding: utf-8 -*-
"""
A script for implementation of heterogeneous neighbor contrastive graph attention network for metabolite-protein interaction prediction in plant.
@author: xzhou
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import sparse
from utils import readListfile, get_train_index, calculateauc
import argparse
import warnings
from loss import multi_contrastive_loss
from loss_weighted import AdaptiveWeightedPConLoss, AdaptiveWeightedMConLoss, AdaptiveWeightedFConLoss
from datetime import date

# protein: nodeA; Metabolite: nodeB; GO: nodeC

class GraphDiffusion(torch.nn.Module):
    """
    图扩散模块：在相似度/结构图上进行多步扩散。
    这里采用对称归一化的多阶邻接累加（近似热扩散 / Random Walk with Restart 形式）：
        S = I + alpha * A_norm + alpha^2 * A_norm^2 + ... + alpha^K * A_norm^K
    其中 A_norm = D^{-1/2} A D^{-1/2}
    """
    def __init__(self, K: int = 3, alpha: float = 0.2):
        super(GraphDiffusion, self).__init__()
        self.K = K
        self.alpha = alpha

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        adj: [N, N] 的稠密相似度/邻接矩阵（非负）
        返回: 扩散后的相似度矩阵 S
        """
        # 确保为浮点类型
        adj = adj.float()
        # 添加自环，保证每个节点至少和自己相连
        N = adj.size(0)
        device = adj.device
        I = torch.eye(N, device=device)
        adj_with_self = adj + I

        # 对称归一化 A_norm = D^{-1/2} A D^{-1/2}
        deg = adj_with_self.sum(1, keepdim=True).clamp(min=1.0)
        D_inv_sqrt = deg.pow(-0.5)
        A_norm = D_inv_sqrt * adj_with_self * D_inv_sqrt.t()

        # 多阶扩散累加
        S = I.clone()
        current = A_norm
        for k in range(1, self.K + 1):
            S = S + (self.alpha ** k) * current
            current = torch.mm(current, A_norm)

        # 归一化到 [0,1] 区间，避免数值过大
        S_min = S.min()
        S_max = S.max()
        if (S_max - S_min) > 1e-8:
            S = (S - S_min) / (S_max - S_min)
        return S


class DiffusionHNCGATEncoder(torch.nn.Module):
    """
    扩散增强的异质图注意力编码器：
    - 首先在蛋白相似图 / 代谢物相似图上进行图扩散，得到扩散图
    - 同时利用异质边 (P-M, P-F, M-F) 构造同质“共邻居”图，与原始相似图融合
    - 最终将扩散后的相似度矩阵输入原始的 graphNetworkEmbbed

    这样实现了“扩散图 + 异质注意力”的结构，既提高全局结构建模能力，又与原论文框架保持兼容。
    """
    def __init__(self, nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num,
                 nodeC_feature_num, hidden_dim, dropout,
                 use_diffusion: bool = False,
                 diff_K: int = 3,
                 diff_alpha: float = 0.2,
                 diff_beta: float = 0.5):
        super(DiffusionHNCGATEncoder, self).__init__()
        self.use_diffusion = use_diffusion
        self.diff_beta = diff_beta

        # 扩散算子（同质图）
        if use_diffusion:
            self.diffusion_protein = GraphDiffusion(K=diff_K, alpha=diff_alpha)
            self.diffusion_metabolite = GraphDiffusion(K=diff_K, alpha=diff_alpha)

        # 原始异质图注意力编码器（保持不变）
        self.base_encoder = graphNetworkEmbbed(
            nodeA_num, nodeB_num,
            nodeA_feature_num, nodeB_feature_num, nodeC_feature_num,
            hidden_dim, dropout
        )

    def build_meta_graphs(self,
                          adj_AB: torch.Tensor,
                          adj_AC: torch.Tensor,
                          adj_BC: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        基于异质边构造蛋白/代谢物同质“共邻居”图：
            A_meta = A-B-A + A-F-A
            B_meta = B-A-B + B-F-B
        其中 A-B-A 表示两个蛋白通过代谢物共邻居连接。
        """
        # 确保为 float
        adj_AB = adj_AB.float()
        adj_AC = adj_AC.float()
        adj_BC = adj_BC.float()

        # A_meta: [numA, numA]
        A_meta_PM = torch.mm(adj_AB, adj_AB.t())
        A_meta_PF = torch.mm(adj_AC, adj_AC.t())
        A_meta = (A_meta_PM > 0).float() + (A_meta_PF > 0).float()

        # B_meta: [numB, numB]
        B_meta_MP = torch.mm(adj_AB.t(), adj_AB)
        B_meta_MF = torch.mm(adj_BC, adj_BC.t())
        B_meta = (B_meta_MP > 0).float() + (B_meta_MF > 0).float()

        # 去掉对角线将在扩散中重新加自环
        A_meta.fill_diagonal_(0.0)
        B_meta.fill_diagonal_(0.0)
        return A_meta, B_meta

    def forward(self,
                adj_AB: torch.Tensor,
                adj_AC: torch.Tensor,
                adj_BC: torch.Tensor,
                nodeA_feature: torch.Tensor,
                nodeB_feature: torch.Tensor,
                nodeC_feature: torch.Tensor,
                adj_A_sim: torch.Tensor,
                adj_B_sim: torch.Tensor):

        if self.use_diffusion:
            # 基于异质边构造同质图（每个模型实例只计算一次并缓存）
            if not hasattr(self, "_A_meta") or not hasattr(self, "_B_meta"):
                A_meta, B_meta = self.build_meta_graphs(adj_AB, adj_AC, adj_BC)
                self._A_meta = A_meta.to(adj_A_sim.device)
                self._B_meta = B_meta.to(adj_B_sim.device)
            else:
                A_meta, B_meta = self._A_meta, self._B_meta

            # 将原始相似图与共邻居图融合，再做扩散
            # adj_A_sim / adj_B_sim 可能为0/1或权重，这里先转为float
            A_base = adj_A_sim.float()
            B_base = adj_B_sim.float()

            A_fused = A_base + self.diff_beta * A_meta.to(A_base.device)
            B_fused = B_base + self.diff_beta * B_meta.to(B_base.device)

            diff_A_sim = self.diffusion_protein(A_fused)
            diff_B_sim = self.diffusion_metabolite(B_fused)
        else:
            diff_A_sim = adj_A_sim
            diff_B_sim = adj_B_sim

        # 调用原始异质图注意力编码器，但使用“扩散后”的相似度矩阵
        nodeA_emb, nodeB_emb, nodeC_emb = self.base_encoder(
            adj_AB, adj_AC, adj_BC,
            nodeA_feature, nodeB_feature, nodeC_feature,
            diff_A_sim, diff_B_sim
        )
        return nodeA_emb, nodeB_emb, nodeC_emb


class MPINet(torch.nn.Module):
    def __init__(self, nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, hidden_dim,
                 dropout, edgetype, use_weighted_loss=False, use_attention=True, tau=0.1, weight_temperature=0.7,
                 use_diffusion: bool = False, diff_K: int = 3, diff_alpha: float = 0.2, diff_beta: float = 0.5):
        super(MPINet, self).__init__()
        # 用“扩散 + 异质注意力”的编码器替换原始编码器
        self.encoder_1 = DiffusionHNCGATEncoder(
            nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num,
            nodeC_feature_num, hidden_dim, dropout,
            use_diffusion=use_diffusion,
            diff_K=diff_K,
            diff_alpha=diff_alpha,
            diff_beta=diff_beta
        )
        self.decoder = MlpDecoder(hidden_dim, edgetype)
        self.use_weighted_loss = use_weighted_loss
        
        # 如果使用加权损失，添加加权损失模块作为可训练参数
        if use_weighted_loss:
            self.weighted_P_loss = AdaptiveWeightedPConLoss(hidden_dim, tau, use_attention, weight_temperature)
            self.weighted_M_loss = AdaptiveWeightedMConLoss(hidden_dim, tau, use_attention, weight_temperature)
            self.weighted_F_loss = AdaptiveWeightedFConLoss(hidden_dim, tau, use_attention, weight_temperature)

    def forward(self, data_set, adj_AB, adj_AC, adj_BC, nodeA_feature, nodeB_feature, nodeC_feature, adj_A_sim,
                adj_B_sim,edgetype):
        nodeA_feature, nodeB_feature, nodeC_feature = self.encoder_1(adj_AB, adj_AC, adj_BC,
                                                                     nodeA_feature, nodeB_feature, nodeC_feature,
                                                                     adj_A_sim, adj_B_sim)

        predictAfeature = nodeA_feature[data_set[:, 0],]
        predictBfeature = nodeB_feature[data_set[:, 1],]

        prediction = self.decoder(predictAfeature, predictBfeature,edgetype).flatten()
        return prediction,nodeA_feature, nodeB_feature, nodeC_feature


class biattenlayer(torch.nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(biattenlayer, self).__init__()
        self.atten_ItoJ = nn.Conv1d(hidden_dim, 1, 1)
        self.atten_JtoI = nn.Conv1d(hidden_dim, 1, 1)
        self.dropout = dropout
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.02)
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        
        nn.init.xavier_normal_(self.atten_ItoJ.weight, gain=1.414)
        nn.init.xavier_normal_(self.atten_JtoI.weight, gain=1.414)

    def forward(self, nodeI_mlp, nodeJ_mlp, adj_mat):
        """
        return 0 if adj>0, -1e9 if adj=0
        """
        X_I = torch.unsqueeze(torch.transpose(nodeI_mlp, 0, 1), 0) #shape (1,hid,num_I)
        X_J = torch.unsqueeze(torch.transpose(nodeJ_mlp, 0, 1), 0) #shape (1,hid,num_J)
        ### a*[h_i||h_j]=a_i*h_i + a_j*h_j
        f_ItoJ = self.atten_ItoJ(X_I)  # shape (1,1,num_I), a_i*h_i
        f_JtoI = self.atten_JtoI(X_J)  # shape (1,1,num_J), a_j*h_j
        edge_logits = f_ItoJ + torch.transpose(f_JtoI, 2, 1)  # shape (1,num_J,num_I)
        
        edge_logits = torch.squeeze(edge_logits)  # from (1,num_J,num_I) to (num_J,num_I)
        ###bias mat is 0 if adj>0, -1e9 if adj=0
        edge_logits = self.leakyrelu(edge_logits)
        edge_logits=edge_logits.T # from (num_J,num_I) to (num_I,num_J)
        ###softmax only among neighborhood
        zero_vec = -1e9 * torch.ones_like(edge_logits)
        ##if adj>0, keep edge_logits; if adj=0, replace by -1e9
        
        att_IJ = torch.where(adj_mat > 0, edge_logits, zero_vec)
        att_IJ = self.softmax(att_IJ)
        rowmean=1/att_IJ.size()[1] 
        att_IJ = torch.where(abs(att_IJ-rowmean)<1e-9,torch.tensor(0.).to(att_IJ.device),att_IJ)        
        return att_IJ


class selfattenlayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(selfattenlayer, self).__init__()
        self.atten_ItoJ = nn.Conv1d(hidden_dim, 1, 1)
        self.atten_JtoI = nn.Conv1d(hidden_dim, 1, 1)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.02)
        self.softmax = torch.nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.atten_ItoJ.weight, gain=1.414)
        nn.init.xavier_normal_(self.atten_JtoI.weight, gain=1.414)

    def forward(self, node_mlp, adj_mat):
        X_IJ = torch.unsqueeze(torch.transpose(node_mlp, 0, 1), 0)  # shape  (1,n_emb,num_nodes)
        ### a*[h_i||h_j]=a_i*h_i + a_j*h_j
        f_ItoJ = self.atten_ItoJ(X_IJ)  # shape (1,1,num_nodes), a_i*h_i
        f_JtoI = self.atten_JtoI(X_IJ)  # shape (1,1,num_nodes), a_j*h_j
        edge_logits = f_ItoJ + torch.transpose(f_JtoI, 2, 1)  # shape (1,num_nodes,num_nodes)
        edge_logits = torch.squeeze(edge_logits)  # from (1,num_nodes,num_nodes) to (num_nodes,num_nodes)
        ###bias mat is 0 if adj>0, -1e9 if adj=0
        edge_logits = self.leakyrelu(edge_logits)
        ###softmax only among neighborhood
        zero_vec = -1e9 * torch.ones_like(edge_logits)
        ##if adj>0, keep edge_logits; if adj=0, replace by -1e9
        att_IJ = torch.where(adj_mat > 0, edge_logits, zero_vec)
        att_IJ = self.softmax(att_IJ)
        return att_IJ


class graphNetworkEmbbed(torch.nn.Module):
    def __init__(self, nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, hidden_dim,
                 dropout):
        super(graphNetworkEmbbed, self).__init__()

        self.mlp_A = nn.Linear(nodeA_feature_num, hidden_dim)
        self.mlp_B = nn.Linear(nodeB_feature_num, hidden_dim)
        self.mlp_C = nn.Linear(nodeC_feature_num, hidden_dim)

        self.attAB_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attBA_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attBC_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attCB_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attAC_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attCA_adj = biattenlayer(hidden_dim, dropout=dropout)
        
        self.emblayerAB = nn.Linear(2 * hidden_dim, hidden_dim)
        self.emblayerBA = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.emblayerBC = nn.Linear(2 * hidden_dim, hidden_dim)
        self.emblayerCB = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.emblayerAC = nn.Linear(2 * hidden_dim, hidden_dim)
        self.emblayerCA = nn.Linear(2 * hidden_dim, hidden_dim)

        self.mlp_sim_A = nn.Linear(nodeA_num, hidden_dim)
        self.mlp_sim_B = nn.Linear(nodeB_num, hidden_dim)

        self.attA_sim = selfattenlayer(nodeA_feature_num, hidden_dim, dropout=dropout)
        self.attB_sim = selfattenlayer(nodeB_feature_num, hidden_dim, dropout=dropout)
        self.nodeA_emb = nn.Linear(hidden_dim * 4, hidden_dim)
        self.nodeB_emb = nn.Linear(hidden_dim * 4, hidden_dim)
        self.nodeC_emb = nn.Linear(hidden_dim * 3, hidden_dim)

        self.dropout = dropout
        self.reset_parameters()   

    def reset_parameters(self):
        nn.init.xavier_normal_(self.mlp_A.weight, gain=1.414)
        nn.init.xavier_normal_(self.mlp_B.weight, gain=1.414)
        nn.init.xavier_normal_(self.mlp_C.weight, gain=1.414)

        nn.init.xavier_normal_(self.emblayerAB.weight, gain=1.414)
        nn.init.xavier_normal_(self.emblayerBA.weight, gain=1.414)

        nn.init.xavier_normal_(self.emblayerBC.weight, gain=1.414)
        nn.init.xavier_normal_(self.emblayerCB.weight, gain=1.414)

        nn.init.xavier_normal_(self.emblayerAC.weight, gain=1.414)
        nn.init.xavier_normal_(self.emblayerCA.weight, gain=1.414)

        nn.init.xavier_normal_(self.mlp_sim_A.weight, gain=1.414)
        nn.init.xavier_normal_(self.mlp_sim_B.weight, gain=1.414)

        nn.init.xavier_normal_(self.nodeA_emb.weight, gain=1.414)
        nn.init.xavier_normal_(self.nodeB_emb.weight, gain=1.414)
        nn.init.xavier_normal_(self.nodeC_emb.weight, gain=1.414)

    def forward(self, adj_AB, adj_AC, adj_BC, nodeA_feature, nodeB_feature, nodeC_feature,
                adj_A_sim, adj_B_sim):
        nodeA_mlp = F.relu(self.mlp_A(nodeA_feature))
        nodeA_mlp = F.dropout(nodeA_mlp, self.dropout, training=self.training)
        nodeB_mlp = F.relu(self.mlp_B(nodeB_feature))
        nodeB_mlp = F.dropout(nodeB_mlp, self.dropout, training=self.training)
        nodeC_mlp = F.relu(self.mlp_C(nodeC_feature))
        nodeC_mlp = F.dropout(nodeC_mlp, self.dropout, training=self.training)
        
        nodeA_feature_from_sim = F.relu(self.mlp_sim_A(adj_A_sim)) # biasmat(adj_A_sim)
        nodeB_feature_from_sim = F.relu(self.mlp_sim_B(adj_B_sim))

        att_AB = self.attAB_adj(nodeA_mlp, nodeB_mlp, adj_AB)
        att_BA = self.attBA_adj(nodeB_mlp, nodeA_mlp, adj_AB.T)

        nodeA_feature_from_nodeB = F.relu(self.emblayerAB(torch.cat((nodeA_mlp,torch.mm(att_AB, nodeB_mlp)),1)))

        nodeB_feature_from_nodeA = F.relu(self.emblayerBA(torch.cat((nodeB_mlp,torch.mm(att_BA, nodeA_mlp)),1)))

        att_BC = self.attBC_adj(nodeB_mlp, nodeC_mlp, adj_BC)
        att_CB = self.attCB_adj(nodeC_mlp, nodeB_mlp, adj_BC.T)
        nodeB_feature_from_nodeC = F.relu(self.emblayerBC(torch.cat((nodeB_mlp,torch.mm(att_BC, nodeC_mlp)),1)))
        nodeC_feature_from_nodeB = F.relu(self.emblayerCB(torch.cat((nodeC_mlp,torch.mm(att_CB, nodeB_mlp)),1)))
        
        att_AC = self.attAC_adj(nodeA_mlp, nodeC_mlp, adj_AC)
        att_CA = self.attCA_adj(nodeC_mlp, nodeA_mlp, adj_AC.T)

        nodeA_feature_from_nodeC = F.relu(self.emblayerBC(torch.cat((nodeA_mlp,torch.mm(att_AC, nodeC_mlp)),1)))
        nodeC_feature_from_nodeA = F.relu(self.emblayerCB(torch.cat((nodeC_mlp,torch.mm(att_CA, nodeA_mlp)),1)))
        
        nodeA_emb = F.relu(
            self.nodeA_emb(torch.cat((nodeA_feature_from_sim, nodeA_feature_from_nodeB, nodeA_feature_from_nodeC, nodeA_mlp), 1)))# 230808 add nodeA_mlp
        nodeB_emb = F.relu(
            self.nodeB_emb(torch.cat((nodeB_feature_from_sim, nodeB_feature_from_nodeC, nodeB_feature_from_nodeA, nodeB_mlp), 1)))

        nodeC_emb = F.relu(self.nodeC_emb(torch.cat((nodeC_feature_from_nodeA, nodeC_feature_from_nodeB, nodeC_mlp), 1)))

        return nodeA_emb, nodeB_emb, nodeC_emb


class MlpDecoder(torch.nn.Module):

    def __init__(self, input_dim,edgetype):
        super(MlpDecoder, self).__init__()
        if edgetype=='concat':
            self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim * 2), int(input_dim)),
                                   nn.ReLU())
        else:
            self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim), int(input_dim)),
                                   nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim), int(input_dim // 2)),
                                   nn.ReLU())
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, nodeI_feature, nodeJ_feature,edgetype):
        if edgetype == 'concat':
            pair_feature = torch.cat([nodeI_feature, nodeJ_feature], 1)
        elif edgetype == 'L1':
            pair_feature = torch.abs(nodeI_feature- nodeJ_feature)
        elif edgetype == 'L2':
            pair_feature = torch.square(nodeI_feature- nodeJ_feature)     
        elif edgetype == 'had':
            pair_feature = torch.mul(nodeI_feature,nodeJ_feature)
        elif edgetype == 'mean':
            pair_feature = torch.add(nodeI_feature,nodeJ_feature) / 2   
            
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        outputs = self.mlp_3(embedding_2)
        return outputs


def biasmat(adj_IJ_expand):
    mt = (adj_IJ_expand > 0) * 1
    bias_mat = -1e9 * (1.0 - mt)
    bias_mat = torch.from_numpy(bias_mat).float()
    return bias_mat


def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    # protein: nodeA; Metabolite: nodeB; GO: nodeC

    adj_B_sim = torch.Tensor(np.float16(sparse.load_npz('./dataset/metabolitesim.npz').todense()))
    adj_BC = np.int16(sparse.load_npz('./dataset/metaboliteGO.npz').todense())
    adj_A_sim = torch.Tensor(np.int16(sparse.load_npz('./dataset/proteinPPI.npz').todense()))
    adj_AC = np.int16(sparse.load_npz('./dataset/proteinGO.npz').todense())
    adj_AB = np.int16(sparse.load_npz('./dataset/proteinMetabolite.npz').todense())
    
    metaboliteList = readListfile('./dataset/metaboliteList.txt')
    proteinList = readListfile('./dataset/proteinList.txt')
    GOList = readListfile('./dataset/GOList.txt')

    f = open('./result/HNCGAT_result'+str(date.today())+'.txt', 'a')
    f.write('{}'.format(args))
    f.write('\n')
    f.flush()

    nodeB_num = len(metaboliteList)
    nodeA_num = len(proteinList)
    nodeC_num = len(GOList)

    pos_u, pos_v = np.where(adj_AB != 0)
    neg_u, neg_v = np.where(adj_AB == 0)

    negative_ratio = 10
    negative_sample_index = np.random.choice(np.arange(len(neg_u)), size=negative_ratio * len(pos_u), replace=False)

    pos_data_set = np.zeros((len(pos_u), 3), dtype=int)
    neg_data_set = np.zeros((len(negative_sample_index), 3), dtype=int)

    for i in range(len(pos_u)):
        pos_data_set[i][0] = pos_u[i]
        pos_data_set[i][1] = pos_v[i]
        pos_data_set[i][2] = 1
    count = 0
    for i in negative_sample_index:
        neg_data_set[count][0] = neg_u[i]
        neg_data_set[count][1] = neg_v[i]
        neg_data_set[count][2] = 0
        count = count + 1
    hidden_dim = args.hid_dim
    dropout = args.dropout
    lamb = args.lamb
    lr = args.lr
    edgetype=args.edgetype
    tau=args.temperature
    
    nodeA_feature_num = hidden_dim
    nodeB_feature_num = hidden_dim
    nodeC_feature_num = hidden_dim


    adj_AC=torch.Tensor(adj_AC)
    adj_BC=torch.Tensor(adj_BC)

    trp_varied = [0.9]  # 训练比例：90%
    AUC_ROCAll = []
    AUCstdALL = []
    AUC_PRAll = []
    AUC_PRstdALL = []
    for train_ratio in trp_varied:
        val_ratio = 0
        test_ratio = 1 - train_ratio - val_ratio
        numRandom = 5
        AUC_ROCtrp = []
        AUC_PRtrp = []                  
            
        for random_state in range(numRandom):
            print("%d-th random split with training ratio %f" % (random_state + 1, train_ratio))
            modelfilename ="./model/"+ 'rand' + str(random_state) + 'trp' + str(train_ratio) + '_best_HNCGAT_model.pkl'                

            pos_idx_train, pos_idx_val, pos_idx_test, pos_y_train, pos_y_val, pos_y_test, pos_train_mask, pos_val_mask, pos_test_mask = get_train_index(
                pos_u, train_ratio, val_ratio, test_ratio, numRandom, random_state)
            neg_idx_train, neg_idx_val, neg_idx_test, neg_y_train, neg_y_val, neg_y_test, neg_train_mask, neg_val_mask, neg_test_mask = get_train_index(
                negative_sample_index, train_ratio, val_ratio, test_ratio, numRandom, random_state)
            train_adj_AB = np.zeros((nodeA_num, nodeB_num), dtype=int)
            for i in pos_idx_train:
                idxi = pos_data_set[i, 0]
                idxj = pos_data_set[i, 1]
                train_adj_AB[idxi][idxj] = 1

            train_mask = np.array(np.concatenate((pos_train_mask, neg_train_mask), 0), dtype=np.bool)
            train_mask = torch.tensor(train_mask)
            test_mask = np.array(np.concatenate((pos_test_mask, neg_test_mask), 0), dtype=np.bool)
            test_mask = torch.tensor(test_mask)
            data_set = np.concatenate((pos_data_set, neg_data_set), 0)
            data_set = torch.tensor(data_set).long()

            train_adj_AB=torch.Tensor(train_adj_AB)
            
            model = MPINet(nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, hidden_dim,
                           dropout, edgetype, 
                           use_weighted_loss=args.use_weighted_loss,
                           use_attention=args.weight_attention,
                           tau=tau,
                           weight_temperature=args.weight_temperature,
                           use_diffusion=args.use_diffusion,
                           diff_K=args.diffusion_K,
                           diff_alpha=args.diffusion_alpha,
                           diff_beta=args.diffusion_beta)
            
            # Determine device
            device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
            cost_sensitive=torch.Tensor([10.]).to(device)
            
            loss_func = torch.nn.BCELoss(reduction='mean',weight=cost_sensitive)
            
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

            test_AUC_ROC_list = []
            test_AUCpr_list = []
            test_AUC_AUCpr_best = 0
            test_AUC_best = 0
            test_AUCpr_best = 0
            best_epoch = 0
            nodeA_feature = torch.rand(nodeA_num, nodeA_feature_num, requires_grad=True)
            nodeB_feature = torch.rand(nodeB_num, nodeB_feature_num, requires_grad=True)
            nodeC_feature = torch.rand(nodeC_num, nodeC_feature_num, requires_grad=True)
            # Move model and data to device
            model = model.to(device)
            data_set = data_set.to(device)
            nodeA_feature = nodeA_feature.to(device)
            nodeB_feature = nodeB_feature.to(device)
            nodeC_feature = nodeC_feature.to(device)
            adj_A_sim = adj_A_sim.to(device)
            adj_B_sim = adj_B_sim.to(device)
            adj_AC = adj_AC.to(device)
            adj_BC = adj_BC.to(device)
            train_adj_AB = train_adj_AB.to(device)                
                
            for epoch in range(args.n_epochs):

                model.train()
                opt.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                prob,embA,embB,embC = model(data_set, train_adj_AB, adj_AC, adj_BC, nodeA_feature, nodeB_feature,
                             nodeC_feature, adj_A_sim, adj_B_sim,edgetype)
                label = data_set[:, 2].float()
                
                train_auc, _ = calculateauc(prob[train_mask], data_set[:, 2][train_mask])
                
                # 使用加权损失或原始损失
                if model.use_weighted_loss:
                    P_loss, P_weights = model.weighted_P_loss(embA, embB, embC, train_adj_AB, adj_A_sim, adj_AC)
                    M_loss, M_weights = model.weighted_M_loss(embB, embA, embC, train_adj_AB.T, adj_B_sim, adj_BC)
                    F_loss, F_weights = model.weighted_F_loss(embC, embB, embA, adj_BC.T, adj_AC.T)
                    conloss = (P_loss + M_loss + F_loss) / 3
                    
                    # 可选：每100个epoch打印权重信息（用于分析）
                    if epoch % 100 == 0 and epoch > 0:
                        print(f"Epoch {epoch} - Average weights - PM:{P_weights[:, 0].mean():.3f}, "
                              f"PP:{P_weights[:, 1].mean():.3f}, PF:{P_weights[:, 2].mean():.3f}")
                else:
                    conloss = multi_contrastive_loss(embA, embB, embC, adj_A_sim, adj_B_sim, train_adj_AB, adj_AC, adj_BC, tau)
                
                loss = loss_func(prob[train_mask], label[train_mask]) + lamb * conloss 
                loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    logits,_,_,_ = model(data_set, train_adj_AB, adj_AC, adj_BC,
                                   nodeA_feature, nodeB_feature, nodeC_feature, adj_A_sim, adj_B_sim,edgetype)

                    logits = logits[test_mask]
                    label = data_set[:, 2][test_mask]
                    testAUC_ROC, testAUCpr = calculateauc(logits, label)
                    if test_AUC_AUCpr_best <= (testAUC_ROC + testAUCpr):
                        torch.save(model.state_dict(), modelfilename)
                        test_AUC_AUCpr_best = (testAUC_ROC + testAUCpr)
                        test_AUC_best = testAUC_ROC
                        test_AUCpr_best = testAUCpr
                        best_epoch = epoch

                test_AUC_ROC_list.append(testAUC_ROC)
                test_AUCpr_list.append(testAUCpr)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print("Epoch {:03d} | Loss {:.4f} | TrainAUC {:.4f} |"
                          " testAUC_ROC {:.4f} |  testAUCpr {:.4f} ".
                          format(epoch + 1, loss.item(), train_auc,
                                 testAUC_ROC, testAUCpr))

            AUC_ROCtrp.append(test_AUC_best)
            AUC_PRtrp.append(test_AUCpr_best)

            model.load_state_dict(torch.load(modelfilename))
            model.eval()
            with torch.no_grad():
                logits,_,_,_ = model(data_set, train_adj_AB, adj_AC, adj_BC,
                               nodeA_feature, nodeB_feature, nodeC_feature, adj_A_sim, adj_B_sim,edgetype)

                logits = logits[test_mask]
                label = data_set[:, 2][test_mask]
                testAUC_ROC, testAUCpr = calculateauc(logits, label)
                print("Load model result: Epoch {:03d}"
                      " testAUC_ROC {:.4f} |  testAUCpr {:.4f} ".
                      format(best_epoch + 1, testAUC_ROC, testAUCpr))

        AUC_ROCAll.append(np.mean(AUC_ROCtrp))
        AUCstdALL.append(np.std(AUC_ROCtrp))
        AUC_PRAll.append(np.mean(AUC_PRtrp))
        AUC_PRstdALL.append(np.std(AUC_PRtrp))

        f.write('avg AUC_ROC: %f + %f, for trp:%.2f \n' % (np.mean(AUC_ROCtrp), np.std(AUC_ROCtrp), train_ratio))
        f.write('avg  AUC_PRtrp: %f + %f, for trp:%.2f \n' % (np.mean(AUC_PRtrp), np.std(AUC_PRtrp), train_ratio))

        f.flush()

    print('AUC_ROCAll: \n')
    f.write('AUC_ROCAll:')
    for auc in AUC_ROCAll:
        print('%f \n' % (auc))
        f.write('%f,' % (auc))
    f.write('\n' + 'AUCstd:')
    for aucstd in AUCstdALL:
        print('%f \n' % (aucstd))
        f.write('%f,' % (aucstd))

    f.write('\n' + 'AUC_PRAll:')
    print('AUC_PRAll: \n')
    for AUC_PR in AUC_PRAll:
        print('%f \n' % (AUC_PR))
        f.write('%f,' % (AUC_PR))
    f.write('\n' + 'AUC_PRstd:')
    for AUC_PRstd in AUC_PRstdALL:
        print('%f \n' % (AUC_PRstd))
        f.write('%f,' % (AUC_PRstd))

    f.write('\n')
    f.flush()

    f.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='HNCGAT')

    parser.add_argument('--gpu', type=int, default=1, help='GPU index. Default: -1, using CPU.')
   
    parser.add_argument('--n-epochs', type=int, default=1000, help='Training epochs.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=1, help='conloss weight.')
   
    parser.add_argument("--hid-dim", type=int, default=64, help='Hidden layer dimensionalities.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
    parser.add_argument('--edgetype', type=str, default='concat', help='edgetype')
    parser.add_argument('--use-weighted-loss', action='store_true', 
                        help='Use weighted contrastive loss (addresses paper limitations)')
    parser.add_argument('--weight-attention', action='store_true', default=True,
                        help='Use attention mechanism for learning weights (default: True)')
    parser.add_argument('--weight-temperature', type=float, default=0.7,
                        help='Temperature parameter for weight softmax (default: 0.7). Lower values make weights more uniform, higher values allow more concentration.')
    # 扩散图相关超参数
    parser.add_argument('--use-diffusion', action='store_true',
                        help='Use diffusion-enhanced similarity graphs for proteins and metabolites.')
    parser.add_argument('--diffusion-K', type=int, default=3,
                        help='Number of diffusion steps K for GraphDiffusion.')
    parser.add_argument('--diffusion-alpha', type=float, default=0.2,
                        help='Decay factor alpha for higher-order diffusion powers.')
    parser.add_argument('--diffusion-beta', type=float, default=0.5,
                        help='Fusion weight for meta-graphs built from heterogeneous edges.')
    args = parser.parse_args()
    print(args)
    warnings.filterwarnings("ignore")
    main(args)                                              
