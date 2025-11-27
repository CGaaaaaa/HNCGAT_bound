# -*- coding: utf-8 -*-
"""
难负样本挖掘增强的加权邻居对比损失

创新点：
1. 在原有加权对比损失基础上，加入难负样本挖掘机制
2. 只选择top-K个"最难"的负样本（即与anchor相似度最高但标签为0的样本）
3. 或者给难负样本更大的权重，使模型更关注容易混淆的样本

理论依据：
- MoCo, SimCLR等对比学习方法都使用了难负样本挖掘
- 在MPI预测中，大部分负样本（非交互对）都是"显然不相关"的，对训练帮助不大
- 只关注"难负样本"能让模型学习更精准的决策边界

参考文献：
- Robinson et al. "Contrastive Learning with Hard Negative Samples" ICLR 2021
- Kalantidis et al. "Hard Negative Mixing for Contrastive Learning" NeurIPS 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import sim


class HardNegativeWeightedPConLoss(nn.Module):
    """
    带难负样本挖掘的自适应加权蛋白质对比损失
    
    Args:
        hidden_dim: 隐藏层维度
        tau: 温度参数
        use_attention: 是否使用注意力机制学习邻居类型权重
        weight_temperature: 权重softmax的温度参数
        hard_negative_ratio: 难负样本比例（0-1之间），0表示使用所有负样本，0.5表示只用最难的50%
        hard_negative_weight: 难负样本的额外权重（>1表示给难负样本更大权重）
    """
    def __init__(self, hidden_dim, tau=0.1, use_attention=True, weight_temperature=0.7,
                 hard_negative_ratio=0.3, hard_negative_weight=2.0):
        super(HardNegativeWeightedPConLoss, self).__init__()
        self.tau = tau
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.weight_temperature = weight_temperature
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_weight = hard_negative_weight
        
        if use_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 3),
                nn.Softmax(dim=-1)
            )
        else:
            self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def compute_neighbor_aggregation(self, anchor_emb, neighbor_emb, adj):
        neighbor_agg = torch.mm(adj, neighbor_emb)
        neighbor_count = adj.sum(1, keepdim=True).clamp(min=1)
        return neighbor_agg / neighbor_count
    
    def select_hard_negatives(self, similarity_matrix, positive_mask):
        """
        选择难负样本
        
        Args:
            similarity_matrix: [N, M] 相似度矩阵
            positive_mask: [N, M] 正样本掩码（1表示正样本，0表示负样本）
        
        Returns:
            hard_negative_mask: [N, M] 难负样本掩码
            negative_weights: [N, M] 负样本权重矩阵
        """
        # 负样本掩码
        negative_mask = (positive_mask == 0).float()
        
        # 对于每个anchor，找出其负样本的相似度
        # 将正样本位置的相似度设为-inf，这样不会被选中
        neg_similarities = similarity_matrix.clone()
        neg_similarities = torch.where(positive_mask > 0, 
                                      torch.tensor(float('-inf')).to(similarity_matrix.device),
                                      neg_similarities)
        
        # 计算每个anchor要选择多少个难负样本
        num_negatives = negative_mask.sum(1)  # [N]
        num_hard_negatives = (num_negatives * self.hard_negative_ratio).long().clamp(min=1)
        
        # 为每个anchor选择top-K难负样本
        hard_negative_mask = torch.zeros_like(negative_mask)
        negative_weights = torch.ones_like(negative_mask)
        
        for i in range(similarity_matrix.size(0)):
            if num_negatives[i] > 0:
                # 获取该anchor的负样本相似度
                neg_sims = neg_similarities[i]
                # 选择top-K个最高相似度的负样本
                k = num_hard_negatives[i].item()
                if k > 0:
                    _, top_indices = torch.topk(neg_sims, k=min(k, int(num_negatives[i].item())))
                    hard_negative_mask[i, top_indices] = 1.0
                    # 给难负样本更大的权重
                    negative_weights[i, top_indices] = self.hard_negative_weight
        
        return hard_negative_mask, negative_weights
    
    def forward(self, embP, embM, embF, PM_adj, PP_adj, PF_adj):
        """
        计算带难负样本挖掘的加权对比损失
        """
        # 聚合三种邻居类型的表示
        PM_repr = self.compute_neighbor_aggregation(embP, embM, PM_adj)
        PP_repr = self.compute_neighbor_aggregation(embP, embP, PP_adj)
        PF_repr = self.compute_neighbor_aggregation(embP, embF, PF_adj)
        
        # 学习邻居类型权重
        if self.use_attention:
            neighbor_features = torch.stack([PM_repr, PP_repr, PF_repr], dim=1)
            raw_weights = self.attention_mlp(neighbor_features.view(embP.size(0), -1))
            weights = F.softmax(raw_weights / self.weight_temperature, dim=-1)
        else:
            weights = F.softmax(self.weights, dim=0).unsqueeze(0).expand(embP.size(0), -1)
        
        # 归一化嵌入
        embP_norm = F.normalize(embP)
        embM_norm = F.normalize(embM)
        embF_norm = F.normalize(embF)
        
        # 计算相似度
        f = lambda x: torch.exp(x / self.tau)
        PM_sim = f(sim(embP_norm, embM_norm))
        PP_sim = f(sim(embP_norm, embP_norm))
        PF_sim = f(sim(embP_norm, embF_norm))
        
        # 正样本对
        PM_positive = (PM_sim * PM_adj).sum(1)
        PP_positive = (PP_sim * PP_adj).sum(1)
        PF_positive = (PF_sim * PF_adj).sum(1)
        
        # 加权正样本对（分子）
        weighted_positive = (weights[:, 0] * PM_positive + 
                            weights[:, 1] * PP_positive + 
                            weights[:, 2] * PF_positive)
        
        # === 难负样本挖掘 ===
        # 对PP和PF进行难负样本挖掘（PM不做，因为PM的负样本是预测目标）
        
        # PP难负样本
        PP_hard_neg_mask, PP_neg_weights = self.select_hard_negatives(PP_sim, PP_adj)
        PP_negative_mask = (PP_adj == 0).float()
        # 只保留难负样本
        PP_weighted_neg = PP_sim * PP_negative_mask * PP_neg_weights
        
        # PF难负样本
        PF_hard_neg_mask, PF_neg_weights = self.select_hard_negatives(PF_sim, PF_adj)
        PF_negative_mask = (PF_adj == 0).float()
        PF_weighted_neg = PF_sim * PF_negative_mask * PF_neg_weights
        
        # 分母：正样本 + 难负样本
        PM_all = PM_positive  # PM只考虑正样本（原始设计）
        PP_all = PP_positive + PP_weighted_neg.sum(1)  # PP：正样本 + 加权难负样本
        PF_all = PF_positive + PF_weighted_neg.sum(1)  # PF：正样本 + 加权难负样本
        
        # 加权分母
        weighted_denominator = (weights[:, 0] * PM_all + 
                               weights[:, 1] * PP_all + 
                               weights[:, 2] * PF_all)
        
        # 计算损失
        nei_count = (PM_adj.sum(1) + PP_adj.sum(1) + PF_adj.sum(1)).clamp(min=1)
        loss = weighted_positive / weighted_denominator.clamp(min=1e-10)
        loss = loss / nei_count
        loss = loss.clamp(min=1e-10)
        
        return (-torch.log(loss)).mean(), weights


class HardNegativeWeightedMConLoss(nn.Module):
    """
    带难负样本挖掘的自适应加权代谢物对比损失
    """
    def __init__(self, hidden_dim, tau=0.1, use_attention=True, weight_temperature=0.7,
                 hard_negative_ratio=0.3, hard_negative_weight=2.0):
        super(HardNegativeWeightedMConLoss, self).__init__()
        self.tau = tau
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.weight_temperature = weight_temperature
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_weight = hard_negative_weight
        
        if use_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 3),
                nn.Softmax(dim=-1)
            )
        else:
            self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def compute_neighbor_aggregation(self, anchor_emb, neighbor_emb, adj):
        neighbor_agg = torch.mm(adj, neighbor_emb)
        neighbor_count = adj.sum(1, keepdim=True).clamp(min=1)
        return neighbor_agg / neighbor_count
    
    def select_hard_negatives(self, similarity_matrix, positive_mask):
        negative_mask = (positive_mask == 0).float()
        neg_similarities = similarity_matrix.clone()
        neg_similarities = torch.where(positive_mask > 0, 
                                      torch.tensor(float('-inf')).to(similarity_matrix.device),
                                      neg_similarities)
        
        num_negatives = negative_mask.sum(1)
        num_hard_negatives = (num_negatives * self.hard_negative_ratio).long().clamp(min=1)
        
        hard_negative_mask = torch.zeros_like(negative_mask)
        negative_weights = torch.ones_like(negative_mask)
        
        for i in range(similarity_matrix.size(0)):
            if num_negatives[i] > 0:
                neg_sims = neg_similarities[i]
                k = num_hard_negatives[i].item()
                if k > 0:
                    _, top_indices = torch.topk(neg_sims, k=min(k, int(num_negatives[i].item())))
                    hard_negative_mask[i, top_indices] = 1.0
                    negative_weights[i, top_indices] = self.hard_negative_weight
        
        return hard_negative_mask, negative_weights
    
    def forward(self, embM, embP, embF, MP_adj, MM_adj, MF_adj):
        MP_repr = self.compute_neighbor_aggregation(embM, embP, MP_adj)
        MM_repr = self.compute_neighbor_aggregation(embM, embM, MM_adj)
        MF_repr = self.compute_neighbor_aggregation(embM, embF, MF_adj)
        
        if self.use_attention:
            neighbor_features = torch.stack([MP_repr, MM_repr, MF_repr], dim=1)
            raw_weights = self.attention_mlp(neighbor_features.view(embM.size(0), -1))
            weights = F.softmax(raw_weights / self.weight_temperature, dim=-1)
        else:
            weights = F.softmax(self.weights, dim=0).unsqueeze(0).expand(embM.size(0), -1)
        
        embM_norm = F.normalize(embM)
        embP_norm = F.normalize(embP)
        embF_norm = F.normalize(embF)
        
        f = lambda x: torch.exp(x / self.tau)
        MP_sim = f(sim(embM_norm, embP_norm))
        MM_sim = f(sim(embM_norm, embM_norm))
        MF_sim = f(sim(embM_norm, embF_norm))
        
        MP_positive = (MP_sim * MP_adj).sum(1)
        MM_positive = (MM_sim * MM_adj).sum(1)
        MF_positive = (MF_sim * MF_adj).sum(1)
        
        weighted_positive = (weights[:, 0] * MP_positive + 
                            weights[:, 1] * MM_positive + 
                            weights[:, 2] * MF_positive)
        
        # 难负样本挖掘
        MM_hard_neg_mask, MM_neg_weights = self.select_hard_negatives(MM_sim, MM_adj)
        MM_negative_mask = (MM_adj == 0).float()
        MM_weighted_neg = MM_sim * MM_negative_mask * MM_neg_weights
        
        MF_hard_neg_mask, MF_neg_weights = self.select_hard_negatives(MF_sim, MF_adj)
        MF_negative_mask = (MF_adj == 0).float()
        MF_weighted_neg = MF_sim * MF_negative_mask * MF_neg_weights
        
        MP_all = MP_positive
        MM_all = MM_positive + MM_weighted_neg.sum(1)
        MF_all = MF_positive + MF_weighted_neg.sum(1)
        
        weighted_denominator = (weights[:, 0] * MP_all + 
                               weights[:, 1] * MM_all + 
                               weights[:, 2] * MF_all)
        
        nei_count = (MP_adj.sum(1) + MM_adj.sum(1) + MF_adj.sum(1)).clamp(min=1)
        loss = weighted_positive / weighted_denominator.clamp(min=1e-10)
        loss = loss / nei_count
        loss = loss.clamp(min=1e-10)
        
        return (-torch.log(loss)).mean(), weights


class HardNegativeWeightedFConLoss(nn.Module):
    """
    带难负样本挖掘的自适应加权功能注释对比损失
    """
    def __init__(self, hidden_dim, tau=0.1, use_attention=True, weight_temperature=0.7,
                 hard_negative_ratio=0.3, hard_negative_weight=2.0):
        super(HardNegativeWeightedFConLoss, self).__init__()
        self.tau = tau
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.weight_temperature = weight_temperature
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_weight = hard_negative_weight
        
        if use_attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )
        else:
            self.weights = nn.Parameter(torch.ones(2) / 2)
    
    def compute_neighbor_aggregation(self, anchor_emb, neighbor_emb, adj):
        neighbor_agg = torch.mm(adj, neighbor_emb)
        neighbor_count = adj.sum(1, keepdim=True).clamp(min=1)
        return neighbor_agg / neighbor_count
    
    def select_hard_negatives(self, similarity_matrix, positive_mask):
        negative_mask = (positive_mask == 0).float()
        neg_similarities = similarity_matrix.clone()
        neg_similarities = torch.where(positive_mask > 0, 
                                      torch.tensor(float('-inf')).to(similarity_matrix.device),
                                      neg_similarities)
        
        num_negatives = negative_mask.sum(1)
        num_hard_negatives = (num_negatives * self.hard_negative_ratio).long().clamp(min=1)
        
        hard_negative_mask = torch.zeros_like(negative_mask)
        negative_weights = torch.ones_like(negative_mask)
        
        for i in range(similarity_matrix.size(0)):
            if num_negatives[i] > 0:
                neg_sims = neg_similarities[i]
                k = num_hard_negatives[i].item()
                if k > 0:
                    _, top_indices = torch.topk(neg_sims, k=min(k, int(num_negatives[i].item())))
                    hard_negative_mask[i, top_indices] = 1.0
                    negative_weights[i, top_indices] = self.hard_negative_weight
        
        return hard_negative_mask, negative_weights
    
    def forward(self, embF, embM, embP, FM_adj, FP_adj):
        FM_repr = self.compute_neighbor_aggregation(embF, embM, FM_adj)
        FP_repr = self.compute_neighbor_aggregation(embF, embP, FP_adj)
        
        if self.use_attention:
            neighbor_features = torch.stack([FM_repr, FP_repr], dim=1)
            raw_weights = self.attention_mlp(neighbor_features.view(embF.size(0), -1))
            weights = F.softmax(raw_weights / self.weight_temperature, dim=-1)
        else:
            weights = F.softmax(self.weights, dim=0).unsqueeze(0).expand(embF.size(0), -1)
        
        embF_norm = F.normalize(embF)
        embM_norm = F.normalize(embM)
        embP_norm = F.normalize(embP)
        
        f = lambda x: torch.exp(x / self.tau)
        FM_sim = f(sim(embF_norm, embM_norm))
        FP_sim = f(sim(embF_norm, embP_norm))
        
        FM_positive = (FM_sim * FM_adj).sum(1)
        FP_positive = (FP_sim * FP_adj).sum(1)
        
        weighted_positive = (weights[:, 0] * FM_positive + 
                            weights[:, 1] * FP_positive)
        
        # 难负样本挖掘
        FM_hard_neg_mask, FM_neg_weights = self.select_hard_negatives(FM_sim, FM_adj)
        FM_negative_mask = (FM_adj == 0).float()
        FM_weighted_neg = FM_sim * FM_negative_mask * FM_neg_weights
        
        FP_hard_neg_mask, FP_neg_weights = self.select_hard_negatives(FP_sim, FP_adj)
        FP_negative_mask = (FP_adj == 0).float()
        FP_weighted_neg = FP_sim * FP_negative_mask * FP_neg_weights
        
        FM_all = FM_positive + FM_weighted_neg.sum(1)
        FP_all = FP_positive + FP_weighted_neg.sum(1)
        
        weighted_denominator = (weights[:, 0] * FM_all + 
                               weights[:, 1] * FP_all)
        
        nei_count = (FM_adj.sum(1) + FP_adj.sum(1)).clamp(min=1)
        loss = weighted_positive / weighted_denominator.clamp(min=1e-10)
        loss = loss / nei_count
        loss = loss.clamp(min=1e-10)
        
        return (-torch.log(loss)).mean(), weights

