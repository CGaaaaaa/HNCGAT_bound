# -*- coding: utf-8 -*-
"""
加权邻居对比损失实现
解决论文Limitations中提到的"所有邻居被平等对待"的问题

改进点：
1. 为不同邻居类型（PM, PP, PF）学习自适应权重
2. 使用注意力机制为每个节点学习特定的权重
3. 解决原论文中所有邻居被平等对待的局限性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import sim  # 复用原有的相似度计算函数


class AdaptiveWeightedPConLoss(nn.Module):
    """
    自适应加权蛋白质对比损失
    为PM（蛋白质-代谢物）、PP（蛋白质-蛋白质）、PF（蛋白质-功能注释）
    三种邻居类型学习节点特定的权重
    
    参考论文Limitations:
    "all the heterogeneous neighbors of the anchor are treated equally 
    in the current neighbor contrastive loss"
    """
    def __init__(self, hidden_dim, tau=0.1, use_attention=True):
        super(AdaptiveWeightedPConLoss, self).__init__()
        self.tau = tau
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        
        if use_attention:
            # 使用注意力机制学习权重（节点特定的权重）
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 3),
                nn.Softmax(dim=-1)
            )
        else:
            # 可学习的全局权重（所有节点共享相同权重）
            self.weights = nn.Parameter(torch.ones(3) / 3)
    
    def compute_neighbor_aggregation(self, anchor_emb, neighbor_emb, adj):
        """
        聚合邻居表示
        """
        # 计算每个节点的邻居聚合表示
        neighbor_agg = torch.mm(adj, neighbor_emb)
        neighbor_count = adj.sum(1, keepdim=True).clamp(min=1)
        return neighbor_agg / neighbor_count
    
    def forward(self, embP, embM, embF, PM_adj, PP_adj, PF_adj):
        """
        计算加权对比损失
        
        Args:
            embP: 蛋白质节点嵌入 [N_P, hidden_dim]
            embM: 代谢物节点嵌入 [N_M, hidden_dim]
            embF: 功能注释节点嵌入 [N_F, hidden_dim]
            PM_adj: 蛋白质-代谢物邻接矩阵
            PP_adj: 蛋白质-蛋白质邻接矩阵
            PF_adj: 蛋白质-功能注释邻接矩阵
        
        Returns:
            loss: 加权对比损失
            weights: 学习到的权重 [N_P, 3] (用于分析和可视化)
        """
        # 聚合三种邻居类型的表示
        PM_repr = self.compute_neighbor_aggregation(embP, embM, PM_adj)
        PP_repr = self.compute_neighbor_aggregation(embP, embP, PP_adj)
        PF_repr = self.compute_neighbor_aggregation(embP, embF, PF_adj)
        
        # 学习权重
        if self.use_attention:
            # 节点特定的权重：每个节点学习自己的权重分配
            neighbor_features = torch.stack([PM_repr, PP_repr, PF_repr], dim=1)  # [N_P, 3, hidden_dim]
            weights = self.attention_mlp(neighbor_features.view(embP.size(0), -1))  # [N_P, 3]
        else:
            # 全局权重：所有节点共享相同的权重
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
        
        # 加权正样本对
        PM_positive = (PM_sim * PM_adj).sum(1)  # [N_P]
        PP_positive = (PP_sim * PP_adj).sum(1)
        PF_positive = (PF_sim * PF_adj).sum(1)
        
        weighted_positive = (weights[:, 0] * PM_positive + 
                            weights[:, 1] * PP_positive + 
                            weights[:, 2] * PF_positive)
        
        # 负样本对（所有非邻居）
        PM_negative = PM_sim.sum(1) - PM_positive
        PP_negative = PP_sim.sum(1) - PP_positive
        PF_negative = PF_sim.sum(1) - PF_positive
        
        weighted_negative = (weights[:, 0] * PM_negative + 
                            weights[:, 1] * PP_negative + 
                            weights[:, 2] * PF_negative)
        
        # 计算损失
        nei_count = (PM_adj.sum(1) + PP_adj.sum(1) + PF_adj.sum(1)).clamp(min=1)
        loss = weighted_positive / (weighted_positive + weighted_negative)
        loss = loss / nei_count
        loss = loss.clamp(min=1e-10)  # 避免log(0)
        
        return (-torch.log(loss)).mean(), weights


class AdaptiveWeightedMConLoss(nn.Module):
    """
    自适应加权代谢物对比损失
    为MP（代谢物-蛋白质）、MM（代谢物-代谢物）、MF（代谢物-功能注释）
    三种邻居类型学习权重
    """
    def __init__(self, hidden_dim, tau=0.1, use_attention=True):
        super(AdaptiveWeightedMConLoss, self).__init__()
        self.tau = tau
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        
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
    
    def forward(self, embM, embP, embF, MP_adj, MM_adj, MF_adj):
        # 聚合邻居表示
        MP_repr = self.compute_neighbor_aggregation(embM, embP, MP_adj)
        MM_repr = self.compute_neighbor_aggregation(embM, embM, MM_adj)
        MF_repr = self.compute_neighbor_aggregation(embM, embF, MF_adj)
        
        # 学习权重
        if self.use_attention:
            neighbor_features = torch.stack([MP_repr, MM_repr, MF_repr], dim=1)
            weights = self.attention_mlp(neighbor_features.view(embM.size(0), -1))
        else:
            weights = F.softmax(self.weights, dim=0).unsqueeze(0).expand(embM.size(0), -1)
        
        # 归一化
        embM_norm = F.normalize(embM)
        embP_norm = F.normalize(embP)
        embF_norm = F.normalize(embF)
        
        # 计算相似度
        f = lambda x: torch.exp(x / self.tau)
        MP_sim = f(sim(embM_norm, embP_norm))
        MM_sim = f(sim(embM_norm, embM_norm))
        MF_sim = f(sim(embM_norm, embF_norm))
        
        # 加权正样本
        MP_positive = (MP_sim * MP_adj).sum(1)
        MM_positive = (MM_sim * MM_adj).sum(1)
        MF_positive = (MF_sim * MF_adj).sum(1)
        weighted_positive = (weights[:, 0] * MP_positive + 
                            weights[:, 1] * MM_positive + 
                            weights[:, 2] * MF_positive)
        
        # 负样本
        MP_negative = MP_sim.sum(1) - MP_positive
        MM_negative = MM_sim.sum(1) - MM_positive
        MF_negative = MF_sim.sum(1) - MF_positive
        weighted_negative = (weights[:, 0] * MP_negative + 
                            weights[:, 1] * MM_negative + 
                            weights[:, 2] * MF_negative)
        
        # 计算损失
        nei_count = (MP_adj.sum(1) + MM_adj.sum(1) + MF_adj.sum(1)).clamp(min=1)
        loss = weighted_positive / (weighted_positive + weighted_negative)
        loss = loss / nei_count
        loss = loss.clamp(min=1e-10)
        
        return (-torch.log(loss)).mean(), weights


class AdaptiveWeightedFConLoss(nn.Module):
    """
    自适应加权功能注释对比损失
    为FM（功能注释-代谢物）、FP（功能注释-蛋白质）
    两种邻居类型学习权重
    """
    def __init__(self, hidden_dim, tau=0.1, use_attention=True):
        super(AdaptiveWeightedFConLoss, self).__init__()
        self.tau = tau
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        
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
    
    def forward(self, embF, embM, embP, FM_adj, FP_adj):
        # 聚合邻居表示
        FM_repr = self.compute_neighbor_aggregation(embF, embM, FM_adj)
        FP_repr = self.compute_neighbor_aggregation(embF, embP, FP_adj)
        
        # 学习权重
        if self.use_attention:
            neighbor_features = torch.stack([FM_repr, FP_repr], dim=1)
            weights = self.attention_mlp(neighbor_features.view(embF.size(0), -1))
        else:
            weights = F.softmax(self.weights, dim=0).unsqueeze(0).expand(embF.size(0), -1)
        
        # 归一化
        embF_norm = F.normalize(embF)
        embM_norm = F.normalize(embM)
        embP_norm = F.normalize(embP)
        
        # 计算相似度
        f = lambda x: torch.exp(x / self.tau)
        FM_sim = f(sim(embF_norm, embM_norm))
        FP_sim = f(sim(embF_norm, embP_norm))
        
        # 加权正样本
        FM_positive = (FM_sim * FM_adj).sum(1)
        FP_positive = (FP_sim * FP_adj).sum(1)
        weighted_positive = (weights[:, 0] * FM_positive + 
                            weights[:, 1] * FP_positive)
        
        # 负样本
        FM_negative = FM_sim.sum(1) - FM_positive
        FP_negative = FP_sim.sum(1) - FP_positive
        weighted_negative = (weights[:, 0] * FM_negative + 
                            weights[:, 1] * FP_negative)
        
        # 计算损失
        nei_count = (FM_adj.sum(1) + FP_adj.sum(1)).clamp(min=1)
        loss = weighted_positive / (weighted_positive + weighted_negative)
        loss = loss / nei_count
        loss = loss.clamp(min=1e-10)
        
        return (-torch.log(loss)).mean(), weights


def multi_contrastive_loss_weighted(embP, embM, embF, PP_adj, MM_adj, PM_adj, PF_adj, MF_adj, 
                                     tau, hidden_dim, use_weighted=True, use_attention=True):
    """
    加权多视图对比损失
    
    Args:
        embP, embM, embF: 节点嵌入
        PP_adj, MM_adj, PM_adj, PF_adj, MF_adj: 邻接矩阵
        tau: 温度参数
        hidden_dim: 隐藏层维度
        use_weighted: 是否使用加权损失
        use_attention: 是否使用注意力机制学习权重
    
    Returns:
        loss: 总对比损失
        weights: 学习到的权重元组 (P_weights, M_weights, F_weights) 或 None
    """
    if use_weighted:
        # 创建加权损失模块（需要在forward中动态创建，因为需要是nn.Module）
        # 这里我们直接调用，但实际使用时需要在模型中定义
        P_loss_module = AdaptiveWeightedPConLoss(hidden_dim, tau, use_attention)
        M_loss_module = AdaptiveWeightedMConLoss(hidden_dim, tau, use_attention)
        F_loss_module = AdaptiveWeightedFConLoss(hidden_dim, tau, use_attention)
        
        P_loss, P_weights = P_loss_module(embP, embM, embF, PM_adj, PP_adj, PF_adj)
        M_loss, M_weights = M_loss_module(embM, embP, embF, PM_adj.T, MM_adj, MF_adj)
        F_loss, F_weights = F_loss_module(embF, embM, embP, MF_adj.T, PF_adj.T)
        
        return (P_loss + M_loss + F_loss) / 3, (P_weights, M_weights, F_weights)
    else:
        # 使用原始损失
        from loss import multi_contrastive_loss
        return multi_contrastive_loss(embP, embM, embF, PP_adj, MM_adj, PM_adj, PF_adj, MF_adj, tau), None

