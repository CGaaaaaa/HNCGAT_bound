# -*- coding: utf-8 -*-
"""
边特征增强的MLP解码器

创新点：
1. 在原有的node embedding基础上，加入边特征（edge features）
2. 边特征包括：
   - 共同功能注释数量（common GO terms）
   - 共同邻居数量（common neighbors）
   - 扩散图上的相似度（diffusion similarity）
3. 将node embedding和edge features拼接后送入MLP，提升边预测性能

理论依据：
- SEAL (Learning from Local Structure for Link Prediction, NeurIPS 2018) 
  提出了从边周围的局部子图结构中提取特征来预测链接
- 边级别的特征能够捕捉到纯node embedding无法表达的"关系特有信息"
- 在小样本场景下，这些显式的结构特征比纯embedding更可靠

参考文献：
- Zhang & Chen. "Link Prediction Based on Graph Neural Networks" NeurIPS 2018
- You et al. "Design Space for Graph Neural Networks" NeurIPS 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFeatureExtractor(nn.Module):
    """
    边特征提取器：从图结构中提取边级别的特征
    
    特征包括：
    1. 共同功能注释数量（蛋白和代谢物有多少个共同的GO term）
    2. 共同蛋白邻居数量（两个代谢物有多少个共同的蛋白邻居）
    3. 共同代谢物邻居数量（两个蛋白有多少个共同的代谢物邻居）
    4. 扩散图上的相似度（如果使用了扩散）
    5. 原始相似度（PP或MM图上的相似度）
    """
    def __init__(self, use_diffusion=False):
        super(EdgeFeatureExtractor, self).__init__()
        self.use_diffusion = use_diffusion
        # 缓存计算好的特征矩阵，避免重复计算
        self._cached_features = {}
    
    def compute_common_neighbors(self, adj_AB, node_i_indices, node_j_indices):
        """
        计算共同邻居数量
        
        Args:
            adj_AB: [N_A, N_B] 邻接矩阵（蛋白-代谢物）
            node_i_indices: [batch_size] 节点i的索引
            node_j_indices: [batch_size] 节点j的索引
        
        Returns:
            common_neighbors: [batch_size] 共同邻居数量
        """
        # 对于蛋白i和蛋白j，计算它们的共同代谢物邻居
        # 或者对于代谢物i和代谢物j，计算它们的共同蛋白邻居
        neighbors_i = adj_AB[node_i_indices]  # [batch_size, N_B]
        neighbors_j = adj_AB[node_j_indices]  # [batch_size, N_B]
        common = (neighbors_i * neighbors_j).sum(1)  # [batch_size]
        return common
    
    def compute_common_annotations(self, adj_AC, adj_BC, protein_indices, metabolite_indices):
        """
        计算蛋白和代谢物的共同功能注释数量
        
        Args:
            adj_AC: [N_A, N_C] 蛋白-GO邻接矩阵
            adj_BC: [N_B, N_C] 代谢物-GO邻接矩阵
            protein_indices: [batch_size] 蛋白索引
            metabolite_indices: [batch_size] 代谢物索引
        
        Returns:
            common_annotations: [batch_size] 共同GO term数量
        """
        protein_gos = adj_AC[protein_indices]  # [batch_size, N_C]
        metabolite_gos = adj_BC[metabolite_indices]  # [batch_size, N_C]
        common = (protein_gos * metabolite_gos).sum(1)  # [batch_size]
        return common
    
    def compute_similarity_features(self, adj_A_sim, adj_B_sim, diff_A_sim, diff_B_sim,
                                   protein_indices, metabolite_indices):
        """
        计算相似度特征（原始相似度 + 扩散相似度）
        
        Returns:
            protein_sim: [batch_size] 蛋白对之间的相似度
            metabolite_sim: [batch_size] 代谢物对之间的相似度
            protein_diff_sim: [batch_size] 蛋白对扩散后的相似度
            metabolite_diff_sim: [batch_size] 代谢物对扩散后的相似度
        """
        # 注意：这里我们提取的是"蛋白i和蛋白j之间的相似度"或"代谢物i和代谢物j之间的相似度"
        # 但在MPI预测中，我们预测的是"蛋白i和代谢物j之间的交互"
        # 所以这里的相似度特征是间接特征，表示"相似的蛋白倾向于和相似的代谢物交互"
        
        # 为了简化，我们可以提取：
        # - 蛋白i在蛋白相似图中的平均相似度（表示该蛋白的"hub"程度）
        # - 代谢物j在代谢物相似图中的平均相似度
        
        protein_avg_sim = adj_A_sim[protein_indices].mean(1)  # [batch_size]
        metabolite_avg_sim = adj_B_sim[metabolite_indices].mean(1)  # [batch_size]
        
        if self.use_diffusion and diff_A_sim is not None and diff_B_sim is not None:
            protein_diff_avg_sim = diff_A_sim[protein_indices].mean(1)
            metabolite_diff_avg_sim = diff_B_sim[metabolite_indices].mean(1)
        else:
            protein_diff_avg_sim = protein_avg_sim
            metabolite_diff_avg_sim = metabolite_avg_sim
        
        return protein_avg_sim, metabolite_avg_sim, protein_diff_avg_sim, metabolite_diff_avg_sim
    
    def extract_edge_features(self, protein_indices, metabolite_indices,
                             adj_AB, adj_AC, adj_BC, adj_A_sim, adj_B_sim,
                             diff_A_sim=None, diff_B_sim=None):
        """
        提取边特征
        
        Args:
            protein_indices: [batch_size] 蛋白索引
            metabolite_indices: [batch_size] 代谢物索引
            adj_AB: 蛋白-代谢物邻接矩阵
            adj_AC: 蛋白-GO邻接矩阵
            adj_BC: 代谢物-GO邻接矩阵
            adj_A_sim: 蛋白相似度矩阵
            adj_B_sim: 代谢物相似度矩阵
            diff_A_sim: 蛋白扩散相似度矩阵（可选）
            diff_B_sim: 代谢物扩散相似度矩阵（可选）
        
        Returns:
            edge_features: [batch_size, num_features] 边特征矩阵
        """
        batch_size = protein_indices.size(0)
        device = protein_indices.device
        
        # 特征1：共同功能注释数量
        common_go = self.compute_common_annotations(adj_AC, adj_BC, protein_indices, metabolite_indices)
        
        # 特征2：蛋白的代谢物邻居数量（度数）
        protein_degree = adj_AB[protein_indices].sum(1)
        
        # 特征3：代谢物的蛋白邻居数量（度数）
        metabolite_degree = adj_AB.T[metabolite_indices].sum(1)
        
        # 特征4-7：相似度特征
        p_sim, m_sim, p_diff_sim, m_diff_sim = self.compute_similarity_features(
            adj_A_sim, adj_B_sim, diff_A_sim, diff_B_sim,
            protein_indices, metabolite_indices
        )
        
        # 拼接所有特征
        edge_features = torch.stack([
            common_go.float(),
            protein_degree.float(),
            metabolite_degree.float(),
            p_sim.float(),
            m_sim.float(),
            p_diff_sim.float(),
            m_diff_sim.float()
        ], dim=1)  # [batch_size, 7]
        
        # 归一化特征（避免不同特征的尺度差异过大）
        edge_features = F.normalize(edge_features, p=2, dim=1)
        
        return edge_features


class EdgeFeatureEnhancedDecoder(nn.Module):
    """
    边特征增强的MLP解码器
    
    在原有的node embedding基础上，加入边特征，提升MPI预测性能
    """
    def __init__(self, input_dim, edgetype, use_edge_features=True, edge_feature_dim=7):
        super(EdgeFeatureEnhancedDecoder, self).__init__()
        self.use_edge_features = use_edge_features
        self.edgetype = edgetype
        
        # 计算输入维度
        if edgetype == 'concat':
            node_feature_dim = int(input_dim * 2)
        else:
            node_feature_dim = int(input_dim)
        
        # 如果使用边特征，输入维度需要加上边特征维度
        if use_edge_features:
            total_input_dim = node_feature_dim + edge_feature_dim
        else:
            total_input_dim = node_feature_dim
        
        # MLP层
        self.mlp_1 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Linear(total_input_dim, int(input_dim)),
            nn.ReLU()
        )
        self.mlp_2 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Linear(int(input_dim), int(input_dim // 2)),
            nn.ReLU()
        )
        self.mlp_3 = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Linear(int(input_dim // 2), 1),
            nn.Sigmoid()
        )
        
        # 边特征提取器
        if use_edge_features:
            self.edge_feature_extractor = EdgeFeatureExtractor(use_diffusion=True)
    
    def forward(self, nodeI_feature, nodeJ_feature, edge_features=None):
        """
        Args:
            nodeI_feature: [batch_size, hidden_dim] 蛋白节点特征
            nodeJ_feature: [batch_size, hidden_dim] 代谢物节点特征
            edge_features: [batch_size, edge_feature_dim] 边特征（可选）
        
        Returns:
            outputs: [batch_size, 1] 预测概率
        """
        # 根据edgetype计算node pair特征
        if self.edgetype == 'concat':
            pair_feature = torch.cat([nodeI_feature, nodeJ_feature], 1)
        elif self.edgetype == 'L1':
            pair_feature = torch.abs(nodeI_feature - nodeJ_feature)
        elif self.edgetype == 'L2':
            pair_feature = torch.square(nodeI_feature - nodeJ_feature)
        elif self.edgetype == 'had':
            pair_feature = torch.mul(nodeI_feature, nodeJ_feature)
        elif self.edgetype == 'mean':
            pair_feature = torch.add(nodeI_feature, nodeJ_feature) / 2
        
        # 如果使用边特征，拼接到pair_feature
        if self.use_edge_features and edge_features is not None:
            pair_feature = torch.cat([pair_feature, edge_features], 1)
        
        # MLP预测
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        outputs = self.mlp_3(embedding_2)
        return outputs

