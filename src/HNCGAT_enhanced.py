# -*- coding: utf-8 -*-
"""
增强版HNCGAT：整合难负样本挖掘 + 边特征增强

创新点总结：
1. 基础：加权邻居对比损失（已有）+ 图扩散（已有）
2. 新增：难负样本挖掘（Hard Negative Mining）
3. 新增：边特征增强（Edge Feature Augmentation）

使用方法：
python HNCGAT_enhanced.py --use-hard-negative --use-edge-features --gpu 0

@author: enhanced by AI assistant
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
from loss_hard_negative import HardNegativeWeightedPConLoss, HardNegativeWeightedMConLoss, HardNegativeWeightedFConLoss
from decoder_edge_feature import EdgeFeatureEnhancedDecoder, EdgeFeatureExtractor
from datetime import date

# 导入原有的编码器模块
from HNCGAT import GraphDiffusion, DiffusionHNCGATEncoder, biattenlayer, selfattenlayer, graphNetworkEmbbed


class EnhancedMPINet(torch.nn.Module):
    """
    增强版MPI预测网络：整合难负样本挖掘 + 边特征增强
    """
    def __init__(self, nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, hidden_dim,
                 dropout, edgetype, 
                 use_weighted_loss=False, 
                 use_attention=True, 
                 tau=0.1, 
                 weight_temperature=0.7,
                 use_diffusion=False, 
                 diff_K=3, 
                 diff_alpha=0.2, 
                 diff_beta=0.5,
                 use_hard_negative=False,
                 hard_negative_ratio=0.3,
                 hard_negative_weight=2.0,
                 use_edge_features=False):
        super(EnhancedMPINet, self).__init__()
        
        # 编码器（使用原有的扩散增强编码器）
        self.encoder_1 = DiffusionHNCGATEncoder(
            nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num,
            nodeC_feature_num, hidden_dim, dropout,
            use_diffusion=use_diffusion,
            diff_K=diff_K,
            diff_alpha=diff_alpha,
            diff_beta=diff_beta
        )
        
        # 解码器（选择是否使用边特征增强）
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.decoder = EdgeFeatureEnhancedDecoder(hidden_dim, edgetype, use_edge_features=True)
        else:
            # 使用原有的MLP解码器
            from HNCGAT import MlpDecoder
            self.decoder = MlpDecoder(hidden_dim, edgetype)
        
        # 对比损失模块（选择是否使用难负样本挖掘）
        self.use_weighted_loss = use_weighted_loss
        self.use_hard_negative = use_hard_negative
        
        if use_weighted_loss:
            if use_hard_negative:
                # 使用难负样本挖掘的加权损失
                self.weighted_P_loss = HardNegativeWeightedPConLoss(
                    hidden_dim, tau, use_attention, weight_temperature,
                    hard_negative_ratio, hard_negative_weight
                )
                self.weighted_M_loss = HardNegativeWeightedMConLoss(
                    hidden_dim, tau, use_attention, weight_temperature,
                    hard_negative_ratio, hard_negative_weight
                )
                self.weighted_F_loss = HardNegativeWeightedFConLoss(
                    hidden_dim, tau, use_attention, weight_temperature,
                    hard_negative_ratio, hard_negative_weight
                )
            else:
                # 使用普通的加权损失
                self.weighted_P_loss = AdaptiveWeightedPConLoss(hidden_dim, tau, use_attention, weight_temperature)
                self.weighted_M_loss = AdaptiveWeightedMConLoss(hidden_dim, tau, use_attention, weight_temperature)
                self.weighted_F_loss = AdaptiveWeightedFConLoss(hidden_dim, tau, use_attention, weight_temperature)
    
    def forward(self, data_set, adj_AB, adj_AC, adj_BC, nodeA_feature, nodeB_feature, nodeC_feature, 
                adj_A_sim, adj_B_sim, edgetype, diff_A_sim=None, diff_B_sim=None):
        """
        Args:
            diff_A_sim, diff_B_sim: 扩散后的相似度矩阵（如果使用边特征，需要传入）
        """
        # 编码
        nodeA_feature, nodeB_feature, nodeC_feature = self.encoder_1(
            adj_AB, adj_AC, adj_BC,
            nodeA_feature, nodeB_feature, nodeC_feature,
            adj_A_sim, adj_B_sim
        )
        
        # 提取要预测的节点对的特征
        predictAfeature = nodeA_feature[data_set[:, 0],]
        predictBfeature = nodeB_feature[data_set[:, 1],]
        
        # 解码
        if self.use_edge_features:
            # 提取边特征
            edge_features = self.decoder.edge_feature_extractor.extract_edge_features(
                data_set[:, 0], data_set[:, 1],
                adj_AB, adj_AC, adj_BC, adj_A_sim, adj_B_sim,
                diff_A_sim, diff_B_sim
            )
            prediction = self.decoder(predictAfeature, predictBfeature, edge_features).flatten()
        else:
            prediction = self.decoder(predictAfeature, predictBfeature, edgetype).flatten()
        
        return prediction, nodeA_feature, nodeB_feature, nodeC_feature


def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    
    # 加载数据
    adj_B_sim = torch.Tensor(np.float16(sparse.load_npz('./dataset/metabolitesim.npz').todense()))
    adj_BC = np.int16(sparse.load_npz('./dataset/metaboliteGO.npz').todense())
    adj_A_sim = torch.Tensor(np.int16(sparse.load_npz('./dataset/proteinPPI.npz').todense()))
    adj_AC = np.int16(sparse.load_npz('./dataset/proteinGO.npz').todense())
    adj_AB = np.int16(sparse.load_npz('./dataset/proteinMetabolite.npz').todense())
    
    metaboliteList = readListfile('./dataset/metaboliteList.txt')
    proteinList = readListfile('./dataset/proteinList.txt')
    GOList = readListfile('./dataset/GOList.txt')
    
    # 创建结果文件
    result_filename = f'./result/HNCGAT_enhanced_result{str(date.today())}.txt'
    f = open(result_filename, 'a')
    f.write('{}'.format(args))
    f.write('\n')
    
    # 打印创新点使用情况
    innovations = []
    if args.use_weighted_loss:
        innovations.append("Weighted Loss")
    if args.use_diffusion:
        innovations.append("Graph Diffusion")
    if args.use_hard_negative:
        innovations.append("Hard Negative Mining")
    if args.use_edge_features:
        innovations.append("Edge Feature Augmentation")
    
    innovation_str = " + ".join(innovations) if innovations else "Baseline"
    print(f"\n{'='*60}")
    print(f"Running Enhanced HNCGAT with: {innovation_str}")
    print(f"{'='*60}\n")
    f.write(f"Innovations: {innovation_str}\n")
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
    
    pos_data_set[:, 0] = pos_u
    pos_data_set[:, 1] = pos_v
    pos_data_set[:, 2] = 1
    
    neg_data_set[:, 0] = neg_u[negative_sample_index]
    neg_data_set[:, 1] = neg_v[negative_sample_index]
    neg_data_set[:, 2] = 0
    
    # 超参数
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    lr = args.lr
    lamb = args.lamb
    tau = args.tau
    edgetype = args.edgetype
    nodeA_feature_num = hidden_dim
    nodeB_feature_num = hidden_dim
    nodeC_feature_num = hidden_dim
    
    adj_AC = torch.Tensor(adj_AC)
    adj_BC = torch.Tensor(adj_BC)
    adj_AB_ori = torch.Tensor(adj_AB)
    
    trp_varied = [args.train_ratio]
    
    for trp in trp_varied:
        print(f"\n{'='*60}")
        print(f"Training with ratio: {trp}")
        print(f"{'='*60}\n")
        
        f.write(f'\ntrp={trp}\n')
        f.flush()
        
        test_AUC_list = []
        test_AUCpr_list = []
        
        for times in range(args.n_runs):
            print(f"\n--- Run {times + 1}/{args.n_runs} ---")
            
            modelfilename = f'./model/enhanced_HNCGAT_model_trp{trp}_run{times}.pkl'
            
            # 划分训练集和测试集（使用原始HNCGAT的调用方式）
            val_ratio = 0.0
            test_ratio = 1.0 - trp
            n_splits = 5
            split_index = times  # 使用当前的run次数作为split_index
            
            pos_idx_train, pos_idx_val, pos_idx_test, pos_y_train, pos_y_val, pos_y_test, pos_train_mask, pos_val_mask, pos_test_mask = get_train_index(
                pos_u, trp, val_ratio, test_ratio, n_splits, split_index)
            neg_idx_train, neg_idx_val, neg_idx_test, neg_y_train, neg_y_val, neg_y_test, neg_train_mask, neg_val_mask, neg_test_mask = get_train_index(
                negative_sample_index, trp, val_ratio, test_ratio, n_splits, split_index)
            
            # 提取索引
            pos_train_index = pos_idx_train
            pos_test_index = pos_idx_test
            neg_train_index = neg_idx_train
            neg_test_index = neg_idx_test
            
            # 转换为布尔掩码（使用get_train_index返回的mask）
            pos_train_mask = pos_train_mask.astype(bool)
            pos_test_mask = pos_test_mask.astype(bool)
            neg_train_mask = neg_train_mask.astype(bool)
            neg_test_mask = neg_test_mask.astype(bool)
            
            # 构建训练用的邻接矩阵（使用正确的维度）
            train_adj_AB = np.zeros((nodeA_num, nodeB_num), dtype=int)
            for i in pos_train_index:
                idxi = pos_data_set[i, 0]
                idxj = pos_data_set[i, 1]
                train_adj_AB[idxi][idxj] = 1
            
            train_mask = np.array(np.concatenate((pos_train_mask, neg_train_mask), 0), dtype=bool)
            train_mask = torch.tensor(train_mask)
            test_mask = np.array(np.concatenate((pos_test_mask, neg_test_mask), 0), dtype=bool)
            test_mask = torch.tensor(test_mask)
            data_set = np.concatenate((pos_data_set, neg_data_set), 0)
            data_set = torch.tensor(data_set).long()
            
            train_adj_AB = torch.Tensor(train_adj_AB)
            
            # 创建增强版模型
            model = EnhancedMPINet(
                nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, 
                hidden_dim, dropout, edgetype,
                use_weighted_loss=args.use_weighted_loss,
                use_attention=args.weight_attention,
                tau=tau,
                weight_temperature=args.weight_temperature,
                use_diffusion=args.use_diffusion,
                diff_K=args.diffusion_K,
                diff_alpha=args.diffusion_alpha,
                diff_beta=args.diffusion_beta,
                use_hard_negative=args.use_hard_negative,
                hard_negative_ratio=args.hard_negative_ratio,
                hard_negative_weight=args.hard_negative_weight,
                use_edge_features=args.use_edge_features
            )
            
            # 设备
            device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
            cost_sensitive = torch.Tensor([10.]).to(device)
            
            loss_func = torch.nn.BCELoss(reduction='mean', weight=cost_sensitive)
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
            
            # 移动到设备
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
            
            # 如果使用扩散，预先计算扩散后的相似度矩阵（用于边特征提取）
            if args.use_diffusion and args.use_edge_features:
                with torch.no_grad():
                    # 构建meta图
                    A_meta_PM = torch.mm(train_adj_AB, train_adj_AB.t())
                    A_meta_PF = torch.mm(adj_AC, adj_AC.t())
                    A_meta = (A_meta_PM > 0).float() + (A_meta_PF > 0).float()
                    A_meta.fill_diagonal_(0.0)
                    
                    B_meta_MP = torch.mm(train_adj_AB.t(), train_adj_AB)
                    B_meta_MF = torch.mm(adj_BC, adj_BC.t())
                    B_meta = (B_meta_MP > 0).float() + (B_meta_MF > 0).float()
                    B_meta.fill_diagonal_(0.0)
                    
                    # 融合并扩散
                    A_fused = adj_A_sim.float() + args.diffusion_beta * A_meta
                    B_fused = adj_B_sim.float() + args.diffusion_beta * B_meta
                    
                    diffusion_A = GraphDiffusion(K=args.diffusion_K, alpha=args.diffusion_alpha).to(device)
                    diffusion_B = GraphDiffusion(K=args.diffusion_K, alpha=args.diffusion_alpha).to(device)
                    
                    diff_A_sim = diffusion_A(A_fused)
                    diff_B_sim = diffusion_B(B_fused)
            else:
                diff_A_sim = adj_A_sim
                diff_B_sim = adj_B_sim
            
            # 训练循环
            for epoch in range(args.n_epochs):
                model.train()
                opt.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                
                prob, embA, embB, embC = model(
                    data_set, train_adj_AB, adj_AC, adj_BC, 
                    nodeA_feature, nodeB_feature, nodeC_feature, 
                    adj_A_sim, adj_B_sim, edgetype,
                    diff_A_sim, diff_B_sim
                )
                
                label = data_set[:, 2].float()
                train_auc, _ = calculateauc(prob[train_mask], data_set[:, 2][train_mask])
                
                # 计算对比损失
                if model.use_weighted_loss:
                    P_loss, P_weights = model.weighted_P_loss(embA, embB, embC, train_adj_AB, adj_A_sim, adj_AC)
                    M_loss, M_weights = model.weighted_M_loss(embB, embA, embC, train_adj_AB.T, adj_B_sim, adj_BC)
                    F_loss, F_weights = model.weighted_F_loss(embC, embB, embA, adj_BC.T, adj_AC.T)
                    conloss = (P_loss + M_loss + F_loss) / 3
                    
                    if epoch % 100 == 0 and epoch > 0:
                        hn_str = " (Hard Neg)" if args.use_hard_negative else ""
                        print(f"Epoch {epoch}{hn_str} - Avg weights - PM:{P_weights[:, 0].mean():.3f}, "
                              f"PP:{P_weights[:, 1].mean():.3f}, PF:{P_weights[:, 2].mean():.3f}")
                else:
                    conloss = multi_contrastive_loss(embA, embB, embC, adj_A_sim, adj_B_sim, train_adj_AB, adj_AC, adj_BC, tau)
                
                loss = loss_func(prob[train_mask], label[train_mask]) + lamb * conloss
                loss.backward()
                opt.step()
                
                # 评估
                model.eval()
                with torch.no_grad():
                    logits, _, _, _ = model(
                        data_set, train_adj_AB, adj_AC, adj_BC,
                        nodeA_feature, nodeB_feature, nodeC_feature, 
                        adj_A_sim, adj_B_sim, edgetype,
                        diff_A_sim, diff_B_sim
                    )
                    
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
                          " testAUC_ROC {:.4f} | testAUCpr {:.4f}".
                          format(epoch, loss.item(), train_auc, testAUC_ROC, testAUCpr))
            
            print(f"\nBest results at epoch {best_epoch}:")
            print(f"AUC: {test_AUC_best:.4f}, AP: {test_AUCpr_best:.4f}")
            
            test_AUC_list.append(test_AUC_best)
            test_AUCpr_list.append(test_AUCpr_best)
        
        # 统计结果
        test_AUC_mean = np.mean(test_AUC_list)
        test_AUC_std = np.std(test_AUC_list)
        test_AUCpr_mean = np.mean(test_AUCpr_list)
        test_AUCpr_std = np.std(test_AUCpr_list)
        
        result_str = f'trp={trp}, AUC={test_AUC_mean:.4f}±{test_AUC_std:.4f}, AP={test_AUCpr_mean:.4f}±{test_AUCpr_std:.4f}'
        print(f"\n{'='*60}")
        print(f"Final Results: {result_str}")
        print(f"{'='*60}\n")
        
        f.write(result_str + '\n')
        f.flush()
    
    f.close()
    print(f"\nResults saved to: {result_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 原有参数
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--edgetype', type=str, default='concat')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    
    # 加权损失参数
    parser.add_argument('--use-weighted-loss', action='store_true', default=False)
    parser.add_argument('--weight-attention', action='store_true', default=True)
    parser.add_argument('--weight-temperature', type=float, default=0.8)
    
    # 扩散参数
    parser.add_argument('--use-diffusion', action='store_true', default=False)
    parser.add_argument('--diffusion-K', type=int, default=3)
    parser.add_argument('--diffusion-alpha', type=float, default=0.2)
    parser.add_argument('--diffusion-beta', type=float, default=0.5)
    
    # 新增：难负样本挖掘参数
    parser.add_argument('--use-hard-negative', action='store_true', default=False,
                       help='Enable hard negative mining in contrastive loss')
    parser.add_argument('--hard-negative-ratio', type=float, default=0.3,
                       help='Ratio of hard negatives to select (0-1)')
    parser.add_argument('--hard-negative-weight', type=float, default=2.0,
                       help='Weight multiplier for hard negatives (>1)')
    
    # 新增：边特征增强参数
    parser.add_argument('--use-edge-features', action='store_true', default=False,
                       help='Enable edge feature augmentation in decoder')
    
    args = parser.parse_args()
    
    main(args)

