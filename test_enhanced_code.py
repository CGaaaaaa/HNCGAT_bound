#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试增强版代码是否有语法错误
"""
import sys
import torch

print("测试导入模块...")

try:
    sys.path.insert(0, './src')
    
    # 测试导入难负样本挖掘模块
    print("1. 测试导入 loss_hard_negative...")
    from loss_hard_negative import HardNegativeWeightedPConLoss, HardNegativeWeightedMConLoss, HardNegativeWeightedFConLoss
    print("   ✓ loss_hard_negative 导入成功")
    
    # 测试导入边特征增强模块
    print("2. 测试导入 decoder_edge_feature...")
    from decoder_edge_feature import EdgeFeatureExtractor, EdgeFeatureEnhancedDecoder
    print("   ✓ decoder_edge_feature 导入成功")
    
    # 测试创建难负样本挖掘损失
    print("3. 测试创建 HardNegativeWeightedPConLoss...")
    loss_module = HardNegativeWeightedPConLoss(
        hidden_dim=64, 
        tau=0.1, 
        use_attention=True,
        weight_temperature=0.8,
        hard_negative_ratio=0.3,
        hard_negative_weight=2.0
    )
    print(f"   ✓ HardNegativeWeightedPConLoss 创建成功")
    print(f"   - Parameters: {sum(p.numel() for p in loss_module.parameters())} 个参数")
    
    # 测试创建边特征提取器
    print("4. 测试创建 EdgeFeatureExtractor...")
    edge_extractor = EdgeFeatureExtractor(use_diffusion=True)
    print("   ✓ EdgeFeatureExtractor 创建成功")
    
    # 测试创建边特征增强解码器
    print("5. 测试创建 EdgeFeatureEnhancedDecoder...")
    decoder = EdgeFeatureEnhancedDecoder(
        input_dim=64,
        edgetype='concat',
        use_edge_features=True,
        edge_feature_dim=7
    )
    print(f"   ✓ EdgeFeatureEnhancedDecoder 创建成功")
    print(f"   - Parameters: {sum(p.numel() for p in decoder.parameters())} 个参数")
    
    # 测试前向传播（用假数据）
    print("6. 测试前向传播...")
    batch_size = 10
    hidden_dim = 64
    num_proteins = 100
    num_metabolites = 50
    num_go = 200
    
    # 创建假数据
    embP = torch.randn(num_proteins, hidden_dim)
    embM = torch.randn(num_metabolites, hidden_dim)
    embF = torch.randn(num_go, hidden_dim)
    
    PM_adj = torch.randint(0, 2, (num_proteins, num_metabolites)).float()
    PP_adj = torch.randint(0, 2, (num_proteins, num_proteins)).float()
    PF_adj = torch.randint(0, 2, (num_proteins, num_go)).float()
    
    # 测试难负样本挖掘损失
    print("   6.1 测试 HardNegativeWeightedPConLoss forward...")
    loss, weights = loss_module(embP, embM, embF, PM_adj, PP_adj, PF_adj)
    print(f"       ✓ Loss: {loss.item():.4f}, Weights shape: {weights.shape}")
    
    # 测试边特征提取
    print("   6.2 测试 EdgeFeatureExtractor...")
    protein_indices = torch.randint(0, num_proteins, (batch_size,))
    metabolite_indices = torch.randint(0, num_metabolites, (batch_size,))
    
    adj_AB = PM_adj
    adj_AC = PF_adj
    adj_BC = torch.randint(0, 2, (num_metabolites, num_go)).float()
    adj_A_sim = torch.rand(num_proteins, num_proteins)
    adj_B_sim = torch.rand(num_metabolites, num_metabolites)
    diff_A_sim = torch.rand(num_proteins, num_proteins)
    diff_B_sim = torch.rand(num_metabolites, num_metabolites)
    
    edge_features = edge_extractor.extract_edge_features(
        protein_indices, metabolite_indices,
        adj_AB, adj_AC, adj_BC, adj_A_sim, adj_B_sim,
        diff_A_sim, diff_B_sim
    )
    print(f"       ✓ Edge features shape: {edge_features.shape}")
    
    # 测试解码器
    print("   6.3 测试 EdgeFeatureEnhancedDecoder forward...")
    nodeI_feature = embP[protein_indices]
    nodeJ_feature = embM[metabolite_indices]
    outputs = decoder(nodeI_feature, nodeJ_feature, edge_features)
    print(f"       ✓ Decoder outputs shape: {outputs.shape}")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！代码没有语法错误。")
    print("="*60)
    print("\n可以开始运行实验了：")
    print("  bash run_enhanced_experiments.sh")
    print("\n或者单独测试某个创新：")
    print("  cd src")
    print("  python HNCGAT_enhanced.py --train_ratio 0.3 --use-weighted-loss --use-diffusion --use-hard-negative --gpu 0")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

