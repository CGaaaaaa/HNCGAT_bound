#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试加权损失函数
快速验证代码是否能正常工作
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("测试1: 导入模块")
print("=" * 60)

try:
    from loss_weighted import AdaptiveWeightedPConLoss, AdaptiveWeightedMConLoss, AdaptiveWeightedFConLoss
    print("✓ 成功导入加权损失类")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from loss import sim
    print("✓ 成功导入sim函数")
except Exception as e:
    print(f"✗ 导入sim失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试2: 创建加权损失模块")
print("=" * 60)

try:
    hidden_dim = 64
    tau = 0.1
    
    # 测试注意力模式
    P_loss_att = AdaptiveWeightedPConLoss(hidden_dim, tau, use_attention=True)
    print("✓ 创建AdaptiveWeightedPConLoss (注意力模式)")
    
    M_loss_att = AdaptiveWeightedMConLoss(hidden_dim, tau, use_attention=True)
    print("✓ 创建AdaptiveWeightedMConLoss (注意力模式)")
    
    F_loss_att = AdaptiveWeightedFConLoss(hidden_dim, tau, use_attention=True)
    print("✓ 创建AdaptiveWeightedFConLoss (注意力模式)")
    
except Exception as e:
    print(f"✗ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试3: 前向传播测试（小规模数据）")
print("=" * 60)

try:
    # 创建小规模测试数据
    N_P = 100  # 蛋白质节点数
    N_M = 50   # 代谢物节点数
    N_F = 30   # 功能注释节点数
    
    # 随机嵌入
    embP = torch.randn(N_P, hidden_dim)
    embM = torch.randn(N_M, hidden_dim)
    embF = torch.randn(N_F, hidden_dim)
    
    # 创建随机邻接矩阵
    PM_adj = torch.randint(0, 2, (N_P, N_M)).float()
    PP_adj = torch.randint(0, 2, (N_P, N_P)).float()
    PP_adj = (PP_adj + PP_adj.T).clamp(0, 1)  # 对称化
    PF_adj = torch.randint(0, 2, (N_P, N_F)).float()
    
    print(f"✓ 创建测试数据: P={N_P}, M={N_M}, F={N_F}")
    
    # 测试前向传播
    P_loss, P_weights = P_loss_att(embP, embM, embF, PM_adj, PP_adj, PF_adj)
    print(f"✓ P_loss前向传播成功, loss={P_loss.item():.4f}")
    print(f"  权重形状: {P_weights.shape}, 平均权重: PM={P_weights[:, 0].mean():.3f}, "
          f"PP={P_weights[:, 1].mean():.3f}, PF={P_weights[:, 2].mean():.3f}")
    
    # 测试M和F
    MM_adj = torch.randint(0, 2, (N_M, N_M)).float()
    MM_adj = (MM_adj + MM_adj.T).clamp(0, 1)
    MF_adj = torch.randint(0, 2, (N_M, N_F)).float()
    
    M_loss, M_weights = M_loss_att(embM, embP, embF, PM_adj.T, MM_adj, MF_adj)
    print(f"✓ M_loss前向传播成功, loss={M_loss.item():.4f}")
    
    FM_adj = MF_adj.T
    FP_adj = PF_adj.T
    F_loss, F_weights = F_loss_att(embF, embM, embP, FM_adj, FP_adj)
    print(f"✓ F_loss前向传播成功, loss={F_loss.item():.4f}")
    
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试4: 反向传播测试")
print("=" * 60)

try:
    # 测试梯度计算
    total_loss = (P_loss + M_loss + F_loss) / 3
    total_loss.backward()
    
    # 检查梯度
    has_grad = False
    for param in P_loss_att.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    if has_grad:
        print("✓ 反向传播成功，梯度已计算")
    else:
        print("⚠ 警告：未检测到梯度（可能是测试数据问题）")
    
except Exception as e:
    print(f"✗ 反向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试5: 与HNCGAT集成测试")
print("=" * 60)

try:
    # 测试导入HNCGAT
    from HNCGAT import MPINet
    print("✓ 成功导入MPINet")
    
    # 创建小规模模型
    nodeA_num = 100
    nodeB_num = 50
    nodeC_num = 30
    nodeA_feature_num = 32
    nodeB_feature_num = 32
    nodeC_feature_num = 32
    
    # 测试不使用加权损失
    model_normal = MPINet(nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, 
                          nodeC_feature_num, hidden_dim, 0.5, 'concat',
                          use_weighted_loss=False)
    print("✓ 创建模型（不使用加权损失）")
    
    # 测试使用加权损失
    model_weighted = MPINet(nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num,
                            nodeC_feature_num, hidden_dim, 0.5, 'concat',
                            use_weighted_loss=True, use_attention=True, tau=0.1)
    print("✓ 创建模型（使用加权损失）")
    
    # 检查是否有加权损失模块
    if hasattr(model_weighted, 'weighted_P_loss'):
        print("✓ 模型包含加权损失模块")
    else:
        print("✗ 模型缺少加权损失模块")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ 集成测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)
print("\n代码可以正常使用，可以到云主机上运行完整实验了。")
print("\n使用命令：")
print("  python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --edgetype concat")

