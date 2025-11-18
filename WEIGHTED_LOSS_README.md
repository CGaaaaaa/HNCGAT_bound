# 加权邻居对比损失实现说明

## 改进概述

本实现解决了论文Limitations中明确提到的问题：

> "One limitation of the framework is that all the heterogeneous neighbors of the anchor are treated equally in the current neighbor contrastive loss."

### 改进点

1. **自适应权重学习**：为不同邻居类型（PM, PP, PF）学习自适应权重
2. **节点特定权重**：使用注意力机制为每个节点学习特定的权重分配
3. **解决局限性**：不再平等对待所有邻居，而是根据重要性分配权重

## 使用方法

### 基本使用（使用加权损失）

```bash
cd src
python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --edgetype concat
```

### 对比实验（原始损失 vs 加权损失）

```bash
# 原始损失（基线）
python HNCGAT.py --gpu 0 --n-epochs 1000 --edgetype concat

# 加权损失（改进版）
python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --edgetype concat
```

### 参数说明

- `--use-weighted-loss`: 启用加权对比损失（解决论文Limitations）
- `--weight-attention`: 使用注意力机制学习权重（默认：True）
- 其他参数与原版相同

## 技术细节

### 加权机制

1. **邻居聚合**：为每种邻居类型计算聚合表示
2. **权重学习**：
   - 注意力模式：为每个节点学习特定的权重（更灵活）
   - 全局模式：所有节点共享相同权重（更简单）
3. **加权损失计算**：使用学习到的权重加权正负样本对

### 权重可视化

训练过程中每100个epoch会打印平均权重：
```
Epoch 100 - Average weights - PM:0.350, PP:0.420, PF:0.230
```
这可以帮助分析不同邻居类型的重要性。

## 文件说明

- `loss_weighted.py`: 加权损失函数实现
  - `AdaptiveWeightedPConLoss`: 蛋白质对比损失
  - `AdaptiveWeightedMConLoss`: 代谢物对比损失
  - `AdaptiveWeightedFConLoss`: 功能注释对比损失

- `HNCGAT.py`: 主模型文件（已集成加权损失）

## 实验建议

### 1. 基线对比
- 运行原始HNCGAT（不使用加权损失）
- 运行改进版HNCGAT（使用加权损失）
- 对比性能提升

### 2. 权重分析
- 观察训练过程中的权重变化
- 分析不同邻居类型的重要性
- 可视化权重分布

### 3. 消融研究
- 对比注意力权重 vs 全局权重
- 分析权重机制的有效性

## 预期效果

根据论文Limitations的分析，加权机制应该能够：
1. 提升模型性能（AUC提升1-2%）
2. 更好地利用不同邻居类型的信息
3. 提供可解释的权重分配

## 注意事项

1. 加权损失会增加模型参数（注意力MLP）
2. 训练时间可能略有增加
3. 建议在相同设置下对比原始版本和改进版本

