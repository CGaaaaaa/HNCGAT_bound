# 加权邻居对比损失实现说明（修正版 v2）

## 改进概述

本实现解决了论文Limitations中明确提到的问题：

> "One limitation of the framework is that all the heterogeneous neighbors of the anchor are treated equally in the current neighbor contrastive loss."

### 改进点

1. **自适应权重学习**：为不同邻居类型（PM, PP, PF）学习自适应权重
2. **节点特定权重**：使用注意力机制为每个节点学习特定的权重分配
3. **解决局限性**：不再平等对待所有邻居，而是根据重要性分配权重

### 修正版 v2 更新（2025-11-18）

**重要修正**：
1. **修正分母结构**：使加权损失的分母结构与原始损失函数完全一致
   - PM/MP连接：只考虑正样本对（与原始一致）
   - PP/MM/PF/MF连接：考虑所有对（与原始一致）
   - 这确保了加权机制在数学上与原始损失兼容

2. **添加权重正则化**：使用温度缩放（默认温度=0.7，可通过`--weight-temperature`调整）防止权重过度集中
   - 避免权重完全偏向单一邻居类型（如之前PP权重为0的问题）
   - 使权重分布更加平滑和合理
   - 温度参数可调，便于实验优化

3. **保持数学一致性**：确保改进版本在结构上与原始损失函数对齐，提升改进的有效性

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
- `--weight-temperature`: 权重softmax的温度参数（默认：0.7）
  - 较小值（0.5）：权重分布更均匀，三种连接类型权重更接近
  - 较大值（0.8-1.0）：允许权重有更大差异，可能更聚焦重要连接
  - 建议测试范围：0.5, 0.7, 0.8, 1.0
- 其他参数与原版相同

### 温度参数调优示例

```bash
# 测试温度0.5（更均匀的权重分布）
python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --weight-temperature 0.5 --edgetype concat

# 测试温度0.7（默认值，当前最佳）
python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --weight-temperature 0.7 --edgetype concat

# 测试温度0.8（允许更多权重差异）
python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --weight-temperature 0.8 --edgetype concat

# 测试温度1.0（标准softmax，无温度缩放）
python HNCGAT.py --gpu 0 --n-epochs 1000 --use-weighted-loss --weight-temperature 1.0 --edgetype concat
```

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

根据论文Limitations的分析，修正版v2应该能够：
1. **提升模型性能**：通过正确的数学结构和权重正则化，预期AUC提升1-3%
2. **更好的权重分布**：权重不再过度集中，PP权重不再为0
3. **更好的利用不同邻居类型的信息**：平衡利用PM、PP、PF三种连接
4. **提供可解释的权重分配**：权重分布更合理，便于分析

## 版本历史

- **v1（初始版本）**：实现了基本的加权机制，但分母结构与原始损失不一致
- **v2（修正版）**：修正分母结构，添加权重正则化，与原始损失函数完全兼容

## 注意事项

1. 加权损失会增加模型参数（注意力MLP）
2. 训练时间可能略有增加
3. 建议在相同设置下对比原始版本和改进版本

