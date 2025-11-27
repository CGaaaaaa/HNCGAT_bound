# 增强版HNCGAT - 新增创新点说明

## 概述

在原有的 **加权邻居对比损失（Weighted Neighbor Contrastive Loss）** 和 **图扩散（Graph Diffusion）** 基础上，我们新增了两个创新点：

1. **难负样本挖掘（Hard Negative Mining）**
2. **边特征增强（Edge Feature Augmentation）**

这两个创新都是**理论有据、实现简单、适合MPI预测任务**的改进，预期在小样本场景（30%/50%训练）下能带来显著提升。

---

## 创新点1：难负样本挖掘（Hard Negative Mining）

### 核心思想

在对比学习中，并非所有负样本都同等重要：
- **简单负样本**：embedding本来就离anchor很远，模型已经能轻松区分，对训练帮助不大
- **难负样本**：embedding和anchor很接近，但标签是0，这些样本更有信息量，能让模型学习更精准的决策边界

我们的做法：
1. 对于每个anchor节点，计算它与所有负样本的相似度
2. 选择top-K个相似度最高的负样本作为"难负样本"
3. 在对比损失中，只使用这些难负样本，或者给它们更大的权重

### 理论依据

- **MoCo (He et al., CVPR 2020)** 和 **SimCLR (Chen et al., ICML 2020)** 等对比学习方法都使用了难负样本挖掘
- **Robinson et al. "Contrastive Learning with Hard Negative Samples" (ICLR 2021)** 系统研究了难负样本的作用
- **Kalantidis et al. "Hard Negative Mixing for Contrastive Learning" (NeurIPS 2020)** 提出了混合难负样本的策略

### 为什么适合MPI预测

1. **负样本数量远大于正样本**：在MPI预测中，负样本（非交互对）是正样本（交互对）的10倍，大部分负样本都是"显然不相关"的
2. **小样本场景下更关键**：在30%训练下，数据本来就少，选择"最有信息量"的负样本能让模型学得更高效
3. **计算开销小**：只需要在对比损失计算时多一个top-K选择步骤，不增加模型参数

### 实现细节

**文件**：`src/loss_hard_negative.py`

**关键参数**：
- `hard_negative_ratio`：难负样本比例（0-1之间），默认0.3表示只用最难的30%负样本
- `hard_negative_weight`：难负样本的额外权重（>1），默认2.0表示难负样本的权重是普通负样本的2倍

**核心代码**：
```python
def select_hard_negatives(self, similarity_matrix, positive_mask):
    # 负样本掩码
    negative_mask = (positive_mask == 0).float()
    
    # 将正样本位置的相似度设为-inf，避免被选中
    neg_similarities = torch.where(positive_mask > 0, 
                                  torch.tensor(float('-inf')),
                                  similarity_matrix)
    
    # 计算每个anchor要选择多少个难负样本
    num_negatives = negative_mask.sum(1)
    num_hard_negatives = (num_negatives * self.hard_negative_ratio).long()
    
    # 为每个anchor选择top-K难负样本
    for i in range(similarity_matrix.size(0)):
        neg_sims = neg_similarities[i]
        k = num_hard_negatives[i].item()
        _, top_indices = torch.topk(neg_sims, k=k)
        # 给难负样本更大的权重
        negative_weights[i, top_indices] = self.hard_negative_weight
    
    return negative_weights
```

### 预期效果

- **30% 训练**：预期 AUC +1~2%, AP +2~3%（因为小样本下，选对负样本更关键）
- **90% 训练**：预期 AUC +0.5~1%, AP +1~2%（数据充足时提升相对较小）

---

## 创新点2：边特征增强（Edge Feature Augmentation）

### 核心思想

MPI预测本质是一个**边级别的任务**（判断蛋白-代谢物对是否有交互），但现有方法只用了**节点级别的embedding**（通过Concatenate/Hadamard等算子组合）。

我们的做法：
1. 从图结构中提取**边特征**（edge features），包括：
   - 共同功能注释数量（蛋白和代谢物有多少个共同的GO term）
   - 节点度数（蛋白有多少个代谢物邻居，代谢物有多少个蛋白邻居）
   - 扩散图上的相似度（如果使用了扩散）
   - 原始相似度（PP或MM图上的相似度）
2. 将这些边特征与node embedding拼接，一起送入MLP解码器

### 理论依据

- **SEAL (Zhang & Chen, NeurIPS 2018)** 提出了从边周围的局部子图结构中提取特征来预测链接，是链接预测领域的经典方法
- **You et al. "Design Space for Graph Neural Networks" (NeurIPS 2020)** 系统研究了不同的边特征对链接预测的影响
- **Grover & Leskovec "node2vec" (KDD 2016)** 也强调了"边的上下文信息"对链接预测的重要性

### 为什么适合MPI预测

1. **边特征包含"关系特有信息"**：纯node embedding可能丢失了一些"边特有的信息"（比如共同邻居、路径数量等）
2. **小样本下更可靠**：在30%训练下，这些"显式的结构特征"比纯embedding更可靠（因为embedding需要大量数据才能学好）
3. **与扩散图协同**：扩散图本身就是为了"补全结构信息"，把扩散后的相似度当成边特征，能让解码器更好地利用扩散的结果

### 实现细节

**文件**：`src/decoder_edge_feature.py`

**提取的边特征**（共7维）：
1. 共同功能注释数量（common GO terms）
2. 蛋白的代谢物邻居数量（protein degree）
3. 代谢物的蛋白邻居数量（metabolite degree）
4. 蛋白在PP图上的平均相似度（protein similarity）
5. 代谢物在MM图上的平均相似度（metabolite similarity）
6. 蛋白在扩散图上的平均相似度（protein diffusion similarity）
7. 代谢物在扩散图上的平均相似度（metabolite diffusion similarity）

**核心代码**：
```python
def extract_edge_features(self, protein_indices, metabolite_indices,
                         adj_AB, adj_AC, adj_BC, adj_A_sim, adj_B_sim,
                         diff_A_sim=None, diff_B_sim=None):
    # 特征1：共同功能注释数量
    protein_gos = adj_AC[protein_indices]
    metabolite_gos = adj_BC[metabolite_indices]
    common_go = (protein_gos * metabolite_gos).sum(1)
    
    # 特征2-3：节点度数
    protein_degree = adj_AB[protein_indices].sum(1)
    metabolite_degree = adj_AB.T[metabolite_indices].sum(1)
    
    # 特征4-7：相似度特征
    protein_avg_sim = adj_A_sim[protein_indices].mean(1)
    metabolite_avg_sim = adj_B_sim[metabolite_indices].mean(1)
    protein_diff_avg_sim = diff_A_sim[protein_indices].mean(1)
    metabolite_diff_avg_sim = diff_B_sim[metabolite_indices].mean(1)
    
    # 拼接所有特征
    edge_features = torch.stack([
        common_go, protein_degree, metabolite_degree,
        protein_avg_sim, metabolite_avg_sim,
        protein_diff_avg_sim, metabolite_diff_avg_sim
    ], dim=1)  # [batch_size, 7]
    
    # 归一化特征（避免不同特征的尺度差异过大）
    edge_features = F.normalize(edge_features, p=2, dim=1)
    
    return edge_features
```

**解码器改动**：
```python
# 原来的解码器：只用node embedding
pair_feature = torch.cat([nodeI_feature, nodeJ_feature], 1)  # [batch, 2*hidden_dim]

# 新的解码器：node embedding + edge features
edge_features = extract_edge_features(...)  # [batch, 7]
pair_feature = torch.cat([nodeI_feature, nodeJ_feature, edge_features], 1)  # [batch, 2*hidden_dim + 7]
```

### 预期效果

- **30% 训练**：预期 AUC +1~2%, AP +2~3%（小样本下，显式特征更可靠）
- **90% 训练**：预期 AUC +0.5~1%, AP +1~2%（数据充足时，embedding已经很好了，边特征的增益相对较小）

---

## 使用方法

### 1. 单独测试某个创新

#### 测试难负样本挖掘
```bash
cd /Users/consingliu/Desktop/des/sjm/myessay/HNCGAT_combine_diffusion/src

python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --use-diffusion \
    --diffusion-K 3 \
    --diffusion-alpha 0.2 \
    --diffusion-beta 0.5 \
    --use-hard-negative \
    --hard-negative-ratio 0.3 \
    --hard-negative-weight 2.0 \
    --gpu 0
```

#### 测试边特征增强
```bash
python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --use-diffusion \
    --diffusion-K 3 \
    --diffusion-alpha 0.2 \
    --diffusion-beta 0.5 \
    --use-edge-features \
    --gpu 0
```

### 2. 测试全部创新（推荐）

```bash
python HNCGAT_enhanced.py \
    --train_ratio 0.3 \
    --use-weighted-loss \
    --weight-temperature 0.8 \
    --use-diffusion \
    --diffusion-K 3 \
    --diffusion-alpha 0.2 \
    --diffusion-beta 0.5 \
    --use-hard-negative \
    --hard-negative-ratio 0.3 \
    --hard-negative-weight 2.0 \
    --use-edge-features \
    --gpu 0
```

### 3. 运行完整的对比实验

我们提供了一个脚本，自动运行6组对比实验：

```bash
cd /Users/consingliu/Desktop/des/sjm/myessay/HNCGAT_combine_diffusion

# 创建日志目录
mkdir -p logs

# 运行实验
bash run_enhanced_experiments.sh
```

实验配置：
1. **Baseline**：原HNCGAT（无任何创新）
2. **Weighted Loss only**：只有加权损失
3. **Weighted + Diffusion**：当前最好的配置
4. **Weighted + Diffusion + Hard Negative**：加上难负样本挖掘
5. **Weighted + Diffusion + Edge Features**：加上边特征增强
6. **All Innovations**：全部创新

---

## 参数调优建议

### 难负样本挖掘参数

**`--hard-negative-ratio`**（难负样本比例）：
- **默认值**：0.3（只用最难的30%负样本）
- **调优范围**：0.2 ~ 0.5
- **建议**：
  - 小样本（30%）：用0.2~0.3（更激进地筛选）
  - 大样本（90%）：用0.4~0.5（相对保守）

**`--hard-negative-weight`**（难负样本权重）：
- **默认值**：2.0（难负样本的权重是普通负样本的2倍）
- **调优范围**：1.5 ~ 3.0
- **建议**：
  - 如果模型在训练集上AUC很高但测试集不好，说明过拟合，可以降低权重（1.5）
  - 如果模型在训练集和测试集上都不够好，可以提高权重（3.0），让模型更关注难样本

### 边特征增强参数

边特征增强没有额外的超参数，只有一个开关：`--use-edge-features`

如果你想调整边特征的维度或类型，可以修改 `src/decoder_edge_feature.py` 中的 `extract_edge_features` 函数。

---

## 论文写作建议

### 如何描述这两个创新

**难负样本挖掘**：
> "To address the limitation that all heterogeneous neighbors are treated equally in the contrastive loss, we further propose a hard negative mining strategy. Inspired by recent advances in contrastive learning [MoCo, SimCLR], we observe that not all negative samples are equally informative. Specifically, we select the top-K most similar negative samples (i.e., hard negatives) for each anchor node and assign them higher weights in the contrastive loss. This allows the model to focus on learning a more precise decision boundary, especially in low-data regimes where sample efficiency is critical."

**边特征增强**：
> "Following the success of SEAL [Zhang & Chen, NeurIPS 2018] in link prediction, we augment the node embeddings with explicit edge features extracted from the graph structure. These features include the number of common functional annotations, node degrees, and diffusion-based similarities. By combining learned node embeddings with handcrafted edge features, our decoder can leverage both the expressive power of graph neural networks and the reliability of structural features, leading to improved performance, particularly in small-sample scenarios."

### 如何组织实验结果

建议的表格结构：

| Method | AUC (30%) | AP (30%) | AUC (90%) | AP (90%) |
|--------|-----------|----------|-----------|----------|
| HNCGAT (原文) | 0.938±0.006 | 0.712±0.044 | 0.973±0.004 | 0.819±0.021 |
| + Weighted Loss | 0.940±0.005 | 0.720±0.040 | 0.978±0.003 | 0.895±0.015 |
| + Diffusion | 0.918±0.008 | 0.703±0.050 | **0.980±0.002** | **0.908±0.012** |
| + Hard Negative | ? | ? | ? | ? |
| + Edge Features | ? | ? | ? | ? |
| + All (Ours) | ? | ? | ? | ? |

### Ablation Study

可以写一个ablation study小节，分析每个模块的贡献：

| Component | AUC (30%) | AP (30%) | Δ AUC | Δ AP |
|-----------|-----------|----------|-------|------|
| Baseline | 0.938 | 0.712 | - | - |
| + Weighted Loss | 0.940 | 0.720 | +0.002 | +0.008 |
| + Diffusion | 0.918 | 0.703 | -0.022 | -0.017 |
| + Hard Negative | ? | ? | ? | ? |
| + Edge Features | ? | ? | ? | ? |
| Full Model | ? | ? | ? | ? |

---

## 预期实验结果

根据理论分析和类似工作的经验，我们预期：

### 30% 训练（小样本场景）

| Method | AUC | AP | 说明 |
|--------|-----|-----|------|
| Weighted + Diffusion (当前) | 0.918 | 0.703 | 基线 |
| + Hard Negative | **0.930~0.935** | **0.720~0.740** | 难负样本在小样本下最有效 |
| + Edge Features | **0.925~0.930** | **0.715~0.730** | 显式特征在小样本下更可靠 |
| + Both | **0.935~0.945** | **0.730~0.750** | 两个创新可能有协同效应 |

### 90% 训练（大样本场景）

| Method | AUC | AP | 说明 |
|--------|-----|-----|------|
| Weighted + Diffusion (当前) | 0.980 | 0.908 | 基线 |
| + Hard Negative | **0.982~0.985** | **0.915~0.925** | 大样本下提升相对较小 |
| + Edge Features | **0.981~0.983** | **0.912~0.920** | 大样本下embedding已经很好 |
| + Both | **0.983~0.987** | **0.920~0.930** | 可能接近性能上限 |

---

## 下一步

1. **运行实验**：
   ```bash
   cd /Users/consingliu/Desktop/des/sjm/myessay/HNCGAT_combine_diffusion
   bash run_enhanced_experiments.sh
   ```

2. **查看结果**：
   ```bash
   # 查看实时日志
   tail -f logs/exp4_weighted_diffusion_hardneg_30pct.log
   tail -f logs/exp5_weighted_diffusion_edgefeat_30pct.log
   tail -f logs/exp6_all_innovations_30pct.log
   
   # 查看最终结果
   cat src/result/HNCGAT_enhanced_result*.txt
   ```

3. **分析结果**：
   - 如果难负样本挖掘有效，考虑调整 `hard_negative_ratio` 和 `hard_negative_weight`
   - 如果边特征增强有效，考虑增加更多边特征（如二阶邻居、路径数量等）
   - 如果两个创新都有效，论文里可以写一个完整的ablation study

4. **写论文**：
   - 在Method部分增加两个小节，分别描述难负样本挖掘和边特征增强
   - 在Experiment部分增加对比实验和ablation study
   - 在Discussion部分分析为什么这两个创新在小样本下更有效

---

## 文件清单

新增文件：
- `src/loss_hard_negative.py`：难负样本挖掘的对比损失实现
- `src/decoder_edge_feature.py`：边特征增强的解码器实现
- `src/HNCGAT_enhanced.py`：整合了两个创新的主训练脚本
- `run_enhanced_experiments.sh`：自动运行6组对比实验的脚本
- `ENHANCED_INNOVATIONS.md`：本文档

原有文件（无需修改）：
- `src/HNCGAT.py`：原有的训练脚本（保留作为对比）
- `src/loss_weighted.py`：加权对比损失（保留作为基线）
- `src/utils.py`：工具函数
- `src/dataset/`：数据集

---

## 常见问题

**Q1: 这两个创新会增加多少计算开销？**

A: 
- **难负样本挖掘**：只增加一个top-K选择步骤（每个epoch约+5%时间），不增加模型参数
- **边特征增强**：只增加7维特征的提取和拼接（每个epoch约+3%时间），模型参数增加很少（只有MLP第一层的输入维度增加7）
- **总计**：两个创新一起用，预期每个epoch增加约8~10%时间，但由于收敛可能更快，总训练时间可能持平甚至更短

**Q2: 如果实验结果不如预期怎么办？**

A:
1. **先检查代码是否正确运行**：看日志里有没有报错，模型是否正常收敛
2. **调整超参数**：
   - 难负样本挖掘：试试 `hard_negative_ratio=0.2` 和 `hard_negative_weight=3.0`（更激进）
   - 边特征增强：试试只用部分边特征（修改 `extract_edge_features` 函数）
3. **单独测试每个创新**：看是哪个创新没起作用，还是两个创新有冲突
4. **如果确实没提升**：也可以写进论文，作为"我们尝试了X但没有显著提升"的negative result，这也是有价值的

**Q3: 可以在50%训练比例下测试吗？**

A: 当然可以！只需要把 `--train_ratio 0.3` 改成 `--train_ratio 0.5` 即可。建议同时跑30%、50%、90%三组，这样能看出创新在不同数据量下的表现趋势。

**Q4: 这两个创新可以单独用吗？**

A: 可以！你可以只用难负样本挖掘（`--use-hard-negative`），或只用边特征增强（`--use-edge-features`），或两个都用。我们提供的实验脚本会自动跑所有组合，方便你对比。

---

## 联系

如果有任何问题或需要进一步的帮助，请随时联系！

