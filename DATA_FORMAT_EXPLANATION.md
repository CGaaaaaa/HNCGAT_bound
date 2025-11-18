# HNCGAT 数据格式和处理方式说明

## 📊 数据格式

### ❌ 不是 xlsx 格式

HNCGAT **不使用 xlsx 文件**，而是使用：

### ✅ 实际使用的数据格式

1. **`.npz` 文件**（NumPy压缩格式）- 主要数据
   - `metabolitesim.npz` - 代谢物相似度矩阵
   - `metaboliteGO.npz` - 代谢物-GO功能注释关联
   - `proteinPPI.npz` - 蛋白质-蛋白质相互作用
   - `proteinGO.npz` - 蛋白质-GO功能注释关联
   - `proteinMetabolite.npz` - **蛋白质-代谢物相互作用**（预测目标）

2. **`.txt` 文件** - 列表文件
   - `metaboliteList.txt` - 代谢物ID列表
   - `proteinList.txt` - 蛋白质ID列表
   - `GOList.txt` - GO功能注释列表

## 🔄 数据处理流程

### 数据加载（代码第259-267行）

```python
# 加载邻接矩阵（.npz格式）
adj_B_sim = torch.Tensor(np.float16(sparse.load_npz('./dataset/metabolitesim.npz').todense()))
adj_BC = np.int16(sparse.load_npz('./dataset/metaboliteGO.npz').todense())
adj_A_sim = torch.Tensor(np.int16(sparse.load_npz('./dataset/proteinPPI.npz').todense()))
adj_AC = np.int16(sparse.load_npz('./dataset/proteinGO.npz').todense())
adj_AB = np.int16(sparse.load_npz('./dataset/proteinMetabolite.npz').todense())

# 加载列表文件（.txt格式）
metaboliteList = readListfile('./dataset/metaboliteList.txt')
proteinList = readListfile('./dataset/proteinList.txt')
GOList = readListfile('./dataset/GOList.txt')
```

### 数据划分（90%训练比例）

**90%训练比例 = 数据划分方式，不是数据格式**

```python
# 第278-282行：从邻接矩阵中提取正负样本
pos_u, pos_v = np.where(adj_AB != 0)  # 正样本（已知的相互作用）
neg_u, neg_v = np.where(adj_AB == 0)  # 负样本（未知的相互作用）

# 第328-331行：根据训练比例划分数据
pos_idx_train, pos_idx_val, pos_idx_test, ... = get_train_index(
    pos_u, train_ratio, val_ratio, test_ratio, numRandom, random_state)
```

## 🎯 关键理解

### 90%训练比例的含义

**不是数据格式，而是数据划分比例**：

```
总数据集（所有蛋白质-代谢物对）
├── 训练集（90%）← 用来训练模型
└── 测试集（10%）← 用来评估模型
```

### 数据是统一的

- **所有实验（30%, 50%, 90%）使用相同的数据集**
- **唯一不同**：训练集和测试集的划分比例
- **数据格式**：都是 `.npz` 和 `.txt` 文件

## 📝 处理流程总结

### 1. 数据加载（一次性）
- 加载所有 `.npz` 文件（邻接矩阵）
- 加载所有 `.txt` 文件（ID列表）
- **这是统一的，所有实验都一样**

### 2. 数据划分（根据训练比例）
- 90%训练比例：90%数据训练，10%数据测试
- 30%训练比例：30%数据训练，70%数据测试
- **这是唯一的区别**

### 3. 模型训练（统一流程）
- 使用划分后的训练集训练模型
- 使用测试集评估性能
- **训练流程完全相同**

## 💡 回答你的问题

### Q: 90%的实验是使用了数据xlsx的进行处理的吗？

**A: 不是**

- ❌ **不使用 xlsx 文件**
- ✅ **使用 .npz 和 .txt 文件**
- ✅ **所有实验（30%, 50%, 90%）使用相同的数据文件**

### Q: 就是进行了统一的运行跑？

**A: 是的，但需要理解"统一"的含义**

- ✅ **数据加载是统一的**：所有实验加载相同的数据文件
- ✅ **训练流程是统一的**：所有实验使用相同的训练代码
- ⚠️ **数据划分是不同的**：90% vs 30% vs 50% 只是划分比例不同

## 🔍 代码流程

```
1. 加载数据（统一）
   ├── 加载 .npz 文件（邻接矩阵）
   └── 加载 .txt 文件（ID列表）

2. 提取样本（统一）
   ├── 正样本：已知的蛋白质-代谢物相互作用
   └── 负样本：未知的相互作用（随机采样）

3. 数据划分（根据训练比例）
   ├── 90%训练：90%训练集，10%测试集
   ├── 50%训练：50%训练集，50%测试集
   └── 30%训练：30%训练集，70%测试集

4. 模型训练（统一流程）
   ├── 使用训练集训练
   └── 使用测试集评估
```

## 📌 总结

1. **数据格式**：`.npz` 和 `.txt`，不是 xlsx
2. **数据来源**：统一的数据集（所有实验相同）
3. **90%训练比例**：只是数据划分方式，不是数据格式
4. **处理流程**：统一的代码流程，只是划分比例不同

**简单说**：
- 数据是统一的（.npz格式）
- 90%只是划分比例（90%训练，10%测试）
- 所有实验使用相同的代码和数据，只是划分比例不同


