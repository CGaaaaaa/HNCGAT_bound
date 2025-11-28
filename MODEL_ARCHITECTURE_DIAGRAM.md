# 增强版HNCGAT模型结构图（Mermaid格式）

## 完整模型流程图

```mermaid
graph TB
    A[输入: 异质图G] --> B[图扩散模块]
    B --> C[扩散后的相似图]
    C --> D[异质GAT编码器]
    D --> E[节点嵌入<br/>embP, embM, embF]
    
    E --> F[加权邻居对比学习]
    E --> G[边特征增强解码器]
    
    F --> H[难负样本挖掘]
    H --> I[加权对比损失<br/>L_contrast]
    
    G --> J[边特征提取<br/>7维特征]
    J --> K[MLP分类器]
    K --> L[预测概率]
    
    I --> M[总损失<br/>L_total = L_BCE + λ·L_contrast]
    L --> M
    
    style B fill:#e1f5ff
    style F fill:#fff4e1
    style H fill:#ffe1f5
    style J fill:#e1ffe1
```

## 图扩散模块详细流程

```mermaid
graph LR
    A1[原始相似图<br/>A_sim, B_sim] --> B1[构建Meta图]
    B1 --> C1[A_meta = PM×PM^T + PF×PF^T<br/>B_meta = MP^T×MP + MF×MF^T]
    C1 --> D1[融合<br/>A_fused = A_sim + β·A_meta]
    D1 --> E1[归一化<br/>A_norm = D^{-1/2}·A·D^{-1/2}]
    E1 --> F1[多阶扩散<br/>S = I + α·A_norm + α²·A_norm² + α³·A_norm³]
    F1 --> G1[扩散相似图<br/>diff_A_sim, diff_B_sim]
    
    style B1 fill:#e1f5ff
    style F1 fill:#ffe1f5
```

## 加权对比学习流程

```mermaid
graph TB
    A2[节点嵌入<br/>embP, embM, embF] --> B2[聚合邻居表示]
    B2 --> C2[PM_repr<br/>PP_repr<br/>PF_repr]
    C2 --> D2[Attention MLP<br/>学习权重]
    D2 --> E2[节点特定权重<br/>w_PM, w_PP, w_PF]
    
    A2 --> F2[计算相似度]
    F2 --> G2[正样本对]
    F2 --> H2[负样本对]
    H2 --> I2[选择Hard Negatives<br/>top-30%最相似的]
    
    G2 --> J2[加权正样本]
    I2 --> K2[加权负样本<br/>Hard Neg权重=2.0]
    E2 --> J2
    E2 --> K2
    J2 --> L2[对比损失<br/>L_contrast]
    K2 --> L2
    
    style D2 fill:#fff4e1
    style I2 fill:#ffe1f5
```

## 边特征增强流程

```mermaid
graph LR
    A3[节点对<br/>protein_i, metabolite_j] --> B3[提取边特征]
    B3 --> C3[1. common_GO]
    B3 --> D3[2. protein_degree]
    B3 --> E3[3. metabolite_degree]
    B3 --> F3[4. protein_avg_sim]
    B3 --> G3[5. metabolite_avg_sim]
    B3 --> H3[6. protein_diff_sim]
    B3 --> I3[7. metabolite_diff_sim]
    
    C3 --> J3[拼接<br/>7维特征]
    D3 --> J3
    E3 --> J3
    F3 --> J3
    G3 --> J3
    H3 --> J3
    I3 --> J3
    
    J3 --> K3[Node Embeddings<br/>embP[i], embM[j]]
    K3 --> L3[边表示<br/>128+7=135维]
    L3 --> M3[MLP分类器]
    M3 --> N3[预测概率]
    
    style B3 fill:#e1ffe1
    style J3 fill:#ffe1f5
```

## 完整数据流

```mermaid
sequenceDiagram
    participant Input as 输入数据
    participant Diff as 图扩散模块
    participant Enc as 异质GAT编码器
    participant Con as 加权对比学习
    participant Dec as 边特征解码器
    participant Loss as 损失函数
    
    Input->>Diff: 原始相似图
    Diff->>Diff: 构建Meta图
    Diff->>Diff: 融合相似图+Meta图
    Diff->>Diff: 多阶扩散(K=3, α=0.2)
    Diff->>Enc: 扩散后的相似图
    
    Input->>Enc: 节点特征
    Enc->>Enc: 异质注意力聚合
    Enc->>Con: 节点嵌入
    Enc->>Dec: 节点嵌入
    
    Con->>Con: 学习自适应权重
    Con->>Con: 难负样本挖掘
    Con->>Loss: 对比损失
    
    Dec->>Dec: 提取边特征(7维)
    Dec->>Dec: MLP分类
    Dec->>Loss: 预测概率
    
    Loss->>Loss: L_total = L_BCE + λ·L_contrast
```

## 模块依赖图

```mermaid
graph TD
    A[图扩散模块] -->|输出扩散图| B[异质GAT编码器]
    B -->|输出节点嵌入| C[加权对比学习]
    B -->|输出节点嵌入| D[边特征解码器]
    
    A -->|扩散相似度| D
    
    C -->|对比损失| E[总损失]
    D -->|预测概率| F[BCE损失]
    F --> E
    
    C -.->|难负样本挖掘| C
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#e1ffe1
```

## 内存优化流程

```mermaid
graph TB
    A4[大batch<br/>200000个样本对] --> B4{是否>2000?}
    B4 -->|是| C4[分批处理<br/>每批2000个]
    B4 -->|否| D4[直接处理]
    
    C4 --> E4[Batch 1: [0:2000]]
    C4 --> F4[Batch 2: [2000:4000]]
    C4 --> G4[Batch N: [N-2000:N]]
    
    E4 --> H4[提取边特征]
    H4 --> I4[清理内存<br/>del + empty_cache]
    I4 --> J4[保存特征]
    
    F4 --> H4
    G4 --> H4
    
    J4 --> K4[拼接所有批次]
    D4 --> K4
    K4 --> L4[归一化特征]
    L4 --> M4[送入MLP]
    
    style C4 fill:#ffe1f5
    style I4 fill:#fff4e1
```

## 创新点位置图

```mermaid
graph TB
    A5[输入层] --> B5[图扩散模块<br/>⭐创新: 多阶扩散+Meta图]
    B5 --> C5[异质GAT编码器]
    C5 --> D5[加权对比学习<br/>⭐创新: 自适应权重]
    D5 --> E5[难负样本挖掘<br/>⭐⭐创新: Hard Negative Mining]
    C5 --> F5[边特征解码器<br/>⭐⭐创新: Edge Features]
    E5 --> G5[对比损失]
    F5 --> H5[预测概率]
    G5 --> I5[总损失]
    H5 --> I5
    
    style B5 fill:#e1f5ff
    style D5 fill:#fff4e1
    style E5 fill:#ffe1f5
    style F5 fill:#e1ffe1
```

