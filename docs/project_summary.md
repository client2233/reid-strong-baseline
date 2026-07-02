# 行人重识别 (Person Re-identification) 项目总结

## 1. 任务设定

### 1.1 任务描述
**行人重识别（Person Re-identification, ReID）** 是指在非重叠视角下的监控网络中，判断不同摄像头拍摄到的图像是否属于同一个行人的技术。本项目的目标是在 **Market1501** 数据集上训练一个行人重识别模型。

### 1.2 数据集：Market1501
| 属性 | 数值 |
|------|------|
| 训练集身份数 (train ids) | 751 |
| 训练集图片数 (train images) | 12,936 |
| 查询集身份数 (query ids) | 750 |
| 查询集图片数 (query images) | 3,368 |
| 图库集身份数 (gallery ids) | 751 |
| 图库集图片数 (gallery images) | 15,913 |
| 摄像头数量 | 6 |

### 1.3 评估指标
- **mAP (mean Average Precision)**：所有查询样本的平均精度均值，反映整体检索性能
- **CMC Rank-1 / Rank-5 / Rank-10**：Top-K 命中率，衡量首位匹配的准确率

---

## 2. 模型结构

### 2.1 整体架构
```
Input Image
       ↓
  Backbone (ResNet18 / ResNet50, 去除 FC 层)
       ↓
   Global Avg Pooling (GAP)
       ↓
    d 维特征向量 (d=512 for ResNet18, d=2048 for ResNet50)
       ↓
 ┌─── BNNeck (BatchNorm1d) ───┐
 │         ↓                   │
 │    d 维特征 (feat)          │
 │         ↓                   │
 │   Linear Classifier         │
 │         ↓                   │
 │   class scores (751类)       │
 └─────────────────────────────┘
```

### 2.2 骨干网络

#### ResNet50（完整版）
- 采用 ImageNet 预训练的 ResNet50，参数量 **25.6M**
- 输出 2048 维特征向量
- 将最后一个 stage 的 stride 改为 1（`LAST_STRIDE=1`），获得更大分辨率的特征图

#### ResNet18（轻量版）
- 采用 ImageNet 预训练的 ResNet18，参数量 **11.2M**（仅为 ResNet50 的 44%）
- 输出 512 维特征向量
- 计算量大幅降低，适合 GPU 显存有限的场景（如 8GB 的 RTX 4060 Laptop）
- 预训练权重从 torchvision 自动下载

### 2.3 BNNeck
- 在全局特征后添加 Batch Normalization 层
- **作用**：缓解 triplet loss 和 cross-entropy loss 在特征空间上的冲突
  - Triplet loss 需要在欧氏空间中进行距离度量
  - Cross-entropy loss 需要特征的各个维度分布相对均匀
  - BNNeck 将特征映射到更适合各自损失函数的空间

---

## 3. 方法

### 3.1 损失函数

#### 3.1.1 Cross-Entropy Loss（ID Loss）
用于分类任务，将行人重识别视为身份分类问题：

$$L_{ID} = -\sum_{i=1}^{N} q_i \log(p_i)$$

其中 $q_i$ 是经过 **标签平滑（Label Smoothing）** 后的真实分布：
$$q_i = \begin{cases} 1 - \frac{N-1}{N} \times \epsilon & \text{if } i = y \\ \frac{\epsilon}{N} & \text{otherwise} \end{cases}$$

- 标签平滑防止模型过于自信，提升泛化能力

#### 3.1.2 Triplet Loss（度量学习）
在批次中挖掘难例三元组（Hard Triplet Mining）：

$$L_{Triplet} = \sum_{i=1}^{P} \sum_{a=1}^{K} \left[ \alpha + \max_{p=1\dots K} \|f_a^{(i)} - f_p^{(i)}\|_2 - \min_{\substack{j=1\dots P \\ j \neq i \\ n=1\dots K}} \|f_a^{(i)} - f_n^{(j)}\|_2 \right]_+$$

- $P$：身份数，$K$：每个身份的样本数
- 拉近相同身份（正样本对），推远不同身份（负样本对）
- 使用 **Batch Hard** 采样策略：选择最困难的正样本和负样本

### 3.2 训练技巧（Bag of Tricks）

| 技巧 | 说明 |
|------|------|
| **Warmup Learning Rate** | 训练初期使用较小的学习率，逐步增加到目标值，防止早期震荡 |
| **Random Erasing** | 随机擦除图像中的矩形区域，增强模型鲁棒性 |
| **Label Smoothing** | 软化真实标签分布，减少过拟合 |
| **BNNeck** | 分离特征用于 triplet loss 和分类 loss |
| **Last Stride=1** | 保留更大尺寸的特征图，提升细粒度特征 |
| **Adam Optimizer** | 自适应学习率优化器，收敛更快 |
| **数据子集采样** | 按身份均衡比例采样，快速验证实验流程 |

---

## 4. 训练配置

### 4.1 两种配置对比

项目提供两套配置，可根据硬件资源选择：

| 配置项 | `softmax_triplet.yml`（完整版） | `lightweight.yml`（轻量版） |
|--------|-------------------------------|---------------------------|
| 骨干网络 | ResNet50 (25.6M 参数) | **ResNet18** (11.2M 参数) |
| 特征维度 | 2048 | **512** |
| 输入图像尺寸 | 256 × 128 | **192 × 96** |
| 批次大小 (IMS_PER_BATCH) | 64 | **16** |
| 每身份样本数 (NUM_INSTANCE) | 4 | **2** |
| 最大 Epoch | 120 | **30** |
| 训练数据比例 | 100% | **50%** (SUBSET_RATIO=0.5) |
| 数据加载线程 | 8 | **4** |
| 学习率衰减节点 | [40, 70] | **[10, 20]** |
| 保存周期 | 50 | **10** |
| 验证周期 | 50 | **5** |
| **预估显存需求** | ~6-8 GB | **~2-3 GB** |
| **30 epoch 训练时间** | ~30 分钟 | **~4 分钟** |

### 4.2 完整版配置（softmax_triplet.yml）
| 超参数 | 默认值 |
|--------|--------|
| 优化器 | Adam (lr=3.5e-4) |
| 批次大小 (IMS_PER_BATCH) | 64 |
| 每个身份样本数 (NUM_INSTANCE) | 4 |
| 输入图像尺寸 | 256×128 |
| 最大 Epoch | 120 |
| 学习率衰减 | StepLR，在 epoch 40 和 70 衰减 0.1 倍 |
| Warmup | 线性 warmup，10 个 iteration |
| Triplet Margin | 0.3 |

### 4.3 轻量版配置（lightweight.yml）
| 超参数 | 默认值 |
|--------|--------|
| 优化器 | Adam (lr=3.5e-4) |
| 批次大小 (IMS_PER_BATCH) | 16 |
| 每个身份样本数 (NUM_INSTANCE) | 2 |
| 输入图像尺寸 | 192×96 |
| 最大 Epoch | 30 |
| 学习率衰减 | StepLR，在 epoch 10 和 20 衰减 0.1 倍 |
| Warmup | 线性 warmup，10 个 iteration |
| 训练数据比例 | 50% (按身份均衡采样) |
| **适用场景** | **RTX 4060 Laptop (8GB) 及以下** |

---

## 5. 实验结果

训练过程中的 mAP 和 Rank-1 曲线保存在输出目录的 `training_curve.png`，详细评估结果保存在 `eval_results.csv`。

### 预期性能（Market1501）
| 指标 | ResNet50（完整版） | ResNet18（轻量版） |
|------|-------------------|-------------------|
| mAP | ~85% | ~75-80%（预期） |
| Rank-1 | ~94% | ~88-92%（预期） |

> 注：轻量版使用更少的参数、更小的图像和更少的数据，性能会有所下降，但训练速度快 5-10 倍，适合快速迭代和流程验证。

---

## 6. 代码结构

```
├── config/                 # 配置文件
│   └── defaults.py         # 默认配置（含 SUBSET_RATIO）
├── configs/                # 实验配置 YAML
│   ├── softmax_triplet.yml # 完整版配置
│   ├── lightweight.yml     # 轻量版配置（适合 4060 Laptop）
│   └── ...
├── data/                   # 数据加载
│   ├── build.py            # 数据加载 + 子集采样逻辑
│   ├── datasets/           # 数据集处理
│   └── samplers.py         # 采样策略
├── engine/                 # 训练/评估引擎
│   ├── trainer.py          # 训练逻辑 + epoch-mAP 曲线绘制
│   └── inference.py        # 推理评估
├── layers/                 # 损失函数
├── modeling/               # 模型定义
│   ├── baselines.py        # 基线模型（支持自动下载预训练权重）
│   └── backbones/          # 骨干网络
├── solver/                 # 优化器和学习率调度
├── tools/                  # 入口脚本
│   ├── train.py            # 训练入口
│   ├── test.py             # 测试入口
│   └── hyperparameter_tuning.py  # 超参数调优脚本
└── utils/                  # 工具函数
```
