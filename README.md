# Bag of Tricks and A Strong ReID Baseline — 中文版

基于 [Bag of Tricks and a Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf) (CVPRW2019, Oral) 的 PyTorch 实现。

本项目在原作基础上扩展了**多种注意力机制**、**Vision Transformer 骨干网络**、**实验数据自动追踪系统**和**轻量化训练模式**。

---

## 📋 目录

- [新增特性](#-新增特性)
- [实验结果对比](#-实验结果对比)
- [环境配置](#-环境配置)
- [数据集准备](#-数据集准备)
- [快速开始](#-快速开始)
- [训练命令](#-训练命令)
- [测试命令](#-测试命令)
- [实验数据与报告](#-实验数据与报告)
- [项目结构](#-项目结构)
- [引用](#-引用)

---

## 🚀 新增特性

### 1. 注意力机制（7 种）

在 `modeling/backbones/attention.py` 中实现，可作为插件插入 ResNet 各层之后。

| 类型 | 配置值 | 核心思想 | 参考论文 |
|------|--------|---------|---------|
| Non-local Block | `non_local` | 自注意力捕获长距离空间依赖 | CVPR 2018 |
| CBAM | `cbam` | 通道注意力 + 空间注意力串联 | ECCV 2018 |
| Coordinate Attention | `coord_att` | 编码方向感知的位置信息 | CVPR 2021 |
| External Attention | `external_att` | 外部记忆单元，线性复杂度 O(N) | 2020 |
| Gather-Excite | `gather_excite` | 上下文聚合增强特征表达 | NeurIPS 2019 |
| **Transformer Encoder** | **`transformer`** | **MHSA + FFN 标准 Transformer** | **NeurIPS 2017** |

**插入位置：**
- `after_layer3` — 在 layer3 之后（推荐，计算量适中）
- `after_layer4` — 在 layer4 之后（最小计算量）
- `after_all` — 每个阶段后都插入（最大计算量，最高性能）

### 2. Vision Transformer 骨干网络

在 `modeling/backbones/vit.py` 中实现，可直接替代 ResNet。

| 模型 | embed_dim | depth | heads | patch_size |
|------|-----------|-------|-------|-----------|
| `vit_tiny_patch16` | 192 | 12 | 3 | 16 |
| `vit_small_patch16` | 384 | 12 | 6 | 16 |
| `vit_base_patch16` | **768** | **12** | **12** | **16** |
| `vit_large_patch16` | 1024 | 24 | 16 | 16 |

兼容 Baseline 的 GAP + BNNeck + Classifier 管线，支持 ImageNet 预训练权重。

### 3. 实验数据自动追踪

训练完成后自动在输出目录生成：

| 文件 | 内容 | 用途 |
|------|------|------|
| `training_log.csv` | 每 epoch 的 loss / acc / lr / mAP / Rank-1/5/10 | 导入 Excel/Origin 自定义绘图 |
| `experiment_summary.json` | 最佳指标 + 训练统计摘要 | 快速提取实验结论 |
| `training_curves.png` | Loss + Train Acc + mAP + Rank-1 四合一图 | 直接插入实验报告 |
| `lr_schedule.png` | 学习率变化曲线 | 分析优化器行为 |

### 4. 轻量化训练模式

为快速验证和资源受限环境优化，使用 `--lightweight` 或 `--fast` 标志：

| 模式 | epoch | 数据量 | 图像尺寸 | Batch | 速度 |
|------|-------|--------|---------|-------|------|
| 完整 | 60 | 100% | 384×128 | 64 | ~4h/组 |
| **轻量** | **30** | **50%** | **192×96** | **16** | **~30min/组** |
| 极速 | 10 | 25% | 192×96 | 16 | ~5min/组 |

---

## 📊 实验结果对比

### 组1：三种架构对比（轻量模式）

| 实验 | 配置 | Best mAP | Best Rank-1 |
|------|------|----------|-------------|
| ResNet50 + CBAM | `att_type=cbam` | — | — |
| ResNet50 + Transformer | `att_type=transformer` | — | — |
| ViT-Base/16 | `model=vit_base_patch16` | — | — |

### 组2：CBAM 位置消融

| 实验 | 位置 | Best mAP | Best Rank-1 |
|------|------|----------|-------------|
| CBAM @ layer3 | `after_layer3` | — | — |
| CBAM @ layer4 | `after_layer4` | — | — |
| CBAM @ all | `after_all` | — | — |

> 运行 `bash tools/run_attention_ablation.sh --lightweight` 后，用 `python tools/summarize_results.py --output_dir outputs_light` 查看结果。

---

## 🔧 环境配置

```bash
# 创建 conda 环境
conda create -n reid python=3.14
conda activate reid

# 安装依赖
pip install torch torchvision
pip install ignite==0.1.2
pip install yacs
pip install matplotlib

# 克隆项目
git clone <your-repo-url>
cd reid-strong-baseline
```

---

## 📁 数据集准备

```bash
mkdir data
```

### Market1501

1. 从 http://www.liangzheng.org/Project/project_reid.html 下载
2. 解压到 `data/` 并重命名为 `market1501`

```
data/
    market1501/
        bounding_box_test/
        bounding_box_train/
        query/
        ...
```

### DukeMTMC-reID

1. 从 https://github.com/layumi/DukeMTMC-reID_evaluation 下载
2. 解压到 `data/` 并重命名为 `dukemtmc-reid`

---

## 🏃 快速开始

```bash
cd reid-strong-baseline

# 轻量化模式（推荐快速验证）
python tools/train.py \
    --config_file configs/lightweight.yml \
    --exp_name "my_experiment" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_layer3" \
    MODEL.DEVICE_ID 0

# 完整训练模式
python tools/train.py \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "transformer" \
    MODEL.ATTENTION.POSITION "after_layer3" \
    SOLVER.MAX_EPOCHS 60 \
    SOLVER.IMS_PER_BATCH 64 \
    MODEL.DEVICE_ID 0
```

---

## 🎯 训练命令

### 单组实验

```bash
# ResNet50 baseline
python tools/train.py --exp_name "resnet50_baseline" MODEL.NAME "resnet50" MODEL.ATTENTION.TYPE "none"

# ResNet50 + CBAM
python tools/train.py --exp_name "resnet50_cbam_l3" MODEL.NAME "resnet50" MODEL.ATTENTION.TYPE "cbam" MODEL.ATTENTION.POSITION "after_layer3"

# ResNet50 + Transformer 插件
python tools/train.py --exp_name "resnet50_transformer_l3" MODEL.NAME "resnet50" MODEL.ATTENTION.TYPE "transformer" MODEL.ATTENTION.POSITION "after_layer3"

# ViT-Base (纯Transformer)
python tools/train.py --exp_name "vit_base" --config_file configs/lightweight.yml MODEL.NAME "vit_base_patch16" MODEL.ATTENTION.TYPE "none" SOLVER.IMS_PER_BATCH 8
```

### 消融实验脚本

```bash
# 全部跑完
bash tools/run_attention_ablation.sh

# 轻量化模式（推荐）
bash tools/run_attention_ablation.sh --lightweight

# 只跑组1（架构对比）
bash tools/run_attention_ablation.sh --lightweight --group 1

# 只跑组2（CBAM位置消融）
bash tools/run_attention_ablation.sh --lightweight --group 2

# 极速调试
bash tools/run_attention_ablation.sh --fast --group 1

# 自定义参数
bash tools/run_attention_ablation.sh --lightweight --epochs 20 --gpu 1 --dataset "dukemtmcreid"
```

### `--exp_name` 说明

- 不指定 `--exp_name`：自动按 `{骨干网络}_{注意力类型}_{位置}` 命名
- 指定 `--exp_name`：使用自定义名称
- 输出目录：完整模式 → `outputs/{exp_name}/`，轻量模式 → `outputs_light/{exp_name}/`

---

## 🔬 测试命令

```bash
# 用欧氏距离 + BN前特征（无 re-ranking）
python tools/test.py \
    --config_file configs/lightweight.yml \
    MODEL.DEVICE_ID 0 \
    DATASETS.NAMES "market1501" \
    TEST.NECK_FEAT "before" \
    TEST.FEAT_NORM "no" \
    MODEL.PRETRAIN_CHOICE "self" \
    TEST.WEIGHT "outputs_light/vit_lightweight/vit_base_patch16_*.pth"

# 用余弦距离 + BN后特征（无 re-ranking）
python tools/test.py \
    --config_file configs/lightweight.yml \
    MODEL.DEVICE_ID 0 \
    DATASETS.NAMES "market1501" \
    TEST.NECK_FEAT "after" \
    TEST.FEAT_NORM "yes" \
    MODEL.PRETRAIN_CHOICE "self" \
    TEST.WEIGHT "outputs_light/vit_lightweight/vit_base_patch16_*.pth"
```

---

## 📈 实验数据与报告

### 查看结果汇总

```bash
# 完整模式结果
python tools/summarize_results.py --output_dir outputs

# 轻量模式结果
python tools/summarize_results.py --output_dir outputs_light

# 输出 LaTeX 表格（直接贴论文）
python tools/summarize_results.py --output_dir outputs_light --latex
```

### 输出示例

```
实验配置                           Best mAP    Best R1  Final mAP  Final R1
────────────────────────────────────────────────────────────────────────────
1A_resnet50_cbam_l3                 85.2%     91.8%      84.7%     91.2%
1B_resnet50_transformer_l3          87.6%     93.5%      87.1%     93.0%
1C_vit_base_patch16                 88.9%     94.1%      88.3%     93.6%
2A_cbam_l3                          85.2%     91.8%      84.7%     91.2%
2B_cbam_l4                          84.1%     90.5%      83.6%     90.1%
2C_cbam_all                         85.8%     92.3%      85.2%     91.9%
🏆 最佳 mAP:    1C_vit_base_patch16 = 88.9% @ epoch 55
🏆 最佳 Rank-1: 1C_vit_base_patch16 = 94.1% @ epoch 50
```

### 实验输出目录结构

```
outputs_light/                          # 轻量模式根目录
├── 1A_resnet50_cbam_l3/
│   ├── training_log.csv                # ← 每 epoch 完整数据
│   ├── experiment_summary.json         # ← 最佳指标摘要
│   ├── training_curves.png             # ← 四合一图表
│   ├── lr_schedule.png                 # ← 学习率曲线
│   ├── log.txt                         # ← 训练日志
│   └── resnet50_*.pth                  # ← 模型检查点
├── 1B_resnet50_transformer_l3/
├── 1C_vit_base_patch16/
├── 2A_cbam_l3/
├── 2B_cbam_l4/
└── 2C_cbam_all/
```

---

## 📁 项目结构

```
reid-strong-baseline/
├── config/
│   ├── defaults.py              # 默认配置（含注意力配置项）
│   └── __init__.py
├── configs/
│   ├── lightweight.yml          # 轻量化训练配置
│   ├── baseline.yml
│   ├── softmax_triplet.yml
│   └── ...
├── data/                        # 数据集目录
├── modeling/
│   ├── baseline.py              # 主模型（支持 CNN + ViT + 注意力）
│   ├── __init__.py              # build_model 入口
│   └── backbones/
│       ├── resnet.py            # ResNet（支持注意力插件）
│       ├── vit.py               # ★ Vision Transformer 骨干
│       ├── attention.py         # ★ 7种注意力机制模块
│       ├── senet.py             # SE-Net
│       └── resnet_ibn_a.py      # IBN-Net
├── engine/
│   └── trainer.py               # ★ 实验数据追踪系统
├── tools/
│   ├── train.py                 # 训练入口（支持 --exp_name）
│   ├── test.py                  # 测试入口
│   ├── run_attention_ablation.sh# ★ 消融实验脚本
│   └── summarize_results.py     # ★ 结果汇总工具
├── utils/
│   └── logger.py                # 日志
└── README.md
```

> ★ 标记为本项目新增或重点修改的文件。

---

## 📚 引用

```bibtex
@InProceedings{Luo_2019_CVPR_Workshops,
  author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2019}
}

@ARTICLE{Luo_2019_Strong_TMM,
  author={H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}},
  journal={IEEE Transactions on Multimedia},
  title={A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification},
  year={2019},
  doi={10.1109/TMM.2019.2958756}
}
```
