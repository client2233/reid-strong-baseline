#!/bin/bash
# ============================================================================
# ReID 注意力机制对比实验脚本
# 自动运行多组实验，结果按参数分目录保存到 outputs/ 下
# 使用方法: bash tools/run_comparison.sh
# ============================================================================

set -e  # 出错时停止

# ========== 通用配置 ==========
DATASET="market1501"
DEVICE_ID="0"
EPOCHS=60
EVAL_PERIOD=5  # 每5个epoch评估一次，获得更平滑的曲线
LOG_PERIOD=50
BATCH_SIZE=64
OUTPUT_BASE="../outputs"

echo "=========================================="
echo "  ReID 注意力机制对比实验"
echo "  数据集: $DATASET"
echo "  GPU: $DEVICE_ID"
echo "  训练轮数: $EPOCHS"
echo "=========================================="
echo ""

# ========== 实验组 1: CNN Baseline vs 各种注意力插件 ==========
echo ">>> [实验组1] CNN Baseline vs 注意力插件"
echo ""

# 1.1 Baseline (无注意力) —— 对照组
echo "------- 实验 1.1: resnet50_baseline -------"
python train.py \
    --exp_name "resnet50_baseline" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "none" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 1.2 ResNet50 + CBAM
echo "------- 实验 1.2: resnet50 + CBAM -------"
python train.py \
    --exp_name "resnet50_cbam_l3" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_layer3" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 1.3 ResNet50 + Coordinate Attention
echo "------- 实验 1.3: resnet50 + CoordAtt -------"
python train.py \
    --exp_name "resnet50_coordatt_l3" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "coord_att" \
    MODEL.ATTENTION.POSITION "after_layer3" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 1.4 ResNet50 + Non-local
echo "------- 实验 1.4: resnet50 + NonLocal -------"
python train.py \
    --exp_name "resnet50_nonlocal_l3" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "non_local" \
    MODEL.ATTENTION.POSITION "after_layer3" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 1.5 ResNet50 + TransformerEncoder (CNN+Transformer混合)
echo "------- 实验 1.5: resnet50 + Transformer -------"
python train.py \
    --exp_name "resnet50_transformer_l3" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "transformer" \
    MODEL.ATTENTION.POSITION "after_layer3" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"


# ========== 实验组 2: Transformer 注意力位置对比 ==========
echo ""
echo ">>> [实验组2] Transformer 注意力位置对比"
echo ""

# 2.1 Transformer after_layer3
echo "------- 实验 2.1: Transformer @ layer3 -------"
python train.py \
    --exp_name "resnet50_transformer_l3" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "transformer" \
    MODEL.ATTENTION.POSITION "after_layer3" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 2.2 Transformer after_layer4
echo "------- 实验 2.2: Transformer @ layer4 -------"
python train.py \
    --exp_name "resnet50_transformer_l4" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "transformer" \
    MODEL.ATTENTION.POSITION "after_layer4" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 2.3 Transformer after_all (最强配置)
echo "------- 实验 2.3: Transformer @ all layers -------"
python train.py \
    --exp_name "resnet50_transformer_all" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "transformer" \
    MODEL.ATTENTION.POSITION "after_all" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"


# ========== 实验组 3: 纯 Transformer (ViT) 对比 ==========
echo ""
echo ">>> [实验组3] 纯 Transformer 骨干网络对比"
echo ""

# 3.1 ViT-Tiny (轻量)
echo "------- 实验 3.1: vit_tiny_patch16 -------"
python train.py \
    --exp_name "vit_tiny_patch16" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "vit_tiny_patch16" \
    MODEL.ATTENTION.TYPE "none" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 3.2 ViT-Small
echo "------- 实验 3.2: vit_small_patch16 -------"
python train.py \
    --exp_name "vit_small_patch16" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "vit_small_patch16" \
    MODEL.ATTENTION.TYPE "none" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 3.3 ViT-Base (标准ViT)
echo "------- 实验 3.3: vit_base_patch16 -------"
python train.py \
    --exp_name "vit_base_patch16" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "vit_base_patch16" \
    MODEL.ATTENTION.TYPE "none" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH 32 \
    DATASETS.NAMES "$DATASET"


# ========== 实验组 4: 消融实验 — 不同注意力机制排布 ==========
echo ""
echo ">>> [实验组4] 消融实验 — 注意力排布位置"
echo ""

# 4.1 CBAM after_layer4 (最轻量)
echo "------- 实验 4.1: CBAM @ layer4 -------"
python train.py \
    --exp_name "resnet50_cbam_l4" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_layer4" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 4.2 CBAM after_all (最强CBAM)
echo "------- 实验 4.2: CBAM @ all layers -------"
python train.py \
    --exp_name "resnet50_cbam_all" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_all" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"

# 4.3 Coordinate Attention after_all
echo "------- 实验 4.3: CoordAtt @ all layers -------"
python train.py \
    --exp_name "resnet50_coordatt_all" \
    MODEL.DEVICE_ID "$DEVICE_ID" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "coord_att" \
    MODEL.ATTENTION.POSITION "after_all" \
    SOLVER.MAX_EPOCHS "$EPOCHS" \
    SOLVER.EVAL_PERIOD "$EVAL_PERIOD" \
    SOLVER.LOG_PERIOD "$LOG_PERIOD" \
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE" \
    DATASETS.NAMES "$DATASET"


# ========== 汇总结果 ==========
echo ""
echo "=========================================="
echo "  所有实验完成！"
echo "=========================================="
echo ""
echo "输出目录结构:"
echo "  $OUTPUT_BASE/"
echo "  ├── resnet50_baseline/"
echo "  │   ├── training_log.csv"
echo "  │   ├── training_curves.png"
echo "  │   ├── experiment_summary.json"
echo "  │   └── ..."
echo "  ├── resnet50_cbam_l3/"
echo "  ├── resnet50_coordatt_l3/"
echo "  ├── resnet50_nonlocal_l3/"
echo "  ├── resnet50_transformer_l3/"
echo "  ├── resnet50_transformer_l4/"
echo "  ├── resnet50_transformer_all/"
echo "  ├── resnet50_cbam_all/"
echo "  ├── resnet50_coordatt_all/"
echo "  ├── vit_tiny_patch16/"
echo "  ├── vit_small_patch16/"
echo "  └── vit_base_patch16/"
echo ""
echo "快速查看最佳结果:"
echo "  grep -h 'best_' */experiment_summary.json"
echo ""
echo "对比所有实验的最终指标:"
echo "  python -c \"import json, glob;"
echo "    for f in sorted(glob.glob('$OUTPUT_BASE/*/experiment_summary.json')):"
echo "      d=json.load(open(f));"
echo "      name=f.split('/')[-2];"
echo "      b=d.get('best_metrics',{});"
echo "      print(f'{name:35s} mAP={b[\\\"best_mAP\\\"]:.2%}  Rank-1={b[\\\"best_rank1\\\"]:.2%}');"
echo "    \""
