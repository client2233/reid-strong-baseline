#!/bin/bash
# ============================================================================
# 注意力机制消融实验 — 支持轻量化模式
# ============================================================================
# 使用方法:
#   完整训练: bash tools/run_attention_ablation.sh
#   轻量:     bash tools/run_attention_ablation.sh --lightweight
#   组1:      bash tools/run_attention_ablation.sh --group 1
#   极速:     bash tools/run_attention_ablation.sh --fast
# ============================================================================
# 注意: 始终从项目根目录运行
# ============================================================================

set -e

# ---------- 参数解析 ----------
GROUP="all"
EPOCHS=60
GPU="0"
DATASET="market1501"
BATCH_SIZE=64
EVAL_PERIOD=5
LIGHTWEIGHT=false
FAST=false

while [[ $# -gt 0 ]]; do
  case "$1" in
  --group)
    GROUP="$2"
    shift 2
    ;;
  --epochs)
    EPOCHS="$2"
    shift 2
    ;;
  --gpu)
    GPU="$2"
    shift 2
    ;;
  --dataset)
    DATASET="$2"
    shift 2
    ;;
  --batch)
    BATCH_SIZE="$2"
    shift 2
    ;;
  --lightweight)
    LIGHTWEIGHT=true
    shift
    ;;
  --fast)
    FAST=true
    shift
    ;;
  *)
    echo "未知参数: $1"
    exit 1
    ;;
  esac
done

# 切换到项目根目录
cd "$(dirname "$0")/.."
echo "工作目录: $(pwd)"

TRAIN_SCRIPT="python tools/train.py"
SUMMARY_SCRIPT="python tools/summarize_results.py"

# ========== 构建公共参数 ==========
COMMON_ARGS=(
  MODEL.DEVICE_ID "$GPU"
  SOLVER.EVAL_PERIOD "$EVAL_PERIOD"
  SOLVER.LOG_PERIOD 50
  DATASETS.NAMES "$DATASET"
  MODEL.PRETRAIN_CHOICE "imagenet"
  MODEL.NECK "bnneck"
  MODEL.IF_WITH_CENTER "no"
)

if [ "$LIGHTWEIGHT" = true ] || [ "$FAST" = true ]; then
  echo ""
  echo "  ⚡ 轻量化模式激活"

  # 使用 lightweight.yml 作为基础配置
  COMMON_ARGS=(
    --config_file "configs/lightweight.yml"
    "${COMMON_ARGS[@]}"
  )

  if [ "$FAST" = true ]; then
    COMMON_ARGS+=(
      SOLVER.MAX_EPOCHS 10
      DATALOADER.SUBSET_RATIO 0.25
    )
    echo "  ⚡ 极速调试模式: 10 epoch, 25% 数据"
  else
    COMMON_ARGS+=(
      SOLVER.MAX_EPOCHS 30
    )
    echo "  ⚡ 轻量模式: 30 epoch, 50% 数据, 图像 192x96"
  fi
else
  COMMON_ARGS+=(
    SOLVER.MAX_EPOCHS "$EPOCHS"
    SOLVER.IMS_PER_BATCH "$BATCH_SIZE"
  )
fi

echo ""
echo "=========================================="
echo "  注意力机制消融实验"
echo "  数据集: $DATASET  |  GPU: $GPU"
if [ "$LIGHTWEIGHT" = true ]; then
  echo "  模式: 轻量化"
elif [ "$FAST" = true ]; then
  echo "  模式: 极速调试"
else
  echo "  模式: 完整训练 ($EPOCHS epoch)"
fi
echo "=========================================="
echo ""

# ====================================================================
# 组1: 三种架构对比
# ====================================================================
run_group1() {
  echo ">>> ========================================="
  echo ">>> 组1: 三种架构对比"
  echo ">>>   A) ResNet50 + CBAM"
  echo ">>>   B) ResNet50 + Transformer插件"
  echo ">>>   C) ViT-Base (纯Transformer)"
  echo ">>> ========================================="
  echo ""

  echo "------- [1A] ResNet50 + CBAM @ layer3 -------"
  $TRAIN_SCRIPT \
    --exp_name "1A_resnet50_cbam_l3" \
    "${COMMON_ARGS[@]}" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_layer3"

  echo "------- [1B] ResNet50 + Transformer @ layer3 -------"
  $TRAIN_SCRIPT \
    --exp_name "1B_resnet50_transformer_l3" \
    "${COMMON_ARGS[@]}" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "transformer" \
    MODEL.ATTENTION.POSITION "after_layer3"

  echo "------- [1C] ViT-Base/16 (纯Transformer) -------"
  if [ "$LIGHTWEIGHT" = true ] || [ "$FAST" = true ]; then
    VIT_OPTS=(SOLVER.IMS_PER_BATCH 8)
  else
    VIT_OPTS=(SOLVER.IMS_PER_BATCH 32)
  fi
  $TRAIN_SCRIPT \
    --exp_name "1C_vit_base_patch16" \
    "${COMMON_ARGS[@]}" \
    "${VIT_OPTS[@]}" \
    MODEL.NAME "vit_base_patch16" \
    MODEL.ATTENTION.TYPE "none"

  echo ""
  echo ">>> 组1 完成！"
  echo ""
}

# ====================================================================
# 组2: CBAM 位置消融
# ====================================================================
run_group2() {
  echo ">>> ========================================="
  echo ">>> 组2: CBAM 在不同位置的消融实验"
  echo ">>>   A) after_layer3"
  echo ">>>   B) after_layer4"
  echo ">>>   C) after_all"
  echo ">>> ========================================="
  echo ""

  echo "------- [2A] CBAM @ layer3 -------"
  $TRAIN_SCRIPT \
    --exp_name "2A_cbam_l3" \
    "${COMMON_ARGS[@]}" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_layer3"

  echo "------- [2B] CBAM @ layer4 -------"
  $TRAIN_SCRIPT \
    --exp_name "2B_cbam_l4" \
    "${COMMON_ARGS[@]}" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_layer4"

  echo "------- [2C] CBAM @ all layers -------"
  $TRAIN_SCRIPT \
    --exp_name "2C_cbam_all" \
    "${COMMON_ARGS[@]}" \
    MODEL.NAME "resnet50" \
    MODEL.ATTENTION.TYPE "cbam" \
    MODEL.ATTENTION.POSITION "after_all"

  echo ""
  echo ">>> 组2 完成！"
  echo ""
}

# ====================================================================
# 执行
# ====================================================================
case "$GROUP" in
1) run_group1 ;;
2) run_group2 ;;
all)
  run_group1
  run_group2
  ;;
*)
  echo "错误: --group 只能是 1, 2, 或 all"
  exit 1
  ;;
esac

# ====================================================================
# 汇总
# ====================================================================
echo ""
echo "=========================================="
echo "  所有实验完成！汇总对比:"
echo "=========================================="
echo ""

if [ "$LIGHTWEIGHT" = true ] || [ "$FAST" = true ]; then
  OUT_DIR="outputs_light"
else
  OUT_DIR="outputs"
fi

$SUMMARY_SCRIPT --output_dir "$OUT_DIR"

echo ""
echo "=========================================="
echo "  实验输出目录: $OUT_DIR"
echo "  目录结构:     find $OUT_DIR -maxdepth 2 -type d"
echo "  LaTeX表格:    python tools/summarize_results.py --output_dir $OUT_DIR --latex"
echo "=========================================="
