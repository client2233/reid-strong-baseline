# encoding: utf-8
"""
超参数调优脚本 — 自动运行多组实验，绘制超参数-性能对比曲线/表格
使用方法:
    python tools/hyperparameter_tuning.py --param BASE_LR --values 0.0001 0.00035 0.001
    python tools/hyperparameter_tuning.py --param WARMUP_ITERS --values 10 100 500
    python tools/hyperparameter_tuning.py --param IMS_PER_BATCH --values 32 64 128
    python tools/hyperparameter_tuning.py --param NUM_INSTANCE --values 2 4 8
"""

import argparse
import itertools
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.append('.')
from config import cfg

# 用于对比的超参数及其可选值
HYPERPARAMETERS = {
    'BASE_LR': {
        'description': 'Base Learning Rate',
        'config_path': 'SOLVER.BASE_LR',
        'values': [0.0001, 0.00035, 0.001],
    },
    'WARMUP_ITERS': {
        'description': 'Warmup Iterations',
        'config_path': 'SOLVER.WARMUP_ITERS',
        'values': [10, 100, 500],
    },
    'IMS_PER_BATCH': {
        'description': 'Batch Size',
        'config_path': 'SOLVER.IMS_PER_BATCH',
        'values': [32, 64, 128],
    },
    'NUM_INSTANCE': {
        'description': 'Number of Instances per ID',
        'config_path': 'SOLVER.NUM_INSTANCE',  # 实际上是 DATALOADER.NUM_INSTANCE
        'values': [2, 4, 8],
    },
    'WEIGHT_DECAY': {
        'description': 'Weight Decay',
        'config_path': 'SOLVER.WEIGHT_DECAY',
        'values': [0.0001, 0.0005, 0.001],
    },
}

# 参数名到实际配置路径的映射（有些参数路径不同）
PARAM_TO_CONFIG_PATH = {
    'BASE_LR': 'SOLVER.BASE_LR',
    'WARMUP_ITERS': 'SOLVER.WARMUP_ITERS',
    'IMS_PER_BATCH': 'SOLVER.IMS_PER_BATCH',
    'NUM_INSTANCE': 'DATALOADER.NUM_INSTANCE',
    'WEIGHT_DECAY': 'SOLVER.WEIGHT_DECAY',
}

# 是否需要对应调整 NUM_WORKERS
BATCH_SIZE_TO_WORKERS = {32: 4, 64: 8, 128: 12}


def run_experiment(config_file, param_name, param_value, output_dir, max_epochs=20):
    """运行一次实验"""
    config_path = PARAM_TO_CONFIG_PATH.get(param_name, f'SOLVER.{param_name}')

    exp_output_dir = os.path.join(output_dir, f'{param_name}={param_value}')
    os.makedirs(exp_output_dir, exist_ok=True)

    cmd = [
        sys.executable, 'tools/train.py',
        '--config_file', config_file,
        f'{config_path}', str(param_value),
        'OUTPUT_DIR', exp_output_dir,
        'SOLVER.MAX_EPOCHS', str(max_epochs),
    ]

    # 如果调整 batch size，同步调整 num_workers
    if param_name == 'IMS_PER_BATCH' and param_value in BATCH_SIZE_TO_WORKERS:
        cmd.extend(['DATALOADER.NUM_WORKERS', str(BATCH_SIZE_TO_WORKERS[param_value])])

    print(f"\n{'='*60}")
    print(f"Running experiment: {param_name} = {param_value}")
    print(f"Output: {exp_output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + '/..')
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"  ⚠ Experiment failed (return code {result.returncode})")
        return None

    # 读取结果
    csv_path = os.path.join(exp_output_dir, 'eval_results.csv')
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            final = rows[-1]
            print(f"  ✓ {param_name}={param_value}  |  "
                  f"mAP={float(final['mAP'])*100:.2f}%  |  "
                  f"Rank-1={float(final['rank1'])*100:.2f}%  |  "
                  f"time={elapsed:.1f}s")
            return {
                'param_value': param_value,
                'mAP': float(final['mAP']),
                'rank1': float(final['rank1']),
                'rank5': float(final['rank5']),
                'rank10': float(final['rank10']),
                'epochs': [r['epoch'] for r in rows],
                'mAP_curve': [float(r['mAP']) for r in rows],
                'rank1_curve': [float(r['rank1']) for r in rows],
                'time': elapsed,
            }

    return None


def plot_comparison(results, param_name, param_info, output_dir):
    """绘制超参数对比图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip plotting")
        return

    # 如果只有最终结果，绘制柱状图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    param_label = param_info['description']
    config_path = PARAM_TO_CONFIG_PATH.get(param_name, param_name)

    # 将参数值转为字符串标签
    labels = [str(r['param_value']) for r in results if r is not None]
    mAPs = [r['mAP'] * 100 for r in results if r is not None]
    rank1s = [r['rank1'] * 100 for r in results if r is not None]

    x = np.arange(len(labels))
    width = 0.35

    # mAP 柱状图
    axes[0].bar(x, mAPs, width, color='royalblue', alpha=0.8)
    axes[0].set_xlabel(param_label)
    axes[0].set_ylabel('mAP (%)')
    axes[0].set_title(f'{param_label} vs mAP')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    for i, v in enumerate(mAPs):
        axes[0].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)

    # Rank-1 柱状图
    axes[1].bar(x, rank1s, width, color='crimson', alpha=0.8)
    axes[1].set_xlabel(param_label)
    axes[1].set_ylabel('Rank-1 Accuracy (%)')
    axes[1].set_title(f'{param_label} vs Rank-1')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    for i, v in enumerate(rank1s):
        axes[1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    # 如果有曲线数据，画 epoch-mAP 对比曲线
    has_curves = all(r.get('mAP_curve') for r in results if r is not None)
    if has_curves:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
        for i, r in enumerate(results):
            if r is None:
                continue
            axes[2].plot(r['epochs'], [v * 100 for v in r['mAP_curve']],
                        'o-', color=colors[i], label=f'{param_name}={r["param_value"]}',
                        alpha=0.7, markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('mAP (%)')
        axes[2].set_title(f'{param_label} — Training mAP Curves')
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No curve data available',
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title(f'{param_label} Comparison')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'hparam_{param_name}_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved to {save_path}")

    # 生成表格 (Markdown)
    table_path = os.path.join(output_dir, f'hparam_{param_name}_table.md')
    with open(table_path, 'w') as f:
        f.write(f'# Hyperparameter Comparison: {param_label} (`{config_path}`)\n\n')
        f.write('| Param Value | mAP | Rank-1 | Rank-5 | Rank-10 |\n')
        f.write('|-------------|-----|--------|--------|--------|\n')
        for r in results:
            if r is None:
                continue
            f.write(f'| {r["param_value"]} '
                    f'| {r["mAP"]*100:.2f}% '
                    f'| {r["rank1"]*100:.2f}% '
                    f'| {r["rank5"]*100:.2f}% '
                    f'| {r["rank10"]*100:.2f}% |\n')
    print(f"Comparison table saved to {table_path}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for ReID')
    parser.add_argument('--config_file', type=str, default='configs/softmax_triplet.yml',
                      help='Base config file')
    parser.add_argument('--param', type=str, required=True,
                      choices=list(HYPERPARAMETERS.keys()),
                      help='Hyperparameter to tune')
    parser.add_argument('--values', type=float, nargs='+', default=None,
                      help='Values to try (overrides defaults)')
    parser.add_argument('--output_dir', type=str, default='../hp_tuning_outputs',
                      help='Root output directory for tuning results')
    parser.add_argument('--max_epochs', type=int, default=20,
                      help='Number of epochs per experiment (default: 20 for quick comparison)')
    args = parser.parse_args()

    # 获取参数信息
    param_info = HYPERPARAMETERS[args.param]
    values = args.values if args.values else param_info['values']

    print(f"{'='*60}")
    print(f"Hyperparameter Tuning: {param_info['description']} ({args.param})")
    print(f"Values to test: {values}")
    print(f"Config file: {args.config_file}")
    print(f"Max epochs per experiment: {args.max_epochs}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for val in values:
        result = run_experiment(args.config_file, args.param, val,
                              args.output_dir, max_epochs=args.max_epochs)
        results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Param Value':>12} | {'mAP':>8} | {'Rank-1':>8} | {'Time':>8}")
    print("-" * 45)
    for r in results:
        if r is None:
            continue
        print(f"{r['param_value']:>12} | {r['mAP']*100:>7.2f}% | {r['rank1']*100:>7.2f}% | {r['time']:>7.1f}s")
    print("-" * 45)

    # 绘图
    plot_comparison(results, args.param, param_info, args.output_dir)


if __name__ == '__main__':
    main()
