# encoding: utf-8
"""
@description: 对比实验结果汇总工具
   扫描 outputs/ 下所有实验目录，汇总指标到表格和 LaTeX 格式

使用方法:
    python tools/summarize_results.py                          # 打印汇总表
    python tools/summarize_results.py --latex                  # 输出 LaTeX 表格
    python tools/summarize_results.py --output_dir "../outputs" # 指定目录
"""

import argparse
import json
import glob
import os


def load_experiment_summary(exp_dir):
    """加载实验目录中的 summary JSON"""
    json_path = os.path.join(exp_dir, 'experiment_summary.json')
    csv_path = os.path.join(exp_dir, 'training_log.csv')
    if not os.path.exists(json_path):
        return None
    
    with open(json_path) as f:
        data = json.load(f)
    
    # 读取 CSV 获取每 epoch 的详细数据（如果有）
    first_loss = last_loss = first_acc = last_acc = None
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
            if len(lines) > 1:
                headers = lines[0].strip().split(',')
                # 查找 train_loss 和 train_acc 的列索引
                try:
                    loss_idx = headers.index('train_loss')
                    acc_idx = headers.index('train_acc')
                    first_loss = lines[1].strip().split(',')[loss_idx]
                    last_loss = lines[-1].strip().split(',')[loss_idx]
                    first_acc = lines[1].strip().split(',')[acc_idx]
                    last_acc = lines[-1].strip().split(',')[acc_idx]
                except (ValueError, IndexError):
                    pass
    
    return {
        'name': os.path.basename(exp_dir),
        'final_mAP': data.get('final_metrics', {}).get('mAP', None),
        'final_rank1': data.get('final_metrics', {}).get('rank1', None),
        'best_mAP': data.get('best_metrics', {}).get('best_mAP', None),
        'best_mAP_epoch': data.get('best_metrics', {}).get('best_mAP_epoch', None),
        'best_rank1': data.get('best_metrics', {}).get('best_rank1', None),
        'best_rank1_epoch': data.get('best_metrics', {}).get('best_rank1_epoch', None),
        'first_loss': first_loss,
        'last_loss': last_loss,
        'first_acc': first_acc,
        'last_acc': last_acc,
    }


def print_summary_table(results, latex=False):
    """打印汇总表格"""
    if not results:
        print("未找到实验结果。请先运行训练。")
        return
    
    # 按实验名称排序
    results.sort(key=lambda r: r['name'])
    
    if latex:
        # LaTeX 表格格式
        print("% ----- LaTeX 表格 -----")
        print("\\begin{table}[htbp]")
        print("\\centering")
        print("\\caption{不同注意力机制的ReID性能对比}")
        print("\\label{tab:attention_comparison}")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("实验配置 & Best mAP & Best Rank-1 & 最终 mAP & 最终 Rank-1 \\\\")
        print("\\midrule")
        for r in results:
            b_mAP = f"{r['best_mAP']:.1%}" if r['best_mAP'] is not None else '-'
            b_r1 = f"{r['best_rank1']:.1%}" if r['best_rank1'] is not None else '-'
            f_mAP = f"{r['final_mAP']:.1%}" if r['final_mAP'] is not None else '-'
            f_r1 = f"{r['final_rank1']:.1%}" if r['final_rank1'] is not None else '-'
            print(f"{r['name']:35s} & {b_mAP} & {b_r1} & {f_mAP} & {f_r1} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
    else:
        # 终端表格
        print("=" * 100)
        print(f"{'实验配置':35s} {'Best mAP':>12s} {'Best R1':>12s} {'Final mAP':>12s} {'Final R1':>12s} {'Loss↓':>10s}")
        print("=" * 100)
        for r in results:
            b_mAP = f"{r['best_mAP']:.2%}" if r['best_mAP'] is not None else '  -'
            b_r1 = f"{r['best_rank1']:.2%}" if r['best_rank1'] is not None else '  -'
            f_mAP = f"{r['final_mAP']:.2%}" if r['final_mAP'] is not None else '  -'
            f_r1 = f"{r['final_rank1']:.2%}" if r['final_rank1'] is not None else '  -'
            
            # 计算 loss 下降幅度
            loss_info = ''
            if r['first_loss'] and r['last_loss']:
                try:
                    fl, ll = float(r['first_loss']), float(r['last_loss'])
                    loss_info = f"{fl:.2f}→{ll:.2f}"
                except:
                    loss_info = '-'
            
            print(f"{r['name']:35s} {b_mAP:>12s} {b_r1:>12s} {f_mAP:>12s} {f_r1:>12s} {loss_info:>10s}")
        print("=" * 100)
        print()
        
        # 高亮最佳结果
        best_map = max(results, key=lambda r: r.get('best_mAP') or 0)
        best_r1 = max(results, key=lambda r: r.get('best_rank1') or 0)
        print(f"🏆 最佳 mAP:    {best_map['name']} = {best_map['best_mAP']:.2%} @ epoch {best_map['best_mAP_epoch']}")
        print(f"🏆 最佳 Rank-1: {best_r1['name']} = {best_r1['best_rank1']:.2%} @ epoch {best_r1['best_rank1_epoch']}")


def main():
    parser = argparse.ArgumentParser(description="对比实验结果汇总")
    parser.add_argument(
        "--output_dir", default="../outputs", help="实验输出根目录"
    )
    parser.add_argument(
        "--latex", action="store_true", help="输出 LaTeX 表格格式"
    )
    args = parser.parse_args()
    
    base_dir = args.output_dir
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', base_dir)
    
    # 扫描所有实验目录
    results = []
    for exp_dir in sorted(glob.glob(os.path.join(base_dir, '*'))):
        if not os.path.isdir(exp_dir):
            continue
        summary = load_experiment_summary(exp_dir)
        if summary:
            results.append(summary)
    
    if not results:
        print(f"在 {base_dir} 下未找到实验数据。")
        print("请先运行训练： python tools/train.py ...")
        return
    
    print(f"找到 {len(results)} 组实验数据：\n")
    print_summary_table(results, latex=args.latex)


if __name__ == "__main__":
    main()
