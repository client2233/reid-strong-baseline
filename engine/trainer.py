# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import json
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP

global ITER
ITER = 0


def _get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _save_experiment_data(output_dir, epoch_data, train_data, eval_data):
    """
    保存实验报告所需的完整数据：CSV + JSON + 图表
    
    Args:
        output_dir: 输出目录
        epoch_data: 每个 epoch 的训练指标列表
        train_data: 训练过程中的迭代级指标（可选）
        eval_data: 验证评估指标列表
    """
    logger = logging.getLogger("reid_baseline.train")
    
    # ======================== 1. 保存完整 CSV ========================
    csv_path = os.path.join(output_dir, 'training_log.csv')
    
    # 收集所有 epoch_keys
    epoch_keys = set()
    for d in epoch_data:
        epoch_keys.update(d.keys())
    epoch_keys = sorted(epoch_keys)
    
    with open(csv_path, 'w') as f:
        f.write(','.join(epoch_keys) + '\n')
        for d in epoch_data:
            f.write(','.join(str(d.get(k, '')) for k in epoch_keys) + '\n')
    logger.info("Training log (CSV) saved to {}".format(csv_path))
    
    # ======================== 2. 保存 JSON 摘要 ========================
    summary = {
        'config': {
            'model_name': epoch_data[0].get('model_name', 'N/A') if epoch_data else 'N/A',
            'num_epochs': len(epoch_data),
            'eval_interval': epoch_data[0].get('eval_period', 1) if epoch_data else 1,
        },
        'final_metrics': {},
        'best_metrics': {},
        'training_summary': {},
    }
    
    if eval_data:
        # 最终指标
        last = eval_data[-1]
        summary['final_metrics'] = {
            'mAP': float(last.get('mAP', 0)),
            'rank1': float(last.get('rank1', 0)),
            'rank5': float(last.get('rank5', 0)),
            'rank10': float(last.get('rank10', 0)),
        }
        
        # 最佳指标
        best_mAP = max(eval_data, key=lambda x: x.get('mAP', 0))
        best_rank1 = max(eval_data, key=lambda x: x.get('rank1', 0))
        summary['best_metrics'] = {
            'best_mAP': float(best_mAP.get('mAP', 0)),
            'best_mAP_epoch': int(best_mAP.get('epoch', 0)),
            'best_rank1': float(best_rank1.get('rank1', 0)),
            'best_rank1_epoch': int(best_rank1.get('epoch', 0)),
        }
    
    if epoch_data:
        train_losses = [d.get('train_loss', None) for d in epoch_data if d.get('train_loss') is not None]
        train_accs = [d.get('train_acc', None) for d in epoch_data if d.get('train_acc') is not None]
        summary['training_summary'] = {
            'initial_loss': float(train_losses[0]) if train_losses else None,
            'final_loss': float(train_losses[-1]) if train_losses else None,
            'min_loss': float(min(train_losses)) if train_losses else None,
            'initial_acc': float(train_accs[0]) if train_accs else None,
            'final_acc': float(train_accs[-1]) if train_accs else None,
            'max_acc': float(max(train_accs)) if train_accs else None,
        }
    
    json_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Experiment summary (JSON) saved to {}".format(json_path))
    
    # ======================== 3. 绘制综合图表 ========================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 确定需要绘制的子图数量
        has_train = any(d.get('train_loss') is not None for d in epoch_data)
        has_eval = len(eval_data) > 0
        
        n_plots = 1  # 至少绘制 loss
        if has_train:
            n_plots += 1  # train acc
        if has_eval:
            n_plots += 2  # mAP + Rank-1
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        epochs_all = [d['epoch'] for d in epoch_data]
        
        # ---- 图1: Training Loss ----
        ax = axes[plot_idx]
        if has_train:
            train_losses = [d.get('train_loss') for d in epoch_data]
            ax.plot(epochs_all, train_losses, 'b-', linewidth=1.5, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
        
        # ---- 图2: Training Accuracy ----
        if has_train:
            ax = axes[plot_idx]
            train_accs = [d.get('train_acc') for d in epoch_data]
            ax.plot(epochs_all, [acc * 100 for acc in train_accs], 
                    'g-', linewidth=1.5, label='Train Acc')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Training Accuracy Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plot_idx += 1
        
        # ---- 图3: mAP Curve ----
        if has_eval:
            ax = axes[plot_idx]
            eval_epochs = [r['epoch'] for r in eval_data]
            mAPs = [r['mAP'] * 100 for r in eval_data]
            ax.plot(eval_epochs, mAPs, 'b-o', linewidth=1.5, label='mAP', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP (%)')
            ax.set_title('mAP Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 标注最佳值
            best_idx = mAPs.index(max(mAPs))
            ax.annotate(f'Best: {max(mAPs):.1f}% @ E{eval_epochs[best_idx]}',
                        xy=(eval_epochs[best_idx], mAPs[best_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='darkred'))
            plot_idx += 1
        
        # ---- 图4: Rank-1 Curve ----
        if has_eval:
            ax = axes[plot_idx]
            rank1s = [r['rank1'] * 100 for r in eval_data]
            ax.plot(eval_epochs, rank1s, 'r-s', linewidth=1.5, label='Rank-1', markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Rank-1 Accuracy (%)')
            ax.set_title('Rank-1 Curve')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 标注最佳值
            best_idx = rank1s.index(max(rank1s))
            ax.annotate(f'Best: {max(rank1s):.1f}% @ E{eval_epochs[best_idx]}',
                        xy=(eval_epochs[best_idx], rank1s[best_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='darkred'))
            plot_idx += 1
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Training curves plot saved to {}".format(save_path))
        
        # ---- 额外图: 学习率曲线 (单独文件) ----
        if 'lr' in epoch_data[0]:
            fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
            lrs = [d['lr'] for d in epoch_data]
            ax2.plot(epochs_all, lrs, 'purple', linewidth=1.5)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            lr_path = os.path.join(output_dir, 'lr_schedule.png')
            plt.savefig(lr_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("LR schedule saved to {}".format(lr_path))
            
    except ImportError:
        logger.warning("matplotlib not installed, skip plotting training curves")
    except Exception as e:
        logger.warning("Failed to plot training curves: {}".format(e))

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        if engine.state.epoch % checkpoint_period == 0:
            checkpointer(engine, {'model': model,
                                  'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    # ---- 实验数据跟踪 ----
    epoch_data = []  # 每 epoch 的训练指标
    eval_results = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def collect_epoch_metrics(engine):
        """收集每个 epoch 的训练指标"""
        record = {
            'epoch': engine.state.epoch,
            'train_loss': engine.state.metrics['avg_loss'],
            'train_acc': engine.state.metrics['avg_acc'],
            'lr': _get_lr(optimizer),
            'model_name': cfg.MODEL.NAME,
            'eval_period': eval_period,
        }
        epoch_data.append(record)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            eval_results.append({
                'epoch': engine.state.epoch,
                'mAP': mAP,
                'rank1': cmc[0],
                'rank5': cmc[4],
                'rank10': cmc[9],
            })

    @trainer.on(Events.COMPLETED)
    def save_experiment_data(engine):
        """训练结束时保存所有实验数据"""
        logger.info("Saving experiment data for report...")
        _save_experiment_data(output_dir, epoch_data, None, eval_results)
        logger.info("All experiment data saved to {}".format(output_dir))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        if engine.state.epoch % checkpoint_period == 0:
            checkpointer(engine, {'model': model,
                                  'optimizer': optimizer,
                                  'center_param': center_criterion,
                                  'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    # ---- 实验数据跟踪 ----
    epoch_data = []
    eval_results = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def collect_epoch_metrics(engine):
        """收集每个 epoch 的训练指标"""
        record = {
            'epoch': engine.state.epoch,
            'train_loss': engine.state.metrics['avg_loss'],
            'train_acc': engine.state.metrics['avg_acc'],
            'lr': _get_lr(optimizer),
            'model_name': cfg.MODEL.NAME,
            'eval_period': eval_period,
        }
        epoch_data.append(record)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            eval_results.append({
                'epoch': engine.state.epoch,
                'mAP': mAP,
                'rank1': cmc[0],
                'rank5': cmc[4],
                'rank10': cmc[9],
            })

    @trainer.on(Events.COMPLETED)
    def save_experiment_data(engine):
        """训练结束时保存所有实验数据"""
        logger.info("Saving experiment data for report...")
        _save_experiment_data(output_dir, epoch_data, None, eval_results)
        logger.info("All experiment data saved to {}".format(output_dir))

    trainer.run(train_loader, max_epochs=epochs)
