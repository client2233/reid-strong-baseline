# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import random

import numpy as np
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def _subset_dataset(dataset, ratio, seed=42):
    """从训练数据中随机抽取 ratio 比例的子集，同时保持身份类别均衡"""
    if ratio >= 1.0:
        return dataset

    rng = random.Random(seed)
    
    # 按身份分组
    pid_to_indices = {}
    for idx, (_, pid, _) in enumerate(dataset):
        pid_to_indices.setdefault(pid, []).append(idx)
    
    # 对每个身份按比例采样
    sampled_indices = []
    for pid, indices in pid_to_indices.items():
        n_sample = max(1, int(len(indices) * ratio))
        sampled = rng.sample(indices, min(n_sample, len(indices)))
        sampled_indices.extend(sampled)
    
    sampled_indices.sort()
    sampled_dataset = [dataset[i] for i in sampled_indices]
    
    print("  Dataset subset: {} -> {} images ({:.0f}%)".format(
        len(dataset), len(sampled_dataset), ratio * 100))
    return sampled_dataset


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    # 应用数据子集比例（用于快速原型验证）
    subset_ratio = cfg.DATALOADER.SUBSET_RATIO
    train_data = _subset_dataset(dataset.train, subset_ratio) if subset_ratio < 1.0 else dataset.train
    train_set = ImageDataset(train_data, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(train_data, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes
