"""
train.py — Detectron2 Mask R-CNN R-50 training for HW3 Cell Instance Segmentation.

Usage
-----
python train.py --data-root /path/to/hw3_data --output-dir ./output_r50_v2
python train.py --resume  # auto-resume from last checkpoint
"""

import argparse
import copy
import os
import random
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    transforms as T,
)
import detectron2.data.detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.structures import BoxMode


# ============================================================================
# Config — all tunable hyper-parameters in one place
# ============================================================================

class Config:
    # Paths
    data_root  = "hw3_data"
    output_dir = "output_r50_v2"

    # Training
    max_iter   = 10000
    batch_size = 2
    base_lr    = 5e-5        # lower LR to avoid early overfitting
    val_ratio  = 0.1
    seed       = 42

    # LR schedule
    warmup_iters     = 200
    lr_decay_steps   = (8000, 9000)
    lr_decay_gamma   = 0.1

    # Evaluation & checkpoint
    eval_period      = 500   # evaluate every N iterations
    checkpoint_period = 500

    # Model
    num_classes  = 4
    score_thresh = 0.05
    nms_thresh   = 0.5

    # Anchors
    anchor_sizes   = [[32], [64], [128], [256], [512]]
    anchor_ratios  = [[0.5, 1.0, 2.0]] * 5

    # Input
    min_size_train = (640, 672, 704, 736, 768, 800)
    max_size_train = 1333
    min_size_test  = 800

    num_workers = 2


# ============================================================================
# Dataset utilities
# ============================================================================

CELL_CLASSES = ['class1', 'class2', 'class3', 'class4']


def normalise_img(img: np.ndarray) -> np.ndarray:
    """Convert any .tif to uint8 RGB."""
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def binary_mask_to_polygon(binary_mask: np.ndarray):
    """Convert binary mask to largest contour polygon."""
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1:
        return None
    return contour.flatten().tolist()


def mask_to_annotations(mask: np.ndarray, category_id: int) -> list:
    """Convert instance mask to list of Detectron2 annotation dicts."""
    annotations = []
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        binary = (mask == inst_id).astype(np.uint8)
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if not rows.any():
            continue
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        poly = binary_mask_to_polygon(binary)
        if poly is None:
            continue
        annotations.append({
            'bbox':         [int(x_min), int(y_min), int(x_max), int(y_max)],
            'bbox_mode':    BoxMode.XYXY_ABS,
            'segmentation': [poly],
            'category_id':  category_id,
            'iscrowd':      0,
        })
    return annotations


def build_dataset_dicts(data_root: str) -> list:
    """Build Detectron2-format dataset dicts from training folder."""
    train_dir = Path(data_root) / 'train'
    dicts, image_id = [], 0

    for folder in sorted(train_dir.iterdir()):
        if not folder.is_dir():
            continue
        img_path = folder / 'image.tif'
        if not img_path.exists():
            continue

        img = normalise_img(tifffile.imread(str(img_path)))
        H, W = img.shape[:2]
        annos = []

        for cat_id, cls in enumerate(CELL_CLASSES):
            mp = folder / f'{cls}.tif'
            if mp.exists():
                annos.extend(mask_to_annotations(
                    tifffile.imread(str(mp)), cat_id))

        if not annos:
            continue

        dicts.append({
            'file_name':   str(img_path),
            'image_id':    image_id,
            'height':      H,
            'width':       W,
            'annotations': annos,
        })
        image_id += 1

    return dicts


def register_datasets(data_root: str, val_ratio: float = 0.1, seed: int = 42):
    """Register train/val splits to Detectron2 DatasetCatalog."""
    random.seed(seed)
    all_dicts = build_dataset_dicts(data_root)
    random.shuffle(all_dicts)
    n_val = max(1, int(len(all_dicts) * val_ratio))

    splits = {
        'cell_train': all_dicts[n_val:],
        'cell_val':   all_dicts[:n_val],
    }
    for name, dicts in splits.items():
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
        DatasetCatalog.register(name, lambda d=dicts: d)
        MetadataCatalog.get(name).set(thing_classes=CELL_CLASSES)

    print(f"Train: {len(splits['cell_train'])}, Val: {len(splits['cell_val'])}")


# ============================================================================
# Augmentation mapper
# ============================================================================

class CellAugMapper(DatasetMapper):
    """Custom mapper with H/V flip, rotation, color jitter, multi-scale resize."""

    def __init__(self, cfg, is_train: bool = True):
        super().__init__(cfg, is_train=is_train)
        if is_train:
            self.augmentations = T.AugmentationList([
                T.RandomFlip(prob=0.5, horizontal=True,  vertical=False),
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomRotation(angle=[-15, 15]),
                T.RandomBrightness(0.8, 1.2),
                T.RandomContrast(0.8, 1.2),
                T.ResizeShortestEdge(
                    short_edge_length=Config.min_size_train,
                    max_size=Config.max_size_train,
                    sample_style='choice',
                ),
            ])

    def __call__(self, dataset_dict: dict) -> dict:
        dataset_dict = copy.deepcopy(dataset_dict)
        img = normalise_img(tifffile.imread(dataset_dict['file_name']))

        if not self.is_train:
            dataset_dict['image'] = torch.as_tensor(
                img.transpose(2, 0, 1).astype('float32'))
            dataset_dict.pop('annotations', None)
            return dataset_dict

        aug_input = T.AugInput(img)
        transforms = self.augmentations(aug_input)
        img = aug_input.image
        dataset_dict['image'] = torch.as_tensor(
            img.transpose(2, 0, 1).astype('float32'))

        annos = [
            utils.transform_instance_annotations(a, [transforms], img.shape[:2])
            for a in dataset_dict.get('annotations', [])
            if a.get('iscrowd', 0) == 0
        ]
        dataset_dict['instances'] = utils.annotations_to_instances(
            annos, img.shape[:2], mask_format='polygon')
        return dataset_dict


# ============================================================================
# Trainer
# ============================================================================

class CellTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=CellAugMapper(cfg, is_train=True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return DatasetEvaluators([
            COCOEvaluator(dataset_name,
                          output_dir=output_folder or cfg.OUTPUT_DIR)
        ])


# ============================================================================
# Config builder
# ============================================================================

def build_cfg(args) -> object:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

    cfg.DATASETS.TRAIN = ('cell_train',)
    cfg.DATASETS.TEST  = ('cell_val',)
    cfg.MODEL.WEIGHTS  = model_zoo.get_checkpoint_url(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

    cfg.MODEL.ROI_HEADS.NUM_CLASSES          = Config.num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST    = Config.score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST      = Config.nms_thresh
    cfg.MODEL.ANCHOR_GENERATOR.SIZES         = Config.anchor_sizes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = Config.anchor_ratios

    cfg.INPUT.MIN_SIZE_TRAIN          = Config.min_size_train
    cfg.INPUT.MAX_SIZE_TRAIN          = Config.max_size_train
    cfg.INPUT.MIN_SIZE_TEST           = Config.min_size_test
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'

    cfg.SOLVER.IMS_PER_BATCH      = args.batch_size
    cfg.SOLVER.BASE_LR            = args.lr
    cfg.SOLVER.MAX_ITER           = args.max_iter
    cfg.SOLVER.STEPS              = Config.lr_decay_steps
    cfg.SOLVER.GAMMA              = Config.lr_decay_gamma
    cfg.SOLVER.WARMUP_ITERS       = Config.warmup_iters
    cfg.SOLVER.CHECKPOINT_PERIOD  = Config.checkpoint_period

    cfg.TEST.EVAL_PERIOD  = Config.eval_period
    cfg.OUTPUT_DIR        = args.output_dir
    cfg.DATALOADER.NUM_WORKERS = Config.num_workers

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


# ============================================================================
# Argument parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='HW3 Cell Instance Segmentation — Train')
    parser.add_argument('--data-root',  default=Config.data_root)
    parser.add_argument('--output-dir', default=Config.output_dir)
    parser.add_argument('--max-iter',   type=int,   default=Config.max_iter)
    parser.add_argument('--batch-size', type=int,   default=Config.batch_size)
    parser.add_argument('--lr',         type=float, default=Config.base_lr)
    parser.add_argument('--resume',     action='store_true',
                        help='Resume from last checkpoint if available')
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    print(f"Data root  : {args.data_root}")
    print(f"Output dir : {args.output_dir}")
    print(f"Max iter   : {args.max_iter}  |  LR: {args.lr}  |  Batch: {args.batch_size}")

    register_datasets(args.data_root, Config.val_ratio, Config.seed)

    cfg = build_cfg(args)

    # Auto-detect resume
    last_ckpt = os.path.join(args.output_dir, 'last_checkpoint')
    resume = args.resume and os.path.exists(last_ckpt)
    if resume:
        print(f"Resuming from last checkpoint in {args.output_dir}")
    else:
        print("Starting training from scratch.")

    trainer = CellTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    print("Training complete!")


if __name__ == '__main__':
    main()
