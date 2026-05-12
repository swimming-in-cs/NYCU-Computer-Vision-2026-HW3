"""
inference.py — Detectron2 Mask R-CNN R-50 inference for HW3 Cell Instance Segmentation.

Usage
-----
python inference.py --checkpoint output_r50_v2/model_0003499.pth \\
                    --data-root hw3_data \\
                    --output submission.zip
"""

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from pycocotools import mask as mask_utils

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# ============================================================================
# Config — keep in sync with train.py
# ============================================================================

class Config:
    num_classes  = 4
    score_thresh = 0.05
    nms_thresh   = 0.5
    anchor_sizes   = [[32], [64], [128], [256], [512]]
    anchor_ratios  = [[0.5, 1.0, 2.0]] * 5
    min_size_test  = 800


# ============================================================================
# Utilities
# ============================================================================

def normalise_img(img: np.ndarray) -> np.ndarray:
    """Convert any .tif to uint8 RGB."""
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode binary mask to COCO RLE format (JSON-serialisable)."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def build_predictor(checkpoint: str) -> DefaultPredictor:
    """Build Detectron2 predictor from checkpoint."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = Config.num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = Config.score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = Config.nms_thresh
    cfg.MODEL.ANCHOR_GENERATOR.SIZES      = Config.anchor_sizes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = Config.anchor_ratios
    cfg.MODEL.WEIGHTS                     = checkpoint
    cfg.INPUT.MIN_SIZE_TEST               = Config.min_size_test

    return DefaultPredictor(cfg)


# ============================================================================
# Inference
# ============================================================================

def run_inference(predictor, test_dicts: list) -> list:
    """Run inference on all test images and return COCO-format results."""
    results = []

    for i, record in enumerate(test_dicts):
        img = normalise_img(tifffile.imread(record['file_name']))
        outputs   = predictor(img)
        instances = outputs['instances'].to('cpu')

        for j in range(len(instances)):
            results.append({
                'image_id':     record['image_id'],
                'category_id':  int(instances.pred_classes[j].item()) + 1,
                'segmentation': binary_mask_to_rle(
                    instances.pred_masks[j].numpy()),
                'score':        float(instances.scores[j].item()),
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_dicts)} done...")

    return results


# ============================================================================
# Argument parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='HW3 Cell Instance Segmentation — Inference')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data-root',  default='hw3_data',
                        help='Path to hw3_data folder')
    parser.add_argument('--output',     default='submission.zip',
                        help='Output zip file path (default: submission.zip)')
    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # ── Load test metadata ────────────────────────────────────────────────────
    meta_path = Path(args.data_root) / 'test_image_name_to_ids.json'
    with open(meta_path) as f:
        meta_list = json.load(f)

    test_dir = Path(args.data_root) / 'test_release'
    test_dicts = []
    for item in meta_list:
        tif = test_dir / item['file_name']
        if tif.exists():
            test_dicts.append({
                'file_name': str(tif),
                'image_id':  item['id'],
            })

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test images: {len(test_dicts)}")

    # ── Build predictor ───────────────────────────────────────────────────────
    predictor = build_predictor(args.checkpoint)

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\nRunning inference...")
    results = run_inference(predictor, test_dicts)
    print(f"\nTotal predictions: {len(results)}")

    # ── Save results ──────────────────────────────────────────────────────────
    result_json = args.output.replace('.zip', '.json')
    with open(result_json, 'w') as f:
        json.dump(results, f)

    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_json, arcname='test-results.json')

    os.remove(result_json)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
