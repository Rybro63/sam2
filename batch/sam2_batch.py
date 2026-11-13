#!/usr/bin/env python3
"""
SAM2 batch segmentation for large microscope slides (.czi) and normal images.

For each image in the input folder:
  1. Read a small overview of the whole slide.
  2. Find the tiles that contain root tissue. Skip empty background and
     black (unscanned) areas.
  3. Read those tiles at full detail and run SAM2 automatic mask generation.
  4. Save, for each tile: a label map, an overlay picture and a table of objects.
     Also save an overview picture showing which tiles were used, and one
     summary table for the whole run.

Example:
  python sam2_batch.py --input /data/input --output /data/output --max-tiles 25

Notes:
  - Tiles do not overlap, so an object that crosses a tile edge is cut in two.
  - For .czi files this reads the first channel / z-plane / time point, which is
    the color image for brightfield slides like the ones tested so far.
  - If a run stops partway, run the same command again: tiles that already
    have results are skipped.
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

MODELS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
IMAGE_EXTS = {".czi", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}

STATS_FIELDS = [
    "image", "width", "height", "tissue_tiles", "processed_tiles",
    "sam2_seconds", "seconds_per_tile", "est_seconds_all_tissue_tiles",
]

SUMMARY_FIELDS = [
    "image", "tile_x", "tile_y", "object_id", "area_px",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "predicted_iou", "stability_score",
]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- image helpers

def to_raw(arr):
    """Squeeze to (h, w) or (h, w, 3) and flip CZI's BGR order to RGB."""
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, 2::-1]  # BGRA -> RGB
    return arr


def brightness_range(raw):
    """1st and 99th percentile of the scanned (non-black) pixels."""
    f = raw.astype(np.float32)
    gray = f.mean(axis=2) if f.ndim == 3 else f
    scanned = gray > 0
    vals = gray[scanned] if
