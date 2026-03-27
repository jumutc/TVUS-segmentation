"""
Evaluate a trained segmentation model on the validation split.

Reproduces the exact same GroupShuffleSplit used during training so the
validation set is identical.  For each validation sample the script:
  - runs inference and computes IoU / NSD / Dice,
  - saves an overlay image with ground-truth and predicted contours in
    different colours on the original frame.
"""

import argparse
import json
import os
import ssl

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from MeshMetrics import DistanceMetrics
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GT_COLOR_BGR = (0, 200, 0)       # green
PRED_COLOR_BGR = (0, 0, 255)     # red
CONTOUR_THICKNESS = 2

# ---------------------------------------------------------------------------
# Data helpers (kept in sync with the training script)
# ---------------------------------------------------------------------------

def _is_control_folder(folder_name):
    return 'control' in folder_name.lower()


def _collect_video_paths(videos_root, include_control=False):
    video_by_name = {}
    for folder in os.listdir(videos_root):
        folder_path = os.path.join(videos_root, folder)
        if not os.path.isdir(folder_path):
            continue
        if _is_control_folder(folder) and not include_control:
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.mp4', '.avi', '.mov')):
                video_by_name[fname] = os.path.join(folder_path, fname)
    return video_by_name


def _polygon_to_mask(polygon_data, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon_data:
        return mask
    pts = []
    if isinstance(polygon_data, dict):
        keys = sorted(polygon_data.keys(), key=lambda k: int(k) if k.isdigit() else k)
        for k in keys:
            pt = polygon_data[k]
            pts.append([int(pt['x'] * width), int(pt['y'] * height)])
    elif isinstance(polygon_data, list) and len(polygon_data) > 0:
        flat = polygon_data[0][0] if isinstance(polygon_data[0][0], (list, tuple)) else polygon_data[0]
        for i in range(0, len(flat), 2):
            pts.append([int(flat[i] * width), int(flat[i + 1] * height)])
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def _create_mask_from_encord_objects(objects, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        if obj.get('shape') != 'polygon':
            continue
        poly = obj.get('polygon')
        if isinstance(poly, dict):
            mask = np.maximum(mask, _polygon_to_mask(poly, height, width))
        elif obj.get('polygons'):
            for poly_contour in obj['polygons']:
                flat = poly_contour[0] if poly_contour and isinstance(poly_contour[0], (list, tuple)) else poly_contour
                if not isinstance(flat, (list, tuple)) or len(flat) < 6:
                    continue
                pts = []
                for i in range(0, len(flat), 2):
                    pts.append([int(flat[i] * width), int(flat[i + 1] * height)])
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def create_df(data_path, control_balance_ratio=0.3):
    metadata_path = os.path.join(data_path, 'metadata_labels.json')
    labels_dir = os.path.join(data_path, 'labels')
    videos_dir = os.path.join(data_path, 'videos')

    for p, desc in [(metadata_path, 'Metadata'), (labels_dir, 'Labels dir'), (videos_dir, 'Videos dir')]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{desc} not found: {p}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    label_to_video = {m['label_hash']: m['title'] for m in metadata if m.get('title')}
    labeled_videos = _collect_video_paths(videos_dir, include_control=False)
    control_videos = {}
    for folder in os.listdir(videos_dir):
        folder_path = os.path.join(videos_dir, folder)
        if not os.path.isdir(folder_path) or not _is_control_folder(folder):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.mp4', '.avi', '.mov')):
                control_videos[fname] = os.path.join(folder_path, fname)

    rows = []
    for label_hash, video_filename in tqdm(label_to_video.items(), desc='Loading labeled data'):
        video_path = labeled_videos.get(video_filename)
        if not video_path or not os.path.isfile(video_path):
            continue
        label_path = os.path.join(labels_dir, f'{label_hash}.json')
        if not os.path.isfile(label_path):
            continue
        with open(label_path) as f:
            label_data = json.load(f)
        if not isinstance(label_data, dict):
            continue

        cap = cv2.VideoCapture(video_path)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        volume_id = os.path.splitext(video_filename)[0]
        for frame_key, frame_data in label_data.items():
            if not frame_key.isdigit():
                continue
            objs = frame_data.get('objects', [])
            if not objs:
                continue
            mask = _create_mask_from_encord_objects(objs, vh, vw)
            rows.append({
                'video_path': video_path,
                'frame_idx': int(frame_key),
                'seg': mask,
                'volume_id': volume_id,
            })

    n_labeled = len(rows)
    n_control_target = max(0, int(n_labeled * control_balance_ratio))

    if n_control_target > 0 and control_videos:
        control_paths = list(control_videos.values())
        added = 0
        for video_path in control_paths:
            if added >= n_control_target:
                break
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            if n_frames <= 0:
                continue
            volume_id = os.path.splitext(os.path.basename(video_path))[0]
            indices = np.linspace(0, n_frames - 1,
                                  min(n_control_target - added, max(1, n_frames // 5)),
                                  dtype=int)
            for fi in indices:
                if added >= n_control_target:
                    break
                rows.append({
                    'video_path': video_path,
                    'frame_idx': int(fi),
                    'seg': np.zeros((vh, vw), dtype=np.uint8),
                    'volume_id': volume_id,
                })
                added += 1

    return pd.DataFrame(rows, index=np.arange(len(rows)))


def _read_frame_from_video(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Dataset (validation only – no augmentation beyond resize)
# ---------------------------------------------------------------------------

class TVUSDataset(Dataset):
    def __init__(self, X, transform):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.loc[idx]
        img = _read_frame_from_video(row['video_path'], int(row['frame_idx']))
        mask = np.zeros(list(row['seg'].shape) + [1])
        mask[row['seg'] > 0, 0] = 1
        aug = self.transform(image=img, mask=mask)
        img = aug['image']
        mask = aug['mask']
        mask = torch.from_numpy(np.transpose(mask, axes=(2, 0, 1))).round().float()
        img = T.ToTensor()(img).float()
        return img, mask, idx


# ---------------------------------------------------------------------------
# Overlay visualisation
# ---------------------------------------------------------------------------

def draw_overlay(original_rgb, gt_mask, pred_mask):
    """
    Draw ground-truth and predicted contours on the original image.

    Args:
        original_rgb: (H, W, 3) uint8 RGB image
        gt_mask:   (H, W) binary mask (bool / uint8)
        pred_mask: (H, W) binary mask (bool / uint8)

    Returns:
        BGR image with coloured contours and a legend
    """
    canvas = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    for mask_arr, color, label in [
        (gt_mask, GT_COLOR_BGR, "Ground Truth (green)"),
        (pred_mask, PRED_COLOR_BGR, "Prediction (red)"),
    ]:
        m = mask_arr.astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, CONTOUR_THICKNESS)

    cv2.putText(canvas, "Ground Truth", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR_BGR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Prediction", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR_BGR, 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained segmentation model on the validation split')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the saved model (.pt)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to data directory (videos/, labels/, metadata_labels.json)')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2],
                        help='Cross-validation fold index to evaluate (default: 0)')
    parser.add_argument('--control-balance-ratio', type=float, default=0.3,
                        help='Ratio of control frames (must match training, default: 0.3)')
    parser.add_argument('--output-dir', type=str, default='eval_output',
                        help='Directory for overlay images and CSV (default: eval_output)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Path for results CSV (default: <output-dir>/metrics.csv)')
    parser.add_argument('--height', type=int, default=512,
                        help='Resize height for model input (default: 512)')
    parser.add_argument('--width', type=int, default=768,
                        help='Resize width for model input (default: 768)')
    parser.add_argument('--nsd-tolerance', type=float, default=6,
                        help='NSD tolerance in pixels (default: 6, same as training)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Binarisation threshold for predictions (default: 0.0)')
    args = parser.parse_args()

    output_dir = args.output_dir
    overlay_dir = os.path.join(output_dir, 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)
    csv_path = args.output_csv or os.path.join(output_dir, 'metrics.csv')

    # ---- build dataframe (identical to training) ----
    print(f"Building dataframe from {args.data_path} ...")
    df = create_df(args.data_path, args.control_balance_ratio)
    print(f"Total samples in dataframe: {len(df)}")

    # ---- reproduce the split ----
    splitter = GroupShuffleSplit(n_splits=3, test_size=0.15, random_state=0)
    splits = list(splitter.split(df.index, groups=df['volume_id']))
    _, val_indices = splits[args.fold]

    val_df = df.loc[val_indices].reset_index(drop=True)
    print(f"Fold {args.fold}: {len(val_df)} validation samples")

    val_volumes = val_df['volume_id'].unique()
    print(f"Validation volumes ({len(val_volumes)}): {sorted(val_volumes)}")

    # ---- load model ----
    print(f"Loading model from {args.model} ...")
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # ---- dataloader ----
    t_val = A.Compose([A.Resize(args.height, args.width, interpolation=cv2.INTER_LINEAR)],
                      is_check_shapes=False)
    val_set = TVUSDataset(val_df, t_val)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False)

    # ---- inference loop ----
    dist_metrics = DistanceMetrics()
    results = []

    print("Running inference ...")
    with torch.no_grad():
        for image, mask_tensor, idx in tqdm(val_loader):
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            elif isinstance(idx, (np.ndarray, list)):
                idx = idx[0] if len(idx) > 0 else idx

            row = val_df.loc[idx]
            video_path = row['video_path']
            frame_idx = int(row['frame_idx'])
            volume_id = row['volume_id']

            image = image.to(device)
            output = model(image).cpu()
            pred = np.squeeze(output.detach().numpy(), axis=(0, 1))

            gt_full = np.squeeze(row['seg'])
            if pred.shape != gt_full.shape:
                pred_full = cv2.resize(pred, (gt_full.shape[1], gt_full.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            else:
                pred_full = pred

            gt_bin = (gt_full > 0).astype(np.uint8)
            pred_bin = (pred_full > args.threshold).astype(np.uint8)

            gt_has_fg = gt_bin.any()
            pred_has_fg = pred_bin.any()

            if gt_has_fg or pred_has_fg:
                dist_metrics.set_input(pred_bin.astype(bool), gt_bin.astype(bool), spacing=(1, 1))
                iou = float(dist_metrics.iou())
                nsd = float(dist_metrics.nsd(args.nsd_tolerance))
                dice = float(dist_metrics.dsc())
            else:
                iou, nsd, dice = 1.0, 1.0, 1.0

            results.append({
                'volume_id': volume_id,
                'video_path': video_path,
                'frame_idx': frame_idx,
                'iou': iou,
                'nsd': nsd,
                'dice': dice,
                'gt_pixels': int(gt_bin.sum()),
                'pred_pixels': int(pred_bin.sum()),
            })

            # ---- overlay ----
            orig_rgb = _read_frame_from_video(video_path, frame_idx)
            overlay = draw_overlay(orig_rgb, gt_bin, pred_bin)
            fname = f"{volume_id}_frame{frame_idx:05d}.png"
            cv2.imwrite(os.path.join(overlay_dir, fname), overlay)

    # ---- save CSV ----
    res_df = pd.DataFrame(results)
    res_df.to_csv(csv_path, index=False)
    print(f"\nPer-sample results saved to {csv_path}")

    # ---- aggregate ----
    iou_vals = res_df['iou'].values
    nsd_vals = res_df['nsd'].values
    dice_vals = res_df['dice'].values

    print(f"\n{'=' * 60}")
    print(f"Fold {args.fold} — {len(res_df)} validation samples")
    print(f"{'=' * 60}")
    print(f"IoU:  {np.mean(iou_vals):.4f} +/- {np.std(iou_vals):.4f}")
    print(f"NSD:  {np.mean(nsd_vals):.4f} +/- {np.std(nsd_vals):.4f}")
    print(f"Dice: {np.mean(dice_vals):.4f} +/- {np.std(dice_vals):.4f}")
    print(f"{'=' * 60}")

    # ---- per-volume breakdown ----
    print("\nPer-volume breakdown:")
    for vid, grp in res_df.groupby('volume_id'):
        print(f"  {vid:30s}  n={len(grp):3d}  "
              f"IoU={grp['iou'].mean():.4f}  "
              f"NSD={grp['nsd'].mean():.4f}  "
              f"Dice={grp['dice'].mean():.4f}")

    print(f"\nOverlay images saved to {overlay_dir}/")


if __name__ == '__main__':
    main()
