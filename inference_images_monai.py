"""
Image inference for MONAI uterus/niche segmentation models on Supervisely exports.

Mirrors inference_video_monai.py, but loads annotated frames from a Supervisely
project root that contains ``ann/`` JSON files and matching ``video/`` files.
Each annotated frame is preprocessed with the same MONAI transforms used at
validation time, then ground-truth and predicted boundaries are overlaid.

MONAI-specific model outputs are handled the same way as video inference:

- BasicUNetPlusPlus returns a list of tensors; the first head is used
- BoundaryWeightedTverskyLoss models emit 2 channels; channel 0 is the mask,
  channel 1 is the unused boundary-distance map

Usage:
    python inference_images_monai.py <model_path> <supervisely_path> [output_dir]
"""

import json
import os.path
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    ScaleIntensityRanged,
)
from skimage import measure
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model input size (from Uterus_Segmentation_Private_Exp_niches_MONAI.py)
height, width = 512, 768

NICHE_CLASS_NAMES = frozenset({"niche"})
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")

GT_COLOR_BGR = (0, 200, 0)  # green
PRED_COLOR_BGR = (0, 0, 255)  # red
CONTOUR_THICKNESS = 3

INFERENCE_TRANSFORM = Compose(
    [
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        Resized(keys=["image"], spatial_size=(height, width), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], data_type="tensor"),
    ]
)


def preprocess_frame(frame):
    """Preprocess a single RGB frame with the MONAI validation pipeline."""
    original_shape = frame.shape[:2]
    data = INFERENCE_TRANSFORM({"image": frame})
    img_tensor = data["image"].float()
    return img_tensor, original_shape


def _primary_model_output(output):
    """Return the main segmentation logits (BasicUNetPlusPlus wraps output in a list)."""
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def _segmentation_logits(output):
    """Keep the binary mask channel; drop an optional boundary-distance head."""
    logits = _primary_model_output(output)
    if logits.ndim < 3:
        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
    # Channel dim is 1 for batched (B, C, H, W) and 0 for unbatched (C, H, W).
    channel_dim = 1 if logits.ndim == 4 else 0
    if logits.shape[channel_dim] > 1:
        logits = logits.narrow(channel_dim, 0, 1)
    return logits


def postprocess_mask(mask_logits, original_shape):
    """Convert model output to a binary mask resized to the original frame size."""
    mask_logits = _segmentation_logits(mask_logits)

    # Sigmoid > 0.5 is equivalent to the training validation threshold (logits > 0).
    mask_binary = (torch.sigmoid(mask_logits) > 0.5).cpu().numpy()

    if mask_binary.ndim == 4:
        mask_binary = mask_binary[0, 0]
    elif mask_binary.ndim == 3:
        mask_binary = mask_binary[0]

    mask_resized = cv2.resize(
        mask_binary.astype(np.uint8) * 255,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return keep_largest_region(mask_resized)


def keep_largest_region(mask):
    """Keep only the largest connected region in the mask using skimage regionprops."""
    mask_binary = (mask > 127).astype(np.uint8)
    labeled_mask = measure.label(mask_binary, connectivity=2)
    regions = measure.regionprops(labeled_mask)

    if len(regions) == 0:
        return np.zeros_like(mask)

    largest_region = max(regions, key=lambda r: r.area)
    return (labeled_mask == largest_region.label).astype(np.uint8) * 255


def _is_niche_class(class_title):
    title = (class_title or "").lower()
    return title in NICHE_CLASS_NAMES or any(name in title for name in NICHE_CLASS_NAMES)


def _video_name_from_ann_path(ann_path):
    name = os.path.basename(ann_path)
    return name[:-5] if name.endswith(".json") else name


def _find_supervisely_ann_files(data_path):
    ann_files = []
    for root, _, files in os.walk(data_path):
        if os.path.basename(root) != "ann":
            continue
        for fname in files:
            if fname.endswith(".json"):
                ann_files.append(os.path.join(root, fname))
    return sorted(ann_files)


def _build_video_index(search_roots):
    video_index = {}
    for search_root in search_roots:
        if not search_root or not os.path.isdir(search_root):
            continue
        for dirpath, _, filenames in os.walk(search_root):
            for fname in filenames:
                if fname.lower().endswith(VIDEO_EXTENSIONS):
                    video_index.setdefault(fname, os.path.join(dirpath, fname))
    return video_index


def _resolve_video_path(video_name, video_index, ann_path):
    if video_name in video_index:
        return video_index[video_name]

    ann_dir = os.path.dirname(ann_path)
    video_dir = os.path.join(os.path.dirname(ann_dir), "video")
    candidate = os.path.join(video_dir, video_name)
    if os.path.isfile(candidate):
        return candidate

    stem = os.path.splitext(video_name)[0]
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(video_dir, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
        indexed = video_index.get(f"{stem}{ext}")
        if indexed:
            return indexed
    return None


def _points_to_contour(points):
    if not points:
        return None
    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    if contour.shape[0] < 3:
        return None
    return contour


def _supervisely_polygon_to_mask(exterior, interior, mask_height, mask_width):
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    exterior_contour = _points_to_contour(exterior)
    if exterior_contour is None:
        return mask

    cv2.fillPoly(mask, [exterior_contour], 255)
    for hole in interior or []:
        hole_contour = _points_to_contour(hole)
        if hole_contour is not None:
            cv2.fillPoly(mask, [hole_contour], 0)
    return mask


def _get_niche_object_keys(annotation):
    return {
        obj["key"]
        for obj in annotation.get("objects", [])
        if _is_niche_class(obj.get("classTitle"))
    }


def _scale_points(points, scale_x, scale_y):
    scaled = []
    for x, y in points:
        scaled.append([int(round(x * scale_x)), int(round(y * scale_y))])
    return scaled


def _create_mask_from_supervisely_frame(frame, niche_keys, ann_width, ann_height, mask_width, mask_height):
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    scale_x = mask_width / ann_width if ann_width else 1.0
    scale_y = mask_height / ann_height if ann_height else 1.0

    for figure in frame.get("figures", []):
        if figure.get("geometryType") != "polygon":
            continue
        if figure.get("objectKey") not in niche_keys:
            continue
        points = figure.get("geometry", {}).get("points", {})
        exterior = _scale_points(points.get("exterior", []), scale_x, scale_y)
        interior = [_scale_points(hole, scale_x, scale_y) for hole in points.get("interior", [])]
        polygon_mask = _supervisely_polygon_to_mask(exterior, interior, mask_height, mask_width)
        mask = np.maximum(mask, polygon_mask)
    return mask


def _probe_video(video_path):
    """Return (width, height) for a readable video, or None if missing/corrupt."""
    if not video_path or not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if vw > 0 and vh > 0:
        cap.release()
        return vw, vh

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None

    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return None
    return w, h


def collect_annotated_samples(data_path):
    """Collect Supervisely annotated frames with non-empty niche masks."""
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    ann_files = _find_supervisely_ann_files(data_path)
    if not ann_files:
        raise FileNotFoundError(f"No Supervisely ann/*.json files found under {data_path}")

    video_index = _build_video_index([data_path])
    samples = []
    missing_videos = 0
    unreadable_videos = 0
    skipped_empty = 0

    for ann_path in tqdm(ann_files, desc="Loading annotations"):
        video_name = _video_name_from_ann_path(ann_path)
        video_path = _resolve_video_path(video_name, video_index, ann_path)
        if not video_path:
            missing_videos += 1
            continue

        with open(ann_path, encoding="utf-8") as f:
            annotation = json.load(f)

        niche_keys = _get_niche_object_keys(annotation)
        frames = annotation.get("frames", [])
        if not niche_keys or not frames:
            skipped_empty += 1
            continue

        ann_width = int(annotation["size"]["width"])
        ann_height = int(annotation["size"]["height"])

        video_size = _probe_video(video_path)
        if video_size is None:
            unreadable_videos += 1
            continue

        vw, vh = video_size
        volume_id = os.path.splitext(os.path.basename(video_name))[0]
        for frame_data in frames:
            frame_idx = int(frame_data["index"])
            mask = _create_mask_from_supervisely_frame(
                frame_data,
                niche_keys,
                ann_width,
                ann_height,
                vw,
                vh,
            )
            if not mask.any():
                continue
            samples.append(
                {
                    "video_path": video_path,
                    "frame_idx": frame_idx,
                    "gt_mask": mask,
                    "volume_id": volume_id,
                }
            )

    print(f"Found {len(ann_files)} annotation file(s) under: {data_path}")
    print(f"Collected {len(samples)} annotated frame(s)")
    if missing_videos:
        print(f"Skipped {missing_videos} annotation file(s) with no matching video")
    if unreadable_videos:
        print(f"Skipped {unreadable_videos} annotation file(s) with unreadable videos")
    if skipped_empty:
        print(f"Skipped {skipped_empty} annotation file(s) without Niche labels")

    return samples


def read_video_frame(cap, video_path, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def draw_overlay(frame_bgr, gt_mask, pred_mask):
    """Draw ground-truth and predicted boundaries on the original BGR frame."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    for mask_arr, color in (
        (gt_mask, GT_COLOR_BGR),
        (pred_mask, PRED_COLOR_BGR),
    ):
        m = (np.asarray(mask_arr) > 0).astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, CONTOUR_THICKNESS)

    cv2.putText(
        canvas, "Ground Truth", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_COLOR_BGR, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "Prediction", (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, PRED_COLOR_BGR, 2, cv2.LINE_AA,
    )
    return canvas


def binary_dice(gt_mask, pred_mask):
    gt = np.asarray(gt_mask) > 0
    pred = np.asarray(pred_mask) > 0
    intersection = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def binary_iou(gt_mask, pred_mask):
    gt = np.asarray(gt_mask) > 0
    pred = np.asarray(pred_mask) > 0
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def process_samples(model, samples, output_dir, model_name):
    """Run inference on annotated frames and write overlay images."""
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["video_path"]].append(sample)

    model.eval()
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    dice_scores = []
    iou_scores = []
    saved = 0
    skipped_unreadable = 0

    with torch.no_grad():
        for video_path, video_samples in tqdm(grouped.items(), desc="Processing videos"):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Warning: Could not open video: {video_path}")
                skipped_unreadable += len(video_samples)
                continue

            try:
                video_samples = sorted(video_samples, key=lambda s: s["frame_idx"])
                for sample in video_samples:
                    frame_idx = sample["frame_idx"]
                    try:
                        frame_bgr = read_video_frame(cap, video_path, frame_idx)
                    except ValueError as exc:
                        print(f"Warning: {exc}")
                        skipped_unreadable += 1
                        continue

                    preprocess_start = time.time()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    img_tensor, original_shape = preprocess_frame(frame_rgb)
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    preprocessing_times.append(time.time() - preprocess_start)

                    inference_start = time.time()
                    output = model(img_tensor)
                    inference_times.append(time.time() - inference_start)

                    postprocess_start = time.time()
                    pred_mask = postprocess_mask(output, original_shape)
                    gt_mask = sample["gt_mask"]
                    if gt_mask.shape[:2] != original_shape:
                        gt_mask = cv2.resize(
                            gt_mask,
                            (original_shape[1], original_shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    overlay = draw_overlay(frame_bgr, gt_mask, pred_mask)
                    postprocessing_times.append(time.time() - postprocess_start)

                    dice_scores.append(binary_dice(gt_mask, pred_mask))
                    iou_scores.append(binary_iou(gt_mask, pred_mask))

                    output_filename = f"{sample['volume_id']}_frame{frame_idx:05d}_{model_name}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, overlay)
                    saved += 1
            finally:
                cap.release()

    if skipped_unreadable:
        print(f"Skipped {skipped_unreadable} frame(s) that could not be read")
    print(f"Saved {saved} overlay image(s) to: {output_dir}")

    return {
        "preprocessing_times": preprocessing_times,
        "inference_times": inference_times,
        "postprocessing_times": postprocessing_times,
        "dice_scores": dice_scores,
        "iou_scores": iou_scores,
        "frame_count": saved,
    }


def extract_model_name(model_path):
    """Extract model name from model path."""
    return Path(model_path).stem


def print_statistics(stats):
    """Print timing and overlap statistics over all annotated frames."""
    all_preprocessing_times = stats["preprocessing_times"]
    all_inference_times = stats["inference_times"]
    all_postprocessing_times = stats["postprocessing_times"]
    total_frames = stats["frame_count"]

    print("\n" + "=" * 80)
    print("TIMING STATISTICS")
    print("=" * 80)
    print(f"Total annotated frames processed: {total_frames}")
    print("\nPer Frame Statistics:")
    print("-" * 80)

    if all_preprocessing_times:
        print("Preprocessing:")
        print(f"  Mean: {np.mean(all_preprocessing_times) * 1000:.3f} ms")
        print(f"  Std:  {np.std(all_preprocessing_times) * 1000:.3f} ms")
        print(f"  Min:  {np.min(all_preprocessing_times) * 1000:.3f} ms")
        print(f"  Max:  {np.max(all_preprocessing_times) * 1000:.3f} ms")
        print(f"  Total: {np.sum(all_preprocessing_times):.3f} s")

    if all_inference_times:
        print("\nModel Inference:")
        print(f"  Mean: {np.mean(all_inference_times) * 1000:.3f} ms")
        print(f"  Std:  {np.std(all_inference_times) * 1000:.3f} ms")
        print(f"  Min:  {np.min(all_inference_times) * 1000:.3f} ms")
        print(f"  Max:  {np.max(all_inference_times) * 1000:.3f} ms")
        print(f"  Total: {np.sum(all_inference_times):.3f} s")

    if all_postprocessing_times:
        print("\nPostprocessing:")
        print(f"  Mean: {np.mean(all_postprocessing_times) * 1000:.3f} ms")
        print(f"  Std:  {np.std(all_postprocessing_times) * 1000:.3f} ms")
        print(f"  Min:  {np.min(all_postprocessing_times) * 1000:.3f} ms")
        print(f"  Max:  {np.max(all_postprocessing_times) * 1000:.3f} ms")
        print(f"  Total: {np.sum(all_postprocessing_times):.3f} s")

    if all_preprocessing_times and all_inference_times and all_postprocessing_times:
        total_times = [
            p + i + post
            for p, i, post in zip(all_preprocessing_times, all_inference_times, all_postprocessing_times)
        ]
        print("\nTotal per Frame (Preprocessing + Inference + Postprocessing):")
        print(f"  Mean: {np.mean(total_times) * 1000:.3f} ms")
        print(f"  Std:  {np.std(total_times) * 1000:.3f} ms")
        print(f"  Min:  {np.min(total_times) * 1000:.3f} ms")
        print(f"  Max:  {np.max(total_times) * 1000:.3f} ms")
        print(f"  Total: {np.sum(total_times):.3f} s")

    if stats["dice_scores"] and stats["iou_scores"]:
        print("\nOverlap vs Ground Truth:")
        print(f"  Dice mean: {np.mean(stats['dice_scores']):.4f}  std: {np.std(stats['dice_scores']):.4f}")
        print(f"  IoU mean:  {np.mean(stats['iou_scores']):.4f}  std: {np.std(stats['iou_scores']):.4f}")

    print("=" * 80)


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_images_monai.py <model_path> <supervisely_path> [output_dir]")
        print("Example: python inference_images_monai.py model_tvus.pt /path/to/TVUS_Niches")
        print("Example: python inference_images_monai.py model_tvus.pt /path/to/TVUS_Niches overlays")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "segmented_images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_name = extract_model_name(model_path)
    print(f"Model name: {model_name}")

    print(f"Loading model from: {model_path}")
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        samples = collect_annotated_samples(input_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not samples:
        print("No annotated frames found to process")
        sys.exit(1)

    stats = process_samples(model, samples, output_dir, model_name)
    print_statistics(stats)
    print("\nAll annotated images processed successfully!")


if __name__ == "__main__":
    main()
