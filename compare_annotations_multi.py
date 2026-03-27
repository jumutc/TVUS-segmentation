import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from MeshMetrics import DistanceMetrics

CONTOUR_COLORS_BGR = [
    (0, 0, 255),      # red
    (255, 0, 0),      # blue
    (0, 200, 0),      # green
    (0, 200, 255),    # orange
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (128, 0, 255),    # pink
    (255, 128, 0),    # light blue
    (0, 128, 255),    # dark orange
]

CONTOUR_COLOR_NAMES = [
    "red", "blue", "green", "orange", "magenta",
    "cyan", "yellow", "pink", "light-blue", "dark-orange",
]


def detect_red_borders(img):
    """
    Detect red-colored pixels (borders) in an image using HSV color space.

    Args:
        img: Input image (H, W, 3) numpy array in BGR format

    Returns:
        Binary mask where True indicates red pixels (borders)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([30, 255, 255])
    lower_red2 = np.array([150, 30, 30])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    return (mask1 | mask2) > 0


def extract_mask_from_annotation(annotation_path):
    """
    Extract binary mask from an annotation image by detecting red borders
    and finding the enclosed region.

    Args:
        annotation_path: Path to annotation image file

    Returns:
        Binary mask (numpy array of bool) where True indicates the annotated region
    """
    img = cv2.imread(annotation_path)
    if img is None:
        raise ValueError(f"Could not load image from {annotation_path}")

    h, w = img.shape[:2]
    red_mask = detect_red_borders(img)

    if np.sum(red_mask) == 0:
        return np.zeros((h, w), dtype=bool)

    non_red_mask = ~red_mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        non_red_mask.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        return np.zeros((h, w), dtype=bool)

    enclosed_components = []
    for i in range(1, num_labels):
        component_mask = (labels == i)
        touches_border = (
            np.any(component_mask[0, :]) or
            np.any(component_mask[-1, :]) or
            np.any(component_mask[:, 0]) or
            np.any(component_mask[:, -1])
        )
        if not touches_border:
            enclosed_components.append(i)

    if enclosed_components:
        largest_component_idx = max(
            enclosed_components, key=lambda idx: stats[idx, cv2.CC_STAT_AREA]
        )
    else:
        largest_component_idx = max(
            range(1, num_labels), key=lambda idx: stats[idx, cv2.CC_STAT_AREA]
        )

    return (labels == largest_component_idx)


def compute_segmentation_metrics(mask1, mask2, spacing=(1, 1)):
    """
    Compute segmentation metrics between two binary masks.

    Returns:
        Dictionary with 'iou', 'nsd', 'dice'
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    metrics = DistanceMetrics()
    metrics.set_input(mask1, mask2, spacing=spacing)

    return {
        'iou': float(metrics.iou()),
        'nsd': float(metrics.nsd(10)),
        'dice': float(metrics.dsc()),
    }


def _sanitize_path_component(name):
    return re.sub(r'[^\w\-.]', '_', name)


def load_blacklist(parent_folder, filename='blacklist.txt'):
    blacklist_path = Path(parent_folder) / filename
    if not blacklist_path.exists():
        return set()

    blacklist = set()
    with open(blacklist_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                blacklist.add(line)
    return blacklist


def _collect_annotation_files(folder_path, image_extensions):
    annotation_files = []
    for ext in image_extensions:
        annotation_files.extend(folder_path.glob(ext))
    annotation_files = [f for f in annotation_files
                        if not f.name.startswith('masked_')]
    return sorted(annotation_files)


def _get_base_image(annotation_path):
    """
    Build a greyscale base image from an annotation by replacing red border
    pixels with the mean grey value of the surrounding non-red region.  The
    result is a 3-channel BGR image suitable for drawing coloured contours on.
    """
    img = cv2.imread(annotation_path)
    if img is None:
        raise ValueError(f"Could not load image from {annotation_path}")

    red_mask = detect_red_borders(img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    non_red_mean = int(np.mean(grey[~red_mask])) if np.any(~red_mask) else 128
    grey[red_mask] = non_red_mean

    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def draw_contours_overlay(base_img, masks, labels, contour_thickness=2):
    """
    Draw contours from multiple masks onto a base image, each in a different colour.

    Args:
        base_img: BGR image to draw on (will be copied)
        masks: list of binary masks
        labels: list of label strings (same length as masks)
        contour_thickness: line thickness for contours

    Returns:
        BGR image with contours drawn
    """
    overlay = base_img.copy()
    h, w = overlay.shape[:2]

    for idx, (mask, label) in enumerate(zip(masks, labels)):
        color = CONTOUR_COLORS_BGR[idx % len(CONTOUR_COLORS_BGR)]

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)

    legend_y = 20
    for idx, label in enumerate(labels):
        color = CONTOUR_COLORS_BGR[idx % len(CONTOUR_COLORS_BGR)]
        color_name = CONTOUR_COLOR_NAMES[idx % len(CONTOUR_COLOR_NAMES)]
        text = f"{label} ({color_name})"
        cv2.putText(
            overlay, text, (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
        legend_y += 18

    return overlay


def find_annotation_groups(parent_folder):
    """
    Find groups of annotation files to compare.

    Supports two layouts:
      1) Per-subfolder groups: each subfolder contains 2+ annotation images.
      2) N-expert layout: N subfolders (one per expert), files matched by name
         across all subfolders.

    Returns:
        List of tuples: [(group_name, [file_path, ...]), ...]
    """
    parent_path = Path(parent_folder)
    if not parent_path.exists():
        raise ValueError(f"Parent folder does not exist: {parent_folder}")

    image_extensions = [
        '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
        '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF',
    ]

    subfolders = [d for d in sorted(parent_path.iterdir()) if d.is_dir()]

    # Layout 2: multiple expert subfolders — match files by name
    if len(subfolders) >= 2:
        per_folder = {}
        for sf in subfolders:
            files = _collect_annotation_files(sf, image_extensions)
            per_folder[sf] = {f.name: f for f in files}

        all_names = [set(m.keys()) for m in per_folder.values()]
        common_names = set.intersection(*all_names) if all_names else set()

        if common_names:
            groups = []
            for name in sorted(common_names):
                file_paths = [str(per_folder[sf][name]) for sf in subfolders]
                groups.append((name, file_paths))

            for sf, name_map in per_folder.items():
                only_here = set(name_map.keys()) - common_names
                if only_here:
                    print(f"Warning: files only in {sf.name}: {sorted(only_here)}")
            return groups

    # Layout 1: each subfolder is one comparison group with 2+ files
    groups = []
    for subfolder in subfolders:
        annotation_files = _collect_annotation_files(subfolder, image_extensions)
        if len(annotation_files) >= 2:
            groups.append((
                subfolder.name,
                [str(f) for f in annotation_files],
            ))
        elif len(annotation_files) == 1:
            print(f"Warning: only 1 file in {subfolder}, need at least 2. Skipping.")

    return groups


def _save_group_outputs(masks, file_paths, group_name, group_idx, tmp_dir):
    """Save masks and overlay image for one comparison group."""
    tmp_dir = Path(tmp_dir)
    subdir_name = f"{group_idx:03d}_{_sanitize_path_component(group_name)}"
    group_dir = tmp_dir / subdir_name
    group_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for i, fp in enumerate(file_paths):
        stem = Path(fp).stem
        parent_name = Path(fp).parent.name
        label = f"{parent_name}/{stem}" if len(file_paths) > 1 else stem
        labels.append(label)

        mask_uint8 = masks[i].astype(np.uint8) * 255
        mask_path = group_dir / f"mask_{i}_{Path(fp).name}"
        cv2.imwrite(str(mask_path), mask_uint8)

    base_img = _get_base_image(file_paths[0])
    overlay = draw_contours_overlay(base_img, masks, labels)
    overlay_path = group_dir / "overlay_contours.png"
    cv2.imwrite(str(overlay_path), overlay)

    return str(group_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple annotation images using red border detection '
                    'and pairwise segmentation metrics'
    )
    parser.add_argument(
        '--parent-folder', type=str, required=True,
        help='Path to parent folder containing subfolders with annotations',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save results CSV (default: print to stdout)',
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Print detailed information for each group',
    )
    parser.add_argument(
        '--save-masks', action='store_true', dest='save_masks',
        help='Save extracted masks and overlay images',
    )
    parser.add_argument(
        '--no-save-masks', action='store_false', dest='save_masks',
        help='Do not save masks or overlay images',
    )
    parser.set_defaults(save_masks=True)
    parser.add_argument(
        '--tmp-dir', type=str, default=None,
        help='Directory for saved masks/overlays (default: tmp/compare_annotation_multi)',
    )

    args = parser.parse_args()

    print(f"Scanning parent folder: {args.parent_folder}")
    groups = find_annotation_groups(args.parent_folder)

    if not groups:
        print("No annotation groups found!")
        sys.exit(1)

    blacklist = load_blacklist(args.parent_folder)
    if blacklist:
        print(f"Blacklist loaded: {len(blacklist)} file(s) excluded")
        original_count = len(groups)
        filtered = []
        for group_name, file_paths in groups:
            clean_paths = [
                fp for fp in file_paths if Path(fp).name not in blacklist
            ]
            if len(clean_paths) >= 2:
                filtered.append((group_name, clean_paths))
        groups = filtered
        skipped = original_count - len(groups)
        if skipped:
            print(f"Skipped {skipped} group(s) due to blacklist")

    if not groups:
        print("No annotation groups remaining after applying blacklist!")
        sys.exit(1)

    total_files = sum(len(fps) for _, fps in groups)
    print(f"Found {len(groups)} group(s) with {total_files} total annotations")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path("tmp") / "compare_annotation_multi"
    if args.save_masks:
        tmp_dir = tmp_dir.resolve()
        print(f"Masks and overlays will be saved to: {tmp_dir}")

    all_pairwise = []
    results = []

    for group_idx, (group_name, file_paths) in enumerate(groups):
        try:
            n = len(file_paths)
            if args.verbose:
                print(f"\nGroup '{group_name}' ({n} annotations):")
                for fp in file_paths:
                    print(f"  - {Path(fp).parent.name}/{Path(fp).name}")

            masks = []
            for fp in file_paths:
                mask = extract_mask_from_annotation(fp)
                masks.append(mask)
                if args.verbose:
                    print(f"  Mask {Path(fp).name}: shape={mask.shape}, "
                          f"pixels={np.sum(mask)}")

            if args.save_masks:
                saved_dir = _save_group_outputs(
                    masks, file_paths, group_name, group_idx, tmp_dir
                )
                if args.verbose:
                    print(f"  Saved to: {saved_dir}")

            for (i, j) in combinations(range(n), 2):
                m = compute_segmentation_metrics(masks[i], masks[j])
                all_pairwise.append(m)

                name_i = Path(file_paths[i]).name
                name_j = Path(file_paths[j]).name
                parent_i = Path(file_paths[i]).parent.name
                parent_j = Path(file_paths[j]).parent.name
                label_i = f"{parent_i}/{name_i}" if parent_i != parent_j else name_i
                label_j = f"{parent_j}/{name_j}" if parent_i != parent_j else name_j

                results.append({
                    'group': group_name,
                    'annotation_a': label_i,
                    'annotation_b': label_j,
                    'iou': m['iou'],
                    'nsd': m['nsd'],
                    'dice': m['dice'],
                })

                if args.verbose:
                    print(f"  {label_i} vs {label_j}: "
                          f"IoU={m['iou']:.4f}  NSD={m['nsd']:.4f}  "
                          f"Dice={m['dice']:.4f}")

        except Exception as e:
            print(f"Error processing group '{group_name}': {e}")
            continue

    if not all_pairwise:
        print("No pairs were successfully processed!")
        sys.exit(1)

    iou_vals = [m['iou'] for m in all_pairwise]
    nsd_vals = [m['nsd'] for m in all_pairwise]
    dice_vals = [m['dice'] for m in all_pairwise]

    print(f"\n{'=' * 60}")
    print(f"Aggregated Pairwise Metrics ({len(all_pairwise)} pairs "
          f"across {len(groups)} groups):")
    print(f"{'=' * 60}")
    print(f"IoU:  {np.mean(iou_vals):.4f} +/- {np.std(iou_vals):.4f}")
    print(f"NSD:  {np.mean(nsd_vals):.4f} +/- {np.std(nsd_vals):.4f}")
    print(f"Dice: {np.mean(dice_vals):.4f} +/- {np.std(dice_vals):.4f}")
    print(f"{'=' * 60}\n")

    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Detailed results saved to: {args.output}")
    elif args.verbose:
        df = pd.DataFrame(results)
        print("\nDetailed Results:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
