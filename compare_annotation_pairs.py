import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from MeshMetrics import DistanceMetrics


def detect_red_borders(img):
    """
    Detect red-colored pixels (borders) in an image using HSV color space.
    Similar approach to preprocess_image.py but returns a binary mask instead of replacing pixels.
    
    Args:
        img: Input image (H, W, 3) numpy array in BGR format (OpenCV default)
    
    Returns:
        Binary mask where True indicates red pixels (borders)
    """
    # Convert BGR to HSV for strict hue-based red detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red detection based on hue with relaxed saturation/value to catch vague red colors:
    # Red hue wraps around: 0-30 degrees (red-orange) and 150-180 degrees (red-magenta)
    # Lower thresholds for saturation and value to detect vague/weak red colors
    
    # Lower bound for red (hue 0-30, enlarged)
    lower_red1 = np.array([0, 30, 30])  # Lower thresholds to catch vague red colors
    upper_red1 = np.array([30, 255, 255])  # Enlarged hue range
    
    # Upper bound for red (hue 150-180, enlarged)
    lower_red2 = np.array([150, 30, 30])  # Lower thresholds to catch vague red colors
    upper_red2 = np.array([180, 255, 255])  # Enlarged hue range
    
    # Apply strict hue-based masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine both masks (red pixels match either hue range)
    red_mask = mask1 | mask2
    
    # Convert to boolean mask
    return red_mask > 0


def extract_mask_from_annotation(annotation_path):
    """
    Extract binary mask from an annotation image by detecting red borders.
    The red borders outline the annotated region, so we extract the enclosed region.
    
    Args:
        annotation_path: Path to annotation image file
    
    Returns:
        Binary mask (numpy array of bool) where True indicates the annotated region
    """
    img = cv2.imread(annotation_path)
    
    if img is None:
        raise ValueError(f"Could not load image from {annotation_path}")
    
    h, w = img.shape[:2]
    
    # Detect red borders
    red_mask = detect_red_borders(img)
    
    # If no red pixels found, return empty mask
    if np.sum(red_mask) == 0:
        return np.zeros((h, w), dtype=bool)
    
    # Create a mask for non-red regions
    non_red_mask = ~red_mask
    
    # Find connected components in non-red regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        non_red_mask.astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 1:
        # No components found, return empty mask
        return np.zeros((h, w), dtype=bool)
    
    # Find components that are not touching the image borders
    # These are likely the enclosed regions
    enclosed_components = []
    
    for i in range(1, num_labels):  # Skip background label 0
        component_mask = (labels == i)
        
        # Check if component touches any border
        touches_border = (
            np.any(component_mask[0, :]) or
            np.any(component_mask[-1, :]) or
            np.any(component_mask[:, 0]) or
            np.any(component_mask[:, -1])
        )
        
        if not touches_border:
            enclosed_components.append(i)
    
    # If we found enclosed components, use the largest one
    # Otherwise, use the largest component overall
    if len(enclosed_components) > 0:
        largest_component_idx = max(enclosed_components, 
                                   key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
    else:
        # Find the largest component (excluding background label 0)
        largest_component_idx = 1
        largest_area = stats[1, cv2.CC_STAT_AREA]
        
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[i, cv2.CC_STAT_AREA]
                largest_component_idx = i
    
    # Create binary mask for the selected component
    mask = (labels == largest_component_idx)
    
    return mask


def compute_segmentation_metrics(mask1, mask2, spacing=(1, 1)):
    """
    Compute segmentation metrics between two binary masks.
    
    Args:
        mask1: First binary mask (numpy array of bool)
        mask2: Second binary mask (numpy array of bool)
        spacing: Spacing for distance metrics (default: (1, 1))
    
    Returns:
        Dictionary with metrics: 'iou', 'nsd', 'dice'
    """
    # Ensure masks have the same shape
    if mask1.shape != mask2.shape:
        # Resize mask2 to match mask1
        mask2 = cv2.resize(mask2.astype(np.uint8), 
                          (mask1.shape[1], mask1.shape[0]), 
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Ensure boolean type
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Initialize metrics calculator
    metrics = DistanceMetrics()
    metrics.set_input(mask1, mask2, spacing=spacing)
    
    # Compute metrics
    iou = metrics.iou()
    nsd = metrics.nsd(6)
    dice = metrics.dsc()
    
    return {
        'iou': float(iou),
        'nsd': float(nsd),
        'dice': float(dice)
    }


def _sanitize_path_component(name):
    """Replace disallowed characters for use in directory/file names."""
    return re.sub(r'[^\w\-.]', '_', name)


def _save_masks_to_tmp(mask1, mask2, file1_path, file2_path, folder_path, pair_idx, tmp_dir):
    """Save extracted masks as PNG images to tmp_dir."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    stem1 = Path(file1_path).stem
    stem2 = Path(file2_path).stem
    folder_name = Path(folder_path).name
    
    if stem1 == stem2:
        subdir_name = f"{pair_idx:03d}_{_sanitize_path_component(stem1)}"
    else:
        subdir_name = f"{pair_idx:03d}_{_sanitize_path_component(folder_name)}_{_sanitize_path_component(stem1)}_vs_{_sanitize_path_component(stem2)}"
    
    pair_dir = tmp_dir / subdir_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as uint8 (0/255) PNG
    mask1_uint8 = (mask1.astype(np.uint8) * 255)
    mask2_uint8 = (mask2.astype(np.uint8) * 255)
    
    path1 = pair_dir / f"mask1_{Path(file1_path).name}"
    path2 = pair_dir / f"mask2_{Path(file2_path).name}"
    
    cv2.imwrite(str(path1), mask1_uint8)
    cv2.imwrite(str(path2), mask2_uint8)
    
    return str(pair_dir)


def _collect_annotation_files(folder_path, image_extensions):
    """Collect annotation image files from a folder, excluding masked_ prefix."""
    annotation_files = []
    for ext in image_extensions:
        annotation_files.extend(folder_path.glob(ext))
    annotation_files = [f for f in annotation_files
                       if not f.name.startswith('masked_')]
    return sorted(annotation_files)


def find_annotation_pairs(parent_folder):
    """
    Find pairs of annotation files in subfolders of the parent folder.
    Supports two layouts:
    1) Per-subfolder pairs: each subfolder contains exactly 2 annotation images.
    2) Two-expert layout: exactly 2 subfolders, each with all annotations from one
       expert; pairs are formed by matching filenames between the two subfolders.
    
    Args:
        parent_folder: Path to parent folder containing subfolders with annotation pairs
    
    Returns:
        List of tuples: [(folder_path, file1_path, file2_path), ...]
    """
    parent_path = Path(parent_folder)
    if not parent_path.exists():
        raise ValueError(f"Parent folder does not exist: {parent_folder}")
    
    # Image extensions to look for
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                        '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
    
    subfolders = [d for d in sorted(parent_path.iterdir()) if d.is_dir()]
    
    # Layout 2: exactly 2 subfolders - match files by filename between experts
    if len(subfolders) == 2:
        folder1, folder2 = subfolders[0], subfolders[1]
        files1 = _collect_annotation_files(folder1, image_extensions)
        files2 = _collect_annotation_files(folder2, image_extensions)
        
        # Build filename -> path maps for matching
        by_name1 = {f.name: f for f in files1}
        by_name2 = {f.name: f for f in files2}
        
        common_names = set(by_name1.keys()) & set(by_name2.keys())
        if common_names:
            # Use parent folder as the logical "folder" for this layout
            pairs = [
                (str(parent_path), str(by_name1[name]), str(by_name2[name]))
                for name in sorted(common_names)
            ]
            if len(by_name1) != len(by_name2) or len(common_names) != len(by_name1):
                only_in_1 = set(by_name1.keys()) - common_names
                only_in_2 = set(by_name2.keys()) - common_names
                if only_in_1:
                    print(f"Warning: Files only in {folder1.name}: {sorted(only_in_1)}")
                if only_in_2:
                    print(f"Warning: Files only in {folder2.name}: {sorted(only_in_2)}")
            return pairs
    
    # Layout 1: each subfolder has 2 files
    pairs = []
    for subfolder in subfolders:
        annotation_files = _collect_annotation_files(subfolder, image_extensions)
        
        if len(annotation_files) == 2:
            pairs.append((str(subfolder), str(annotation_files[0]), str(annotation_files[1])))
        elif len(annotation_files) > 2:
            print(f"Warning: Found {len(annotation_files)} files in {subfolder}, expected 2. Skipping.")
        elif len(annotation_files) == 1:
            print(f"Warning: Found only 1 file in {subfolder}, expected 2. Skipping.")
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Compare annotation pairs using red border detection and segmentation metrics'
    )
    parser.add_argument(
        '--parent-folder',
        type=str,
        help='Path to parent folder containing subfolders with annotation pairs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results CSV (default: print to stdout)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed information for each pair'
    )
    parser.add_argument(
        '--save-masks',
        action='store_true',
        dest='save_masks',
        help='Save extracted masks to tmp folder'
    )
    parser.add_argument(
        '--no-save-masks',
        action='store_false',
        dest='save_masks',
        help='Do not save masks to tmp folder'
    )
    parser.set_defaults(save_masks=True)
    parser.add_argument(
        '--tmp-dir',
        type=str,
        default=None,
        help='Directory for saved masks (default: tmp/compare_annotation_masks)'
    )
    
    args = parser.parse_args()
    
    # Find all annotation pairs
    print(f"Scanning parent folder: {args.parent_folder}")
    pairs = find_annotation_pairs(args.parent_folder)
    
    if len(pairs) == 0:
        print("No annotation pairs found!")
        sys.exit(1)
    
    print(f"Found {len(pairs)} annotation pair(s)")
    
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path("tmp") / "compare_annotation_masks"
    if args.save_masks:
        tmp_dir = Path(tmp_dir).resolve()
        print(f"Masks will be saved to: {tmp_dir}")
    
    # Process each pair
    all_metrics = []
    results = []
    
    for pair_idx, (folder_path, file1_path, file2_path) in enumerate(pairs):
        try:
            if args.verbose:
                print(f"\nProcessing pair in {folder_path}:")
                print(f"  File 1: {Path(file1_path).name}")
                print(f"  File 2: {Path(file2_path).name}")
            
            # Extract masks from annotations
            mask1 = extract_mask_from_annotation(file1_path)
            mask2 = extract_mask_from_annotation(file2_path)
            
            # Save masks to tmp if requested
            if args.save_masks:
                saved_dir = _save_masks_to_tmp(
                    mask1, mask2, file1_path, file2_path,
                    folder_path, pair_idx, tmp_dir
                )
                if args.verbose:
                    print(f"  Masks saved to: {saved_dir}")
            
            if args.verbose:
                print(f"  Mask 1 shape: {mask1.shape}, pixels: {np.sum(mask1)}")
                print(f"  Mask 2 shape: {mask2.shape}, pixels: {np.sum(mask2)}")
            
            # Compute metrics
            metrics = compute_segmentation_metrics(mask1, mask2)
            
            all_metrics.append(metrics)
            results.append({
                'folder': folder_path,
                'file1': Path(file1_path).name,
                'file2': Path(file2_path).name,
                'iou': metrics['iou'],
                'nsd': metrics['nsd'],
                'dice': metrics['dice']
            })
            
            if args.verbose:
                print(f"  IoU: {metrics['iou']:.4f}")
                print(f"  NSD: {metrics['nsd']:.4f}")
                print(f"  Dice: {metrics['dice']:.4f}")
        
        except Exception as e:
            print(f"Error processing pair in {folder_path}: {e}")
            continue
    
    if len(all_metrics) == 0:
        print("No pairs were successfully processed!")
        sys.exit(1)
    
    # Aggregate metrics
    iou_values = [m['iou'] for m in all_metrics]
    nsd_values = [m['nsd'] for m in all_metrics]
    dice_values = [m['dice'] for m in all_metrics]
    
    iou_mean = np.mean(iou_values)
    iou_std = np.std(iou_values)
    nsd_mean = np.mean(nsd_values)
    nsd_std = np.std(nsd_values)
    dice_mean = np.mean(dice_values)
    dice_std = np.std(dice_values)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Aggregated Metrics Across {len(all_metrics)} Pairs:")
    print(f"{'='*60}")
    print(f"IoU:  {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"NSD:  {nsd_mean:.4f} ± {nsd_std:.4f}")
    print(f"Dice: {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"{'='*60}\n")
    
    # Save detailed results if output path is provided
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Detailed results saved to: {args.output}")
    elif args.verbose:
        # Print detailed results table
        df = pd.DataFrame(results)
        print("\nDetailed Results:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
