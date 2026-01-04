import argparse
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


def find_annotation_pairs(parent_folder):
    """
    Find pairs of annotation files in subfolders of the parent folder.
    Assumes each subfolder contains exactly 2 annotation images.
    
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
    
    pairs = []
    
    # Iterate through subfolders
    for subfolder in sorted(parent_path.iterdir()):
        if not subfolder.is_dir():
            continue
        
        # Find all image files in this subfolder
        annotation_files = []
        for ext in image_extensions:
            annotation_files.extend(subfolder.glob(ext))
        
        # Filter out files starting with 'masked_' if needed (optional)
        annotation_files = [f for f in annotation_files 
                          if not f.name.startswith('masked_')]
        
        annotation_files = sorted(annotation_files)
        
        if len(annotation_files) == 2:
            pairs.append((str(subfolder), str(annotation_files[0]), str(annotation_files[1])))
        elif len(annotation_files) > 2:
            print(f"Warning: Found {len(annotation_files)} files in {subfolder}, expected 2. Skipping.")
        elif len(annotation_files) == 1:
            print(f"Warning: Found only 1 file in {subfolder}, expected 2. Skipping.")
        # If 0 files, silently skip
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Compare annotation pairs using red border detection and segmentation metrics'
    )
    parser.add_argument(
        '--parent_folder',
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
    
    args = parser.parse_args()
    
    # Find all annotation pairs
    print(f"Scanning parent folder: {args.parent_folder}")
    pairs = find_annotation_pairs(args.parent_folder)
    
    if len(pairs) == 0:
        print("No annotation pairs found!")
        sys.exit(1)
    
    print(f"Found {len(pairs)} annotation pair(s)")
    
    # Process each pair
    all_metrics = []
    results = []
    
    for folder_path, file1_path, file2_path in pairs:
        try:
            if args.verbose:
                print(f"\nProcessing pair in {folder_path}:")
                print(f"  File 1: {Path(file1_path).name}")
                print(f"  File 2: {Path(file2_path).name}")
            
            # Extract masks from annotations
            mask1 = extract_mask_from_annotation(file1_path)
            mask2 = extract_mask_from_annotation(file2_path)
            
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
