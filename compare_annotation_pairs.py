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


def compute_segmentation_metrics(mask1, mask2, spacing=(1, 1), nsd_tau=10):
    """
    Compute segmentation metrics between two binary masks.
    
    Args:
        mask1: First binary mask (numpy array of bool)
        mask2: Second binary mask (numpy array of bool)
        spacing: Spacing for distance metrics (default: (1, 1))
        nsd_tau: NSD tolerance in pixels (default: 10)
    
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
    nsd = metrics.nsd(nsd_tau)
    dice = metrics.dsc()
    
    return {
        'iou': float(iou),
        'nsd': float(nsd),
        'dice': float(dice)
    }


def _sanitize_path_component(name):
    """Replace disallowed characters for use in directory/file names."""
    return re.sub(r'[^\w\-.]', '_', name)


def _strip_annotation_suffix(stem):
    """Remove trailing _sag_var / _sag / _var from an annotation stem."""
    name = stem
    for suffix in ('_sag_var', '_sag', '_var'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def annotation_stem_to_study_number(annotation_stem):
    """
    Map an annotation filename stem to the Study number in video_measurements.csv.

    Examples:
      VU0800_sag_var     -> VU0800
      VU1000_1_sag_var   -> VU1000_1
      VU0864_2_sag_var   -> VU0864_2
      VU01001_sag_var    -> VU1001
    """
    name = _strip_annotation_suffix(annotation_stem)
    # Fix accidental leading zero in 5-digit IDs (VU01001 -> VU1001)
    name = re.sub(r'^VU0(\d{4})$', r'VU\1', name)
    return name


def load_video_measurements(parent_folder, filename='video_measurements.csv'):
    """
    Load per-study NSD tolerance (tau) from video_measurements.csv.

    Returns:
        Dict mapping study number -> tau (Pixels 1 mm column value)
    """
    csv_path = Path(parent_folder) / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Video measurements CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    if 'Study number' not in df.columns or 'Pixels 1 mm' not in df.columns:
        raise ValueError(
            f"{csv_path} must contain 'Study number' and 'Pixels 1 mm' columns"
        )

    measurements = {}
    for _, row in df.iterrows():
        study = str(row['Study number']).strip()
        tau = row['Pixels 1 mm']
        if pd.isna(tau):
            print(f"Warning: missing Pixels 1 mm for study {study}, skipping row")
            continue
        measurements[study] = float(tau)
    return measurements


def get_nsd_tau_for_annotation(annotation_name, measurements):
    """
    Resolve NSD tau for an annotation filename via its Study number.

    Tries the mapped study number first; if missing, falls back to the base
    ID without a trailing _N suffix (e.g. VU0864_2 -> VU0864).
    """
    stem = Path(annotation_name).stem
    study = annotation_stem_to_study_number(stem)
    if study in measurements:
        return measurements[study], study

    base_study = re.sub(r'_\d+$', '', study)
    if base_study != study and base_study in measurements:
        return measurements[base_study], base_study

    raise KeyError(
        f"No Pixels 1 mm entry for annotation {annotation_name} "
        f"(study number {study})"
    )


def _annotation_identifiers(annotation_name, measurements=None):
    """
    Identifiers that can match a blacklist entry for an annotation file.

    Includes full filename, stem, raw/normalized study IDs, and (when
    measurements are available) the resolved Study number used for NSD tau.
    """
    path = Path(annotation_name)
    name = path.name
    stem = path.stem
    raw_id = _strip_annotation_suffix(stem)
    study = annotation_stem_to_study_number(stem)

    identifiers = {name, stem, raw_id, study}

    if measurements is not None:
        try:
            _, resolved_study = get_nsd_tau_for_annotation(name, measurements)
            identifiers.add(resolved_study)
        except KeyError:
            pass
    else:
        base_study = re.sub(r'_\d+$', '', study)
        if base_study != study:
            identifiers.add(base_study)

    return identifiers


def _expand_blacklist_entry(entry):
    """Expand a blacklist line into matchable filename / stem / study ID forms."""
    path = Path(entry)
    expanded = {entry, path.name, path.stem}
    expanded.add(annotation_stem_to_study_number(path.stem))
    expanded.add(annotation_stem_to_study_number(entry))
    expanded.add(_strip_annotation_suffix(path.stem))
    return {value for value in expanded if value}


def load_blacklist(parent_folder, filename='blacklist.txt'):
    """
    Load a blacklist from a TXT file in the parent folder.

    Each non-empty, non-# line may be a full filename (e.g. VU0866_sag_var.png)
    or a study / sample ID (e.g. VU0866, VU1000_1).

    Args:
        parent_folder: Path to parent folder containing the blacklist file
        filename: Name of the blacklist file (default: blacklist.txt)

    Returns:
        (raw_entries, match_set) where raw_entries preserves the original lines
        and match_set contains all expanded forms used for matching. Both are
        empty if the file does not exist.
    """
    blacklist_path = Path(parent_folder) / filename
    if not blacklist_path.exists():
        return [], set()

    raw_entries = []
    match_set = set()
    with open(blacklist_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                raw_entries.append(line)
                match_set.update(_expand_blacklist_entry(line))
    return raw_entries, match_set


def filter_pairs_by_blacklist(pairs, blacklist_match_set, measurements=None):
    """
    Remove pairs whose annotation identifiers overlap the blacklist.

    Returns:
        (kept_pairs, excluded_pairs) where excluded_pairs is a list of dicts
        with keys file1, file2, matched_ids.
    """
    if not blacklist_match_set:
        return pairs, []

    kept = []
    excluded = []
    for folder_path, file1, file2 in pairs:
        matched = sorted(
            (
                _annotation_identifiers(file1, measurements)
                | _annotation_identifiers(file2, measurements)
            )
            & blacklist_match_set
        )
        if matched:
            excluded.append({
                'file1': Path(file1).name,
                'file2': Path(file2).name,
                'matched_ids': matched,
            })
        else:
            kept.append((folder_path, file1, file2))
    return kept, excluded


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


def _annotators_from_pair(file1_path, file2_path):
    """
    Resolve annotator labels for a compared pair.

    Prefer parent-folder names (two-expert layout). If both files share a
    folder (per-subfolder layout), fall back to the file stems when they differ.
    """
    path1 = Path(file1_path)
    path2 = Path(file2_path)
    annotator_a = path1.parent.name
    annotator_b = path2.parent.name

    if annotator_a == annotator_b:
        stem_a, stem_b = path1.stem, path2.stem
        if stem_a != stem_b:
            annotator_a, annotator_b = stem_a, stem_b

    return annotator_a, annotator_b, f"{annotator_a} vs {annotator_b}"


def _normalize_annotator_pair(annotator_a, annotator_b):
    if annotator_a <= annotator_b:
        return annotator_a, annotator_b
    return annotator_b, annotator_a


def build_average_results(results_df, include_total=False):
    """
    Build per-annotator-pair summary with mean and std per metric.

    When include_total is True, also append a TOTAL row aggregated over all
    rows in results_df (i.e. whatever is in that results file).
    """
    metric_cols = ['iou', 'nsd', 'dice']
    working = results_df.copy()
    pair_cols = working.apply(
        lambda row: _normalize_annotator_pair(
            row['annotator_a'], row['annotator_b'],
        ),
        axis=1,
        result_type='expand',
    )
    working['annotator_a'] = pair_cols[0]
    working['annotator_b'] = pair_cols[1]

    summary_rows = []
    for (annotator_a, annotator_b), group in working.groupby(
        ['annotator_a', 'annotator_b'], sort=True,
    ):
        row = {
            'annotator_a': annotator_a,
            'annotator_b': annotator_b,
            'annotator_pair': f"{annotator_a} vs {annotator_b}",
            'n': len(group),
        }
        for metric in metric_cols:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = (
                group[metric].std(ddof=1) if len(group) > 1 else 0.0
            )
        summary_rows.append(row)

    if include_total:
        total_row = {
            'annotator_a': 'TOTAL',
            'annotator_b': 'TOTAL',
            'annotator_pair': 'TOTAL',
            'n': len(working),
        }
        for metric in metric_cols:
            total_row[f'{metric}_mean'] = working[metric].mean()
            total_row[f'{metric}_std'] = (
                working[metric].std(ddof=1) if len(working) > 1 else 0.0
            )
        summary_rows.append(total_row)

    column_order = [
        'annotator_pair', 'annotator_a', 'annotator_b', 'n',
        'iou_mean', 'iou_std', 'nsd_mean', 'nsd_std', 'dice_mean', 'dice_std',
    ]
    return pd.DataFrame(summary_rows)[column_order]


def _average_output_path(output_path):
    path = Path(output_path)
    return path.with_name(f"{path.stem}_avg{path.suffix}")


def _print_summary_block(summary_df):
    print('=' * 60)
    for _, row in summary_df.iterrows():
        print(
            f"{row['annotator_pair']:>24}  (n={int(row['n']):>3})  "
            f"IoU={row['iou_mean']:.4f} +/- {row['iou_std']:.4f}  "
            f"NSD={row['nsd_mean']:.4f} +/- {row['nsd_std']:.4f}  "
            f"Dice={row['dice_mean']:.4f} +/- {row['dice_std']:.4f}"
        )
    print('=' * 60)
    print()


def find_annotation_pairs(parent_folder):
    """
    Find pairs of annotation files in subfolders of the parent folder.
    Supports two layouts:
    1) Per-subfolder pairs: each subfolder contains exactly 2 annotation images.
    2) Two-expert layout: exactly 2 subfolders, each with all annotations from one
       expert; pairs are formed by matching filenames between the two subfolders.

    Pairs matching blacklist.txt entries (full filenames or study IDs) are
    filtered out in main(), not here.

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
            print(
                f"Annotator pair: {folder1.name} vs {folder2.name}"
            )
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
        required=True,
        help='Path to parent folder containing subfolders with annotation pairs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Path to save results CSV; if the file already exists, '
             'new rows are appended (default: results.csv)'
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
    parser.add_argument(
        '--measurements-csv',
        type=str,
        default='video_measurements.csv',
        help='CSV with per-study NSD tau in the "Pixels 1 mm" column '
             '(default: video_measurements.csv in parent folder)',
    )
    parser.add_argument(
        '--include-total',
        action='store_true',
        help='Include a TOTAL row in results_avg.csv / stdout summary, '
             'aggregated over all rows in the output results CSV',
    )
    
    args = parser.parse_args()
    
    # Find all annotation pairs
    print(f"Scanning parent folder: {args.parent_folder}")
    video_measurements = load_video_measurements(
        args.parent_folder, filename=args.measurements_csv,
    )
    print(
        f"Loaded NSD tau for {len(video_measurements)} studies "
        f"from {args.measurements_csv}"
    )
    pairs = find_annotation_pairs(args.parent_folder)
    
    if len(pairs) == 0:
        print("No annotation pairs found!")
        sys.exit(1)
    
    # Load blacklist (full filenames or study IDs) and filter pairs
    blacklist_entries, blacklist_match_set = load_blacklist(args.parent_folder)
    if blacklist_entries:
        print(
            f"Blacklist loaded: {len(blacklist_entries)} entr"
            f"{'y' if len(blacklist_entries) == 1 else 'ies'} "
            f"(filenames or study IDs)"
        )
        pairs, excluded_pairs = filter_pairs_by_blacklist(
            pairs, blacklist_match_set, measurements=video_measurements,
        )
        if excluded_pairs:
            print(f"Skipped {len(excluded_pairs)} pair(s) due to blacklist:")
            for item in excluded_pairs:
                matched = ', '.join(item['matched_ids'])
                print(f"  - {item['file1']} (matched: {matched})")
        unmatched_entries = [
            entry for entry in blacklist_entries
            if not (
                _expand_blacklist_entry(entry)
                & {
                    matched_id
                    for item in excluded_pairs
                    for matched_id in item['matched_ids']
                }
            )
        ]
        if unmatched_entries:
            print(
                "Warning: blacklist entr"
                f"{'y' if len(unmatched_entries) == 1 else 'ies'} "
                "did not match any pair: "
                + ', '.join(unmatched_entries)
            )

    if len(pairs) == 0:
        print("No annotation pairs remaining after applying blacklist!")
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
            file1_name = Path(file1_path).name
            file2_name = Path(file2_path).name
            annotator_a, annotator_b, annotator_pair = _annotators_from_pair(
                file1_path, file2_path,
            )
            nsd_tau, study_number = get_nsd_tau_for_annotation(
                file1_name, video_measurements,
            )

            if args.verbose:
                print(f"\nProcessing pair in {folder_path}:")
                print(f"  Annotators: {annotator_pair}")
                print(f"  File 1: {file1_name}")
                print(f"  File 2: {file2_name}")
                print(f"  Study: {study_number}, NSD tau={nsd_tau:.0f} px")
            
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
            
            # Compute metrics with per-study NSD tau
            metrics = compute_segmentation_metrics(
                mask1, mask2, nsd_tau=nsd_tau,
            )
            
            all_metrics.append(metrics)
            results.append({
                'folder': folder_path,
                'annotator_pair': annotator_pair,
                'annotator_a': annotator_a,
                'annotator_b': annotator_b,
                'file1': file1_name,
                'file2': file2_name,
                'study_number': study_number,
                'nsd_tau': nsd_tau,
                'iou': metrics['iou'],
                'nsd': metrics['nsd'],
                'dice': metrics['dice']
            })
            
            if args.verbose:
                print(f"  IoU: {metrics['iou']:.4f}")
                print(f"  NSD: {metrics['nsd']:.4f} (tau={nsd_tau:.0f})")
                print(f"  Dice: {metrics['dice']:.4f}")
        
        except Exception as e:
            print(f"Error processing pair in {folder_path}: {e}")
            continue
    
    if len(all_metrics) == 0:
        print("No pairs were successfully processed!")
        sys.exit(1)

    df_new = pd.DataFrame(results)
    avg_df_run = build_average_results(
        df_new, include_total=args.include_total,
    )

    print(f"\n{'=' * 60}")
    print(
        f"Aggregated Pairwise Metrics ({len(all_metrics)} pairs "
        f"in this run):"
    )
    _print_summary_block(avg_df_run)

    # Save detailed results if output path is provided
    if args.output:
        output_path = Path(args.output)
        if output_path.exists() and output_path.stat().st_size > 0:
            existing = pd.read_csv(output_path)
            df = pd.concat([existing, df_new], ignore_index=True)
            df.to_csv(output_path, index=False)
            print(
                f"Appended {len(df_new)} row(s) to existing CSV "
                f"({len(existing)} -> {len(df)}): {output_path}"
            )
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df = df_new
            df.to_csv(output_path, index=False)
            print(f"Detailed results saved to: {output_path}")

        # TOTAL (if requested) is over all rows currently in this results file
        avg_df = build_average_results(
            df, include_total=args.include_total,
        )
        avg_output_path = _average_output_path(output_path)
        avg_df.to_csv(avg_output_path, index=False)
        print(f"Average results saved to: {avg_output_path}")
        if len(df) != len(df_new):
            print(f"\n{'=' * 60}")
            print(
                f"Aggregated Pairwise Metrics over full CSV "
                f"({len(df)} pairs):"
            )
            _print_summary_block(avg_df)
    elif args.verbose:
        print("\nDetailed Results:")
        print(df_new.to_string(index=False))
        print("\nAverage Results:")
        print(avg_df_run.to_string(index=False))


if __name__ == "__main__":
    main()
