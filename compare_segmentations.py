"""
Compare ground truth and predicted segmentation masks by drawing their contours
on the original images.

Usage:
    python compare_segmentations.py <original_images_folder> <ground_truth_folder> <predictions_folder> [output_folder]

The script matches images by name, handling suffixes like "_000" that may be present in mask files.
Ground truth contours are drawn in BLUE, prediction contours are drawn in GREEN.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from skimage import measure
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def get_image_files(folder_path):
    """Get all image files from a folder"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        return []
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def extract_base_name(filename):
    """
    Extract base name from filename, removing common suffixes like _000, _001, etc.
    Also removes common mask suffixes like _mask, _gt, _pred, etc.
    """
    stem = Path(filename).stem
    
    # Remove common numeric suffixes (e.g., _000, _001, _0000)
    import re
    # Pattern to match _XXX or _XXXX at the end where X is a digit
    pattern = r'_\d{3,4}$'
    base_name = re.sub(pattern, '', stem)
    
    # Remove common mask-related suffixes
    suffixes_to_remove = ['_mask', '_gt', '_pred', '_segmentation', '_seg', '_label']
    for suffix in suffixes_to_remove:
        if base_name.lower().endswith(suffix):
            base_name = base_name[:-len(suffix)]
    
    return base_name


def build_file_mapping(original_files, gt_files, pred_files):
    """
    Build a mapping between original images, ground truth masks, and prediction masks
    based on partial name matching.
    
    Only includes entries where BOTH ground truth AND prediction exist.
    """
    # Create dictionaries with base names as keys
    originals_by_name = {}
    for f in original_files:
        base = extract_base_name(f.name)
        originals_by_name[base] = f
    
    gt_by_name = {}
    for f in gt_files:
        base = extract_base_name(f.name)
        gt_by_name[base] = f
    
    pred_by_name = {}
    for f in pred_files:
        base = extract_base_name(f.name)
        pred_by_name[base] = f
    
    # Find names that have BOTH ground truth AND prediction
    gt_names = set(gt_by_name.keys())
    pred_names = set(pred_by_name.keys())
    paired_names = gt_names & pred_names
    
    # Report missing pairs
    gt_only = gt_names - pred_names
    pred_only = pred_names - gt_names
    
    if gt_only:
        print(f"\nWarning: {len(gt_only)} ground truth mask(s) without matching predictions:")
        for name in sorted(gt_only)[:10]:
            print(f"  - {name}")
        if len(gt_only) > 10:
            print(f"  ... and {len(gt_only) - 10} more")
    
    if pred_only:
        print(f"\nWarning: {len(pred_only)} prediction mask(s) without matching ground truths:")
        for name in sorted(pred_only)[:10]:
            print(f"  - {name}")
        if len(pred_only) > 10:
            print(f"  ... and {len(pred_only) - 10} more")
    
    if not paired_names:
        print("\nError: No matching pairs found between ground truth and prediction folders.")
        print(f"  Ground truth base names: {list(gt_by_name.keys())[:5]}...")
        print(f"  Prediction base names: {list(pred_by_name.keys())[:5]}...")
        return []
    
    # Find which paired names also have original images
    paired_with_originals = paired_names & set(originals_by_name.keys())
    paired_without_originals = paired_names - set(originals_by_name.keys())
    
    if paired_without_originals:
        print(f"\nWarning: {len(paired_without_originals)} paired mask(s) without matching original images:")
        for name in sorted(paired_without_originals)[:10]:
            print(f"  - {name}")
        if len(paired_without_originals) > 10:
            print(f"  ... and {len(paired_without_originals) - 10} more")
    
    # Build the mapping (only for complete triplets)
    mapping = []
    for name in sorted(paired_with_originals):
        mapping.append({
            'base_name': name,
            'original': originals_by_name[name],
            'gt': gt_by_name[name],
            'pred': pred_by_name[name]
        })
    
    print(f"\nMatched {len(mapping)} complete triplets (original + GT + prediction)")
    
    return mapping


def load_mask(mask_path):
    """Load a mask image and convert to binary"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not load mask: {mask_path}")
        return None
    
    # Convert to binary (threshold at 127)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def keep_largest_region(mask):
    """Keep only the largest connected region in the mask using skimage regionprops"""
    if mask is None:
        return None
    
    # Convert to binary (0 or 1)
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Label connected regions
    labeled_mask = measure.label(mask_binary, connectivity=2)
    
    # Get region properties
    regions = measure.regionprops(labeled_mask)
    
    if len(regions) == 0:
        # No regions found, return empty mask
        return np.zeros_like(mask)
    
    # Find the largest region by area
    largest_region = max(regions, key=lambda r: r.area)
    
    # Create mask with only the largest region
    largest_mask = (labeled_mask == largest_region.label).astype(np.uint8) * 255
    
    return largest_mask


def draw_contours(image, mask, color, thickness=2):
    """
    Draw contours of the segmentation mask on the image.
    
    Args:
        image: The image to draw on (will be modified in place)
        mask: Binary mask (0 or 255)
        color: BGR color tuple
        thickness: Contour line thickness
    """
    if mask is None:
        return
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on image
    cv2.drawContours(image, contours, -1, color, thickness)


def get_arial_font(size=12):
    """
    Try to load Arial font, fall back to default if not available.
    """
    # Common paths for Arial font on different systems
    font_paths = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",  # Linux with msttcorefonts
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux alternative
        "/usr/share/fonts/TTF/Arial.ttf",  # Arch Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "/System/Library/Fonts/Helvetica.ttc",  # macOS (similar to Arial)
        "/Library/Fonts/Arial.ttf",  # macOS
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    
    # Fall back to default font
    return ImageFont.load_default()


def add_legend(image, font_size=11):
    """
    Add a small legend to the image using PIL for Arial font.
    Ground truth = Blue, Prediction = Green
    
    Args:
        image: OpenCV image (BGR format)
        font_size: Font size for legend text (default: 11)
    
    Returns:
        Image with legend added
    """
    # Convert from BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Font settings
    font = get_arial_font(font_size)
    
    # Legend positioning (scale with font size)
    legend_x = 8
    legend_y = 8
    box_size = max(8, font_size - 1)
    spacing = max(3, font_size // 3)
    line_height = font_size + 3
    
    # Draw ground truth legend (blue)
    # Rectangle for blue color indicator
    draw.rectangle(
        [legend_x, legend_y, legend_x + box_size, legend_y + box_size],
        fill=(0, 0, 255)  # Blue in RGB
    )
    # Text
    draw.text(
        (legend_x + box_size + spacing, legend_y - 1),
        "Ground Truth",
        font=font,
        fill=(255, 255, 255)
    )
    
    # Draw prediction legend (green)
    pred_y = legend_y + line_height
    draw.rectangle(
        [legend_x, pred_y, legend_x + box_size, pred_y + box_size],
        fill=(0, 255, 0)  # Green in RGB
    )
    draw.text(
        (legend_x + box_size + spacing, pred_y - 1),
        "Prediction",
        font=font,
        fill=(255, 255, 255)
    )
    
    # Convert back to BGR for OpenCV
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return result


def process_image_triplet(original_path, gt_path, pred_path, output_path, keep_largest=True, font_size=11):
    """
    Process a single image triplet: draw GT contours in blue and prediction contours in green.
    Only saves output if BOTH masks have non-zero regions.
    
    Args:
        original_path: Path to original image
        gt_path: Path to ground truth mask
        pred_path: Path to prediction mask
        output_path: Path to save the result
        keep_largest: Whether to keep only the largest connected region in masks
        font_size: Font size for legend text (default: 11)
    
    Returns:
        str: 'success', 'skipped_empty_gt', 'skipped_empty_pred', 'skipped_both_empty', or 'error'
    """
    # Load original image
    original = cv2.imread(str(original_path))
    if original is None:
        print(f"Error: Could not load original image: {original_path}")
        return 'error'
    
    # Load masks
    gt_mask = load_mask(gt_path)
    pred_mask = load_mask(pred_path)
    
    if gt_mask is None or pred_mask is None:
        return 'error'
    
    # Resize masks to match original image size if needed
    orig_h, orig_w = original.shape[:2]
    
    if gt_mask.shape != (orig_h, orig_w):
        gt_mask = cv2.resize(gt_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    if pred_mask.shape != (orig_h, orig_w):
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Optionally keep only largest region
    if keep_largest:
        gt_mask = keep_largest_region(gt_mask)
        pred_mask = keep_largest_region(pred_mask)
    
    # Check if both masks have non-zero regions
    gt_has_content = np.any(gt_mask > 0)
    pred_has_content = np.any(pred_mask > 0)
    
    if not gt_has_content and not pred_has_content:
        return 'skipped_both_empty'
    elif not gt_has_content:
        return 'skipped_empty_gt'
    elif not pred_has_content:
        return 'skipped_empty_pred'
    
    # Create output image (copy of original)
    result = original.copy()
    
    # Draw contours
    # Ground truth in BLUE (BGR: 255, 0, 0)
    draw_contours(result, gt_mask, color=(255, 0, 0), thickness=3)
    
    # Prediction in GREEN (BGR: 0, 255, 0)
    draw_contours(result, pred_mask, color=(0, 255, 0), thickness=3)
    
    # Add legend using PIL for Arial font
    result = add_legend(result, font_size=font_size)
    
    # Save result
    cv2.imwrite(str(output_path), result)
    return 'success'


def main():
    if len(sys.argv) < 4:
        print("Usage: python compare_segmentations.py <original_images_folder> <ground_truth_folder> <predictions_folder> [output_folder] [font_size]")
        print("\nExample:")
        print("  python compare_segmentations.py ./images ./gt_masks ./pred_masks ./comparison_results")
        print("  python compare_segmentations.py ./images ./gt_masks ./pred_masks ./comparison_results 14")
        print("\nDescription:")
        print("  - Matches images by name (handles suffixes like _000)")
        print("  - Draws ground truth contours in BLUE")
        print("  - Draws prediction contours in GREEN")
        print("  - Saves comparison images to output folder")
        print("  - Optional font_size parameter (default: 11)")
        sys.exit(1)
    
    original_folder = sys.argv[1]
    gt_folder = sys.argv[2]
    pred_folder = sys.argv[3]
    output_folder = sys.argv[4] if len(sys.argv) > 4 else "segmentation_comparison"
    font_size = int(sys.argv[5]) if len(sys.argv) > 5 else 11
    
    print(f"Original images folder: {original_folder}")
    print(f"Ground truth folder: {gt_folder}")
    print(f"Predictions folder: {pred_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Font size: {font_size}")
    
    # Get image files from each folder
    original_files = get_image_files(original_folder)
    gt_files = get_image_files(gt_folder)
    pred_files = get_image_files(pred_folder)
    
    print(f"\nFound {len(original_files)} original images")
    print(f"Found {len(gt_files)} ground truth masks")
    print(f"Found {len(pred_files)} prediction masks")
    
    if not original_files or not gt_files or not pred_files:
        print("Error: One or more folders are empty or don't contain valid images")
        sys.exit(1)
    
    # Build file mapping
    file_mapping = build_file_mapping(original_files, gt_files, pred_files)
    
    if not file_mapping:
        print("Error: Could not find matching files between folders")
        sys.exit(1)
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated output folder: {output_path.absolute()}")
    
    # Process each triplet
    successful = 0
    skipped_empty_gt = []
    skipped_empty_pred = []
    skipped_both_empty = []
    errors = []
    
    for item in tqdm(file_mapping, desc="Processing images"):
        output_file = output_path / f"{item['base_name']}_comparison.png"
        
        result = process_image_triplet(
            item['original'],
            item['gt'],
            item['pred'],
            output_file,
            font_size=font_size
        )
        
        if result == 'success':
            successful += 1
        elif result == 'skipped_empty_gt':
            skipped_empty_gt.append(item['base_name'])
        elif result == 'skipped_empty_pred':
            skipped_empty_pred.append(item['base_name'])
        elif result == 'skipped_both_empty':
            skipped_both_empty.append(item['base_name'])
        else:
            errors.append(item['base_name'])
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful} images")
    print(f"  (only images with non-zero regions in BOTH masks)")
    
    if skipped_empty_gt:
        print(f"\nSkipped (empty ground truth): {len(skipped_empty_gt)}")
        for name in skipped_empty_gt[:10]:
            print(f"  - {name}")
        if len(skipped_empty_gt) > 10:
            print(f"  ... and {len(skipped_empty_gt) - 10} more")
    
    if skipped_empty_pred:
        print(f"\nSkipped (empty prediction): {len(skipped_empty_pred)}")
        for name in skipped_empty_pred[:10]:
            print(f"  - {name}")
        if len(skipped_empty_pred) > 10:
            print(f"  ... and {len(skipped_empty_pred) - 10} more")
    
    if skipped_both_empty:
        print(f"\nSkipped (both masks empty): {len(skipped_both_empty)}")
        for name in skipped_both_empty[:10]:
            print(f"  - {name}")
        if len(skipped_both_empty) > 10:
            print(f"  ... and {len(skipped_both_empty) - 10} more")
    
    if errors:
        print(f"\nErrors: {len(errors)}")
        for name in errors[:10]:
            print(f"  - {name}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print(f"\nOutput saved to: {output_path.absolute()}")
    print(f"\nLegend:")
    print(f"  BLUE contours  = Ground Truth")
    print(f"  GREEN contours = Prediction")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
