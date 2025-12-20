import sys
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm


def replace_red_channel_pixels(img, window_size=15, verbose=True):
    """
    Replace red-colored pixels with the average of the 10 nearest pixels.
    Uses HSV color space to detect red pixels more accurately.
    
    Args:
        img: Input image (H, W, 3) numpy array in BGR format (OpenCV default)
        window_size: Size of the local window to search for nearest pixels
        verbose: Whether to print progress and statistics
    
    Returns:
        Processed image with red pixels replaced
    """
    processed_img = img.copy()
    h, w = img.shape[:2]
    
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
    
    # Get coordinates of pixels to replace
    red_pixel_coords = np.argwhere(red_mask > 0)
    
    if verbose:
        print(f"Image size: {h}x{w}")
        print(f"Red pixels detected (strict hue): {len(red_pixel_coords)}")
        print(f"Pixels to keep: {h * w - len(red_pixel_coords)}")
    
    if len(red_pixel_coords) == 0:
        if verbose:
            print("No pixels with red channel on to replace!")
        return processed_img
    
    # Process each pixel with red channel on
    # Use 100x100 window to sample boundary pixels
    # 100x100 window = 50 pixels radius from center
    boundary_distances = [50]  # Single 100x100 window
    
    iterator = tqdm(red_pixel_coords, desc="Processing pixels") if verbose else red_pixel_coords
    for y, x in iterator:
        boundary_pixels_list = []
        
        # Sample boundary pixels from windows at different distances
        for dist in boundary_distances:
            # Define window with boundary at this distance
            y_min = max(0, y - dist)
            y_max = min(h, y + dist + 1)
            x_min = max(0, x - dist)
            x_max = min(w, x + dist + 1)
            
            # Extract local window
            local_window = img[y_min:y_max, x_min:x_max]
            win_h, win_w = local_window.shape[:2]
            
            if win_h < 3 or win_w < 3:
                continue
            
            # Extract boundary pixels (edge of the window)
            # Top and bottom rows
            boundary_pixels_list.extend(local_window[0, :].reshape(-1, 3))
            boundary_pixels_list.extend(local_window[-1, :].reshape(-1, 3))
            
            # Left and right columns (excluding corners already counted)
            if win_h > 2:
                boundary_pixels_list.extend(local_window[1:-1, 0].reshape(-1, 3))
                boundary_pixels_list.extend(local_window[1:-1, -1].reshape(-1, 3))
        
        if len(boundary_pixels_list) == 0:
            continue
        
        # Convert to numpy array and calculate median
        boundary_pixels = np.array(boundary_pixels_list)
        median_pixel = np.median(boundary_pixels, axis=0).astype(np.uint8)
        
        # Replace the pixel
        processed_img[y, x] = median_pixel
    
    return processed_img


def process_single_image(input_path, output_path, window_size, verbose=True):
    """Process a single image file"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"{'='*60}")
    
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not load image from {input_path}")
        return False
    
    if verbose:
        print(f"Image loaded: {img.shape}")
    
    # Apply preprocessing
    if verbose:
        print(f"Applying preprocessing with window_size={window_size}...")
    processed_img = replace_red_channel_pixels(img, window_size=window_size, verbose=verbose)
    
    # Save output
    if verbose:
        print(f"Saving processed image to: {output_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    success = cv2.imwrite(output_path, processed_img)
    if not success:
        print(f"Error: Could not save image to {output_path}")
        return False
    
    if verbose:
        print("Done!")
    return True


def find_image_files(directory):
    """Recursively find all image files in a directory, excluding files starting with 'masked_'"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF', '*.TIF']
    image_files = []
    
    for ext in image_extensions:
        # Search recursively
        pattern = os.path.join(directory, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out files starting with 'masked_'
    filtered_files = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        if not filename.startswith('masked_'):
            filtered_files.append(img_path)
    
    return sorted(filtered_files)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single image: python preprocess_image.py <input_image> [output_image] [window_size]")
        print("  Folder (recursive): python preprocess_image.py <input_folder> [window_size]")
        print("\nExamples:")
        print("  python preprocess_image.py input.jpg output.jpg 15")
        print("  python preprocess_image.py ./images 15")
        print("\nNote: For folders, output files are saved in the same folders with '_preprocessed' postfix")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Optional output path (only used for single file mode)
    output_path = sys.argv[2] if len(sys.argv) >= 3 and os.path.isfile(input_path) else None
    
    # Optional window size (adjust index based on whether output_path was provided)
    if os.path.isfile(input_path):
        window_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 15
    else:
        window_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 15
    
    # Check if input is a file or directory
    if os.path.isfile(input_path):
        # Single file mode
        if output_path is None:
            # Generate output filename
            base_name = input_path.rsplit('.', 1)[0]
            ext = input_path.rsplit('.', 1)[1] if '.' in input_path else 'jpg'
            output_path = f"{base_name}_preprocessed.{ext}"
        
        process_single_image(input_path, output_path, window_size, verbose=True)
        
    elif os.path.isdir(input_path):
        # Directory mode - process all images recursively
        print(f"Scanning directory recursively: {input_path}")
        image_files = find_image_files(input_path)
        
        if len(image_files) == 0:
            print(f"No image files found in {input_path} (excluding files starting with 'masked_')")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image file(s) (excluding files starting with 'masked_')")
        print(f"Output files will be saved in the same folders with '_preprocessed' postfix")
        
        # Process each image
        successful = 0
        failed = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # Generate output path in the same folder with _preprocessed postfix
            directory = os.path.dirname(img_path)
            filename = os.path.basename(img_path)
            
            # Split filename and extension
            if '.' in filename:
                base_name, ext = filename.rsplit('.', 1)
                out_filename = f"{base_name}_preprocessed.{ext}"
            else:
                out_filename = f"{filename}_preprocessed"
            
            out_img_path = os.path.join(directory, out_filename)
            
            # Process with minimal verbosity for batch processing
            if process_single_image(img_path, out_img_path, window_size, verbose=False):
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Output files saved in the same folders with '_preprocessed' postfix")
        print(f"{'='*60}")
        
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
