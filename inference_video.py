import sys
import cv2
import numpy as np
import torch
import albumentations as A
from torchvision import transforms as T
from tqdm import tqdm
from pathlib import Path
from skimage import measure
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model input size (from Uterus_Segmentation_Private_Exp.py)
height, width = 512, 768

def preprocess_frame(frame):
    """Preprocess a single frame for model inference"""
    # Resize to model input size
    transform = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_LINEAR)], is_check_shapes=False)
    transformed = transform(image=frame)
    img = transformed['image']
    
    # Convert to tensor
    t = T.Compose([T.ToTensor()])
    img_tensor = t(img).float()
    
    return img_tensor, frame.shape[:2]  # Return tensor and original shape

def postprocess_mask(mask_logits, original_shape):
    """Convert model output to binary mask and resize to original frame size"""
    
    # Convert to binary mask (threshold at 0.5)
    mask_binary = (torch.sigmoid(mask_logits) > 0.5).cpu().numpy()
    
    # Remove batch and channel dimensions
    if len(mask_binary.shape) == 4:
        mask_binary = mask_binary[0, 0]  # [batch, channel, H, W] -> [H, W]
    elif len(mask_binary.shape) == 3:
        mask_binary = mask_binary[0]  # [channel, H, W] -> [H, W]
    
    # Resize mask back to original frame size
    mask_resized = cv2.resize(
        mask_binary.astype(np.uint8) * 255,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Filter to keep only the largest detected region
    mask_resized = keep_largest_region(mask_resized)
    
    return mask_resized

def keep_largest_region(mask):
    """Keep only the largest connected region in the mask using skimage regionprops"""
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

def draw_contours(frame, mask):
    """Draw contours of the segmentation mask on the frame"""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on frame
    frame_with_contours = frame.copy()
    cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 4)  # Green contours, 4px thickness
    
    return frame_with_contours

def process_video(model, video_path, output_path):
    """Process a single video file"""
    print(f"\nProcessing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    model.eval()
    frame_count = 0
    
    # Lists to store timing statistics per frame
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    
    with torch.no_grad():
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                preprocess_start = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_tensor, original_shape = preprocess_frame(frame)
                img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
                preprocessing_time = time.time() - preprocess_start
                preprocessing_times.append(preprocessing_time)
                
                # Run inference
                inference_start = time.time()
                output = model(img_tensor)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                
                # Postprocess mask
                postprocess_start = time.time()
                mask = postprocess_mask(output, original_shape)
                
                # Draw contours on original frame
                frame_with_contours = draw_contours(frame, mask)
                postprocessing_time = time.time() - postprocess_start
                postprocessing_times.append(postprocessing_time)
                
                # Write frame to output video
                out.write(frame_with_contours)
                
                frame_count += 1
                pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Saved output video to: {output_path}")
    print(f"Processed {frame_count} frames")
    
    # Return timing statistics for this video
    return {
        'preprocessing_times': preprocessing_times,
        'inference_times': inference_times,
        'postprocessing_times': postprocessing_times,
        'frame_count': frame_count
    }

def get_video_files(input_path):
    """Get list of video files from a file or folder"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    video_files = []
    
    path = Path(input_path)
    if path.is_file():
        # Single file
        if path.suffix.lower() in video_extensions:
            video_files.append(str(path))
        else:
            print(f"Warning: {input_path} is not a recognized video file")
    elif path.is_dir():
        # Folder - find all video files
        for ext in video_extensions:
            video_files.extend(path.glob(f'*{ext}'))
            video_files.extend(path.glob(f'*{ext.upper()}'))
        video_files = [str(f) for f in video_files]
        video_files.sort()
        print(f"Found {len(video_files)} video file(s) in folder: {input_path}")
    else:
        print(f"Error: {input_path} is not a valid file or folder")
    
    return video_files

def extract_model_name(model_path):
    """Extract model name from model path"""
    model_name = Path(model_path).stem  # Get filename without extension
    return model_name

def print_statistics(all_preprocessing_times, all_inference_times, all_postprocessing_times, total_frames, total_videos):
    """Print statistics over all frames and videos"""
    print("\n" + "="*80)
    print("TIMING STATISTICS")
    print("="*80)
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames processed: {total_frames}")
    print("\nPer Frame Statistics:")
    print("-"*80)
    
    # Preprocessing statistics
    if all_preprocessing_times:
        preprocess_mean = np.mean(all_preprocessing_times)
        preprocess_std = np.std(all_preprocessing_times)
        preprocess_min = np.min(all_preprocessing_times)
        preprocess_max = np.max(all_preprocessing_times)
        print(f"Preprocessing:")
        print(f"  Mean: {preprocess_mean*1000:.3f} ms")
        print(f"  Std:  {preprocess_std*1000:.3f} ms")
        print(f"  Min:  {preprocess_min*1000:.3f} ms")
        print(f"  Max:  {preprocess_max*1000:.3f} ms")
        print(f"  Total: {np.sum(all_preprocessing_times):.3f} s")
    
    # Inference statistics
    if all_inference_times:
        inference_mean = np.mean(all_inference_times)
        inference_std = np.std(all_inference_times)
        inference_min = np.min(all_inference_times)
        inference_max = np.max(all_inference_times)
        print(f"\nModel Inference:")
        print(f"  Mean: {inference_mean*1000:.3f} ms")
        print(f"  Std:  {inference_std*1000:.3f} ms")
        print(f"  Min:  {inference_min*1000:.3f} ms")
        print(f"  Max:  {inference_max*1000:.3f} ms")
        print(f"  Total: {np.sum(all_inference_times):.3f} s")
    
    # Postprocessing statistics
    if all_postprocessing_times:
        postprocess_mean = np.mean(all_postprocessing_times)
        postprocess_std = np.std(all_postprocessing_times)
        postprocess_min = np.min(all_postprocessing_times)
        postprocess_max = np.max(all_postprocessing_times)
        print(f"\nPostprocessing:")
        print(f"  Mean: {postprocess_mean*1000:.3f} ms")
        print(f"  Std:  {postprocess_std*1000:.3f} ms")
        print(f"  Min:  {postprocess_min*1000:.3f} ms")
        print(f"  Max:  {postprocess_max*1000:.3f} ms")
        print(f"  Total: {np.sum(all_postprocessing_times):.3f} s")
    
    # Total per frame statistics
    if all_preprocessing_times and all_inference_times and all_postprocessing_times:
        total_times = [p + i + post for p, i, post in zip(all_preprocessing_times, all_inference_times, all_postprocessing_times)]
        total_mean = np.mean(total_times)
        total_std = np.std(total_times)
        total_min = np.min(total_times)
        total_max = np.max(total_times)
        print(f"\nTotal per Frame (Preprocessing + Inference + Postprocessing):")
        print(f"  Mean: {total_mean*1000:.3f} ms")
        print(f"  Std:  {total_std*1000:.3f} ms")
        print(f"  Min:  {total_min*1000:.3f} ms")
        print(f"  Max:  {total_max*1000:.3f} ms")
        print(f"  Total: {np.sum(total_times):.3f} s")
    
    print("="*80)

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_video.py <model_path> <video_path_or_folder> [postfix]")
        print("Example: python inference_video.py model_tvus.pt video1.mp4")
        print("Example: python inference_video.py model_tvus.pt /path/to/videos/")
        print("Example: python inference_video.py model_tvus.pt video1.mp4 segmented")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    postfix = sys.argv[3] if len(sys.argv) > 3 else "segmented"
    
    # Extract model name
    model_name = extract_model_name(model_path)
    print(f"Model name: {model_name}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Get video files (from file or folder)
    video_paths = get_video_files(input_path)
    
    if not video_paths:
        print("No video files found to process")
        sys.exit(1)
    
    # Aggregate timing statistics across all videos
    all_preprocessing_times = []
    all_inference_times = []
    all_postprocessing_times = []
    total_frames = 0
    total_videos = 0
    
    # Process each video
    for video_path in video_paths:
        # Create output filename: {original_name}_{model_name}_{postfix}.mp4
        video_path_obj = Path(video_path)
        base_name = video_path_obj.stem
        output_dir = video_path_obj.parent
        output_filename = f"{base_name}_{model_name}_{postfix}{video_path_obj.suffix}"
        output_path = str(output_dir / output_filename)
        
        stats = process_video(model, video_path, output_path)
        
        if stats is not None:
            # Aggregate statistics
            all_preprocessing_times.extend(stats['preprocessing_times'])
            all_inference_times.extend(stats['inference_times'])
            all_postprocessing_times.extend(stats['postprocessing_times'])
            total_frames += stats['frame_count']
            total_videos += 1
    
    # Print aggregated statistics
    print_statistics(all_preprocessing_times, all_inference_times, all_postprocessing_times, total_frames, total_videos)
    
    print("\nAll videos processed successfully!")

if __name__ == "__main__":
    main()
