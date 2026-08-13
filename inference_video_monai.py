"""
Video inference for MONAI uterus/niche segmentation models.

Mirrors inference_video.py, but preprocesses frames with the same MONAI
transforms used at validation time in Uterus_Segmentation_Private_Exp_niches_MONAI.py
and handles MONAI-specific model outputs:

- BasicUNetPlusPlus returns a list of tensors; the first head is used
- BoundaryWeightedTverskyLoss models emit 2 channels; channel 0 is the mask,
  channel 1 is the unused boundary-distance map

Usage:
    python inference_video_monai.py <model_path> <video_path_or_folder> [output_dir]
"""

import os.path
import sys
import time
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


def draw_contours(frame, mask):
    """Draw contours of the segmentation mask on the frame."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_with_contours = frame.copy()
    cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 4)
    return frame_with_contours


def process_video(model, video_path, output_path):
    """Process a single video file."""
    print(f"\nProcessing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    model.eval()
    frame_count = 0
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []

    with torch.no_grad():
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                preprocess_start = time.time()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_tensor, original_shape = preprocess_frame(frame_rgb)
                img_tensor = img_tensor.unsqueeze(0).to(device)
                preprocessing_times.append(time.time() - preprocess_start)

                inference_start = time.time()
                output = model(img_tensor)
                inference_times.append(time.time() - inference_start)

                postprocess_start = time.time()
                mask = postprocess_mask(output, original_shape)
                frame_with_contours = draw_contours(frame_bgr, mask)
                postprocessing_times.append(time.time() - postprocess_start)

                out.write(frame_with_contours)

                frame_count += 1
                pbar.update(1)

    cap.release()
    out.release()
    print(f"Saved output video to: {output_path}")
    print(f"Processed {frame_count} frames")

    return {
        "preprocessing_times": preprocessing_times,
        "inference_times": inference_times,
        "postprocessing_times": postprocessing_times,
        "frame_count": frame_count,
    }


def get_video_files(input_path):
    """Get list of video files from a file or folder."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}
    video_files = []

    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in video_extensions:
            video_files.append(str(path))
        else:
            print(f"Warning: {input_path} is not a recognized video file")
    elif path.is_dir():
        for ext in video_extensions:
            video_files.extend(path.glob(f"*{ext}"))
            video_files.extend(path.glob(f"*{ext.upper()}"))
        video_files = [str(f) for f in video_files]
        video_files.sort()
        print(f"Found {len(video_files)} video file(s) in folder: {input_path}")
    else:
        print(f"Error: {input_path} is not a valid file or folder")

    return video_files


def extract_model_name(model_path):
    """Extract model name from model path."""
    return Path(model_path).stem


def print_statistics(all_preprocessing_times, all_inference_times, all_postprocessing_times, total_frames, total_videos):
    """Print statistics over all frames and videos."""
    print("\n" + "=" * 80)
    print("TIMING STATISTICS")
    print("=" * 80)
    print(f"Total videos processed: {total_videos}")
    print(f"Total frames processed: {total_frames}")
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

    print("=" * 80)


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_video_monai.py <model_path> <video_path_or_folder> [output_dir]")
        print("Example: python inference_video_monai.py model_tvus.pt video1.mp4")
        print("Example: python inference_video_monai.py model_tvus.pt /path/to/videos/")
        print("Example: python inference_video_monai.py model_tvus.pt video1.mp4 segmented")
        sys.exit(1)

    model_path = sys.argv[1]
    input_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "segmented"

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

    video_paths = get_video_files(input_path)
    if not video_paths:
        print("No video files found to process")
        sys.exit(1)

    all_preprocessing_times = []
    all_inference_times = []
    all_postprocessing_times = []
    total_frames = 0
    total_videos = 0

    for video_path in video_paths:
        video_path_obj = Path(video_path)
        output_filename = f"{video_path_obj.stem}_{model_name}{video_path_obj.suffix}"
        output_path = f"{output_dir}/{output_filename}"

        stats = process_video(model, video_path, output_path)
        if stats is not None:
            all_preprocessing_times.extend(stats["preprocessing_times"])
            all_inference_times.extend(stats["inference_times"])
            all_postprocessing_times.extend(stats["postprocessing_times"])
            total_frames += stats["frame_count"]
            total_videos += 1

    print_statistics(
        all_preprocessing_times,
        all_inference_times,
        all_postprocessing_times,
        total_frames,
        total_videos,
    )
    print("\nAll videos processed successfully!")


if __name__ == "__main__":
    main()
