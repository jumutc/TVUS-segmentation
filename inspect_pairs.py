import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
# Try to use an interactive backend, with fallbacks
backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'GTKAgg']
backend_set = False
for backend in backends_to_try:
    try:
        matplotlib.use(backend)
        backend_set = True
        print(f"Using matplotlib backend: {backend}")
        break
    except:
        continue

if not backend_set:
    print("Warning: No interactive backend found. Trying default backend.")
    print("If you encounter issues, install tkinter: sudo apt-get install python3-tk")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ImageSegmentationViewer:
    def __init__(self, df=None, image_paths=None, seg_paths=None):
        """
        Initialize the viewer with data.
        
        Args:
            df: DataFrame with columns ['img_path', 'seg_path'] (optional)
            image_paths: List of image paths (optional)
            seg_paths: List of segmentation paths (optional)
        """
        self.current_idx = 0
        
        # Load data from DataFrame or paths
        if df is not None:
            self.image_paths = df['img_path'].tolist()
            self.seg_paths = df['seg_path'].tolist()
            if 'volume_id' in df.columns:
                self.volume_ids = df['volume_id'].tolist()
            else:
                self.volume_ids = [None] * len(self.image_paths)
        elif image_paths is not None and seg_paths is not None:
            self.image_paths = image_paths
            self.seg_paths = seg_paths
            self.volume_ids = [None] * len(self.image_paths)
        else:
            raise ValueError("Must provide either df or both image_paths and seg_paths")
        
        if len(self.image_paths) != len(self.seg_paths):
            raise ValueError(f"Mismatch: {len(self.image_paths)} images vs {len(self.seg_paths)} segmentations")
        
        print(f"Loaded {len(self.image_paths)} image-segmentation pairs")
        
        # Setup matplotlib figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.fig.suptitle('Image-Segmentation Pair Viewer', fontsize=16)
        
        # Adjust layout to leave space for buttons at the bottom
        plt.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95, wspace=0.2)
        
        # Create navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
        ax_next = plt.axes([0.25, 0.02, 0.1, 0.04])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        
        # Display first image
        self.update_display()
        
        # Keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def load_pair(self, idx):
        """Load image and segmentation pair at given index"""
        if idx < 0 or idx >= len(self.image_paths):
            return None, None, None
        
        img_path = self.image_paths[idx]
        seg_path = self.seg_paths[idx]
        volume_id = self.volume_ids[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            return None, None, None
        
        # # Convert BGR to RGB for matplotlib
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load segmentation mask
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if seg is None:
            print(f"Warning: Could not load segmentation: {seg_path}")
            return None, None, None
        
        return img, seg, volume_id
    
    def create_overlay(self, img, seg):
        """Create overlay of image and segmentation mask"""
        overlay = img.copy()
        
        # Create colored mask (green for segmentation)
        mask_colored = np.zeros_like(img)
        mask_colored[seg > 0] = [0, 255, 0]  # Green
        
        # Blend overlay
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)
        
        return overlay
    
    def update_display(self):
        """Update the display with current image pair"""
        img, seg, volume_id = self.load_pair(self.current_idx)
        
        if img is None or seg is None:
            print(f"Error loading pair at index {self.current_idx}")
            return
        
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Display original image
        self.axes[0].imshow(img)
        self.axes[0].set_title(f'Original Image\n{Path(self.image_paths[self.current_idx]).name}')
        self.axes[0].axis('off')
        
        # Display segmentation mask
        self.axes[1].imshow(seg, cmap='gray')
        self.axes[1].set_title(f'Segmentation Mask\n{Path(self.seg_paths[self.current_idx]).name}')
        self.axes[1].axis('off')
        
        # Display overlay
        overlay = self.create_overlay(img, seg)
        self.axes[2].imshow(overlay)
        title = f'Overlay\nIndex: {self.current_idx + 1}/{len(self.image_paths)}'
        if volume_id:
            title += f'\nVolume ID: {volume_id}'
        self.axes[2].set_title(title)
        self.axes[2].axis('off')
        
        # Print info
        print(f"\n[{self.current_idx + 1}/{len(self.image_paths)}] Volume: {volume_id}")
        print(f"  Image: {self.image_paths[self.current_idx]}")
        print(f"  Segmentation: {self.seg_paths[self.current_idx]}")
        print(f"  Image shape: {img.shape}, Mask shape: {seg.shape}")
        print(f"  Mask pixels: {np.sum(seg > 0)} ({100 * np.sum(seg > 0) / seg.size:.2f}%)")
        
        self.fig.canvas.draw()
    
    def next_image(self, event=None):
        """Move to next image"""
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.update_display()
    
    def prev_image(self, event=None):
        """Move to previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def on_key(self, event):
        """Handle keyboard navigation"""
        if event.key == 'right' or event.key == 'n':
            self.next_image()
        elif event.key == 'left' or event.key == 'p':
            self.prev_image()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def show(self):
        """Show the viewer"""
        # Layout already adjusted in __init__, no need for tight_layout
        plt.show()


def load_from_csv(csv_path):
    """Load image-segmentation pairs from CSV file"""
    df = pd.read_csv(csv_path)
    
    if 'img_path' not in df.columns or 'seg_path' not in df.columns:
        raise ValueError("CSV must contain 'img_path' and 'seg_path' columns")
    
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  From CSV: python inspect_pairs.py <csv_file>")
        print("  From paths: python inspect_pairs.py <image_path> <seg_path>")
        print("  From folder: python inspect_pairs.py <image_folder> <seg_folder>")
        print("\nExamples:")
        print("  python inspect_pairs.py input.csv")
        print("  python inspect_pairs.py image.jpg mask.jpg")
        print("  python inspect_pairs.py /path/to/images /path/to/masks")
        print("\nNavigation:")
        print("  Left/Right arrows or P/N keys: Navigate")
        print("  Q key: Quit")
        sys.exit(1)
    
    input_path = sys.argv[1]
    path_obj = Path(input_path)
    
    # Check if it's a CSV file
    if path_obj.suffix.lower() == '.csv':
        print(f"Loading pairs from CSV: {input_path}")
        df = load_from_csv(input_path)
        viewer = ImageSegmentationViewer(df=df)
        viewer.show()
    
    # Check if it's a single image file
    elif path_obj.is_file() and len(sys.argv) >= 3:
        image_path = sys.argv[1]
        seg_path = sys.argv[2]
        print(f"Loading single pair:")
        print(f"  Image: {image_path}")
        print(f"  Segmentation: {seg_path}")
        viewer = ImageSegmentationViewer(image_paths=[image_path], seg_paths=[seg_path])
        viewer.show()
    
    # Check if it's a folder
    elif path_obj.is_dir() and len(sys.argv) >= 3:
        image_folder = Path(sys.argv[1])
        seg_folder = Path(sys.argv[2])
        
        # Find matching image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        seg_paths = []
        
        for ext in image_extensions:
            image_paths.extend(image_folder.glob(f'*{ext}'))
            image_paths.extend(image_folder.glob(f'*{ext.upper()}'))
        
        image_paths = sorted([str(p) for p in image_paths])
        
        # Try to find corresponding segmentation files
        for img_path in image_paths:
            img_name = Path(img_path).stem
            # Try different naming patterns
            seg_candidates = [
                seg_folder / f"masked_{img_name}.png",
                seg_folder / f"{img_name}_mask.png",
                seg_folder / f"{img_name}.png",
                seg_folder / f"mask_{img_name}.png",
            ]
            
            found = False
            for seg_candidate in seg_candidates:
                if seg_candidate.exists():
                    seg_paths.append(str(seg_candidate))
                    found = True
                    break
            
            if not found:
                print(f"Warning: No segmentation found for {img_path}")
        
        if len(image_paths) != len(seg_paths):
            print(f"Warning: Found {len(image_paths)} images but {len(seg_paths)} segmentations")
        
        if len(seg_paths) == 0:
            print("Error: No matching segmentation files found")
            sys.exit(1)
        
        print(f"Found {len(seg_paths)} pairs")
        viewer = ImageSegmentationViewer(image_paths=image_paths, seg_paths=seg_paths)
        viewer.show()
    
    else:
        print(f"Error: Invalid input. {input_path} is not a valid CSV file, image file, or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
