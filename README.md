# TVUS Segmentation: AI-Powered Uterine Segmentation in 2D and 3D Transvaginal Ultrasound

This repository contains the implementation for **AI-Powered Uterine Segmentation in 2D and 3D Transvaginal Ultrasound (TVUS)**, providing deep learning models and tools for automated segmentation of uterine structures in ultrasound images.

## Overview

Transvaginal Ultrasound (TVUS) is a critical imaging modality used in gynecological examinations for diagnosing various uterine conditions. Accurate segmentation of uterine structures is essential for:

- **Clinical Diagnosis**: Identifying abnormalities, fibroids, polyps, and other uterine conditions
- **Treatment Planning**: Assisting in surgical planning and monitoring treatment effectiveness
- **Research**: Enabling quantitative analysis of uterine morphology and volume measurements
- **Automated Analysis**: Reducing manual annotation time and improving consistency

This repository implements multiple deep learning approaches for uterine segmentation:

- **Configurable Model Architectures**: MAnet, DeepLabV3Plus, Unet, and more (using `segmentation-models-pytorch`)
- **Flexible Encoders**: EfficientNet, InceptionResNetV2, and other encoders from `segmentation-models-pytorch`
- **nnUNet v2** framework for medical image segmentation
- Support for both **2D slice-based** and **3D volume** segmentation
- Multiple loss functions (Tversky Loss, Focal Loss, Soft BCE Loss)

## Features

- 🏥 **Configurable Model Architectures**: MAnet, DeepLabV3Plus, Unet, and nnUNet v2 implementations
- 📊 **Comprehensive Evaluation**: IoU, NSD (Normalized Surface Distance), and Dice coefficient metrics
- 🔄 **Cross-Validation**: GroupShuffleSplit with patient-level splitting to prevent data leakage
- 📈 **Experiment Tracking**: Integration with Neptune.ai and Sacred for experiment management
- 🎯 **Flexible Configuration**: Command-line arguments and configuration files for easy customization
- 🎬 **Video Inference**: Process video files with segmentation overlays
- 🔍 **Data Inspection**: Tools for visualizing image-segmentation pairs

## Installation

### Prerequisites

- Python 3.9+ (tested with Python 3.9.21)
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:jumutc/TVUS-segmentation.git
   cd TVUS-segmentation
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   ```bash
   export NEPTUNE_API_TOKEN="your_neptune_api_token"
   export nnUNet_raw="/path/to/nnUNet_raw"
   export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
   export nnUNet_results="/path/to/nnUNet_results"
   ```

## Project Structure

```
TVUS-segmentation/
├── Uterus_Segmentation_Exp.py              # Public dataset experiment (DeepLabV3Plus)
├── Uterus_Segmentation_Private_Exp.py      # Private dataset experiment (DeepLabV3Plus)
├── Uterus_Segmentation_Exp_nnUNet.py       # Public dataset experiment (nnUNet v2)
├── Uterus_Segmentation_Private_Exp_nnUNet.py # Private dataset experiment (nnUNet v2)
├── inference_video.py                       # Video inference script
├── inspect_pairs.py                         # Image-segmentation pair viewer
├── preprocess_image.py                      # Image preprocessing utilities
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## Usage

### Training Models

#### 1. Segmentation Model on Public Dataset (UterUS)

Train a segmentation model using the public UterUS dataset with NIfTI format:

```bash
python Uterus_Segmentation_Exp.py \
    --image_path /path/to/annotated_volumes \
    --seg_path /path/to/annotations \
    --model_output ./models/model.pt \
    --csv_output ./data/input.csv \
    --sacred_runs ./runs \
    --neptune_project jumutc/uterus \
    --dataset_name "UterUS (public)"
```

**Arguments**:
- `--image_path`: Path to directory containing NIfTI image volumes (`*.nii.gz`)
- `--seg_path`: Path to directory containing NIfTI segmentation volumes (`*.nii.gz`)
- `--model_output`: Path to save the trained model (default: `model.pt`)
- `--csv_output`: Path to save the input CSV file (default: `input.csv`)
- `--sacred_runs`: Path to Sacred runs directory (default: `uterus_runs`)
- `--neptune_project`: Neptune project name (default: `jumutc/uterus`)
- `--dataset_name`: Dataset name for logging (default: `UterUS (public)`)

**Note**: Model architecture, encoder, and parameters are configured in the script via Sacred config. See the Configuration section below for details.

#### 2. Segmentation Model on Private Dataset

Train on a private dataset with image files:

```bash
python Uterus_Segmentation_Private_Exp.py \
    --image_path /path/to/images \
    --seg_path /path/to/segmentations \
    --model_output ./models/model_tvus.pt \
    --csv_output ./data/input.csv \
    --sacred_runs ./runs \
    --neptune_project jumutc/uterus \
    --dataset_name "TVUS (private)"
```

**Note**: Model architecture, encoder, and parameters are configured in the script via Sacred config. See the Configuration section below for details.

#### 3. nnUNet v2 on Public Dataset

Train using nnUNet v2 framework:

```bash
python Uterus_Segmentation_Exp_nnUNet.py \
    --image_path /path/to/annotated_volumes \
    --seg_path /path/to/annotations \
    --dataset_id 502 \
    --dataset_name UterUSSegmentation \
    --csv_output ./data/input_nnunet.csv \
    --sacred_runs ./runs \
    --neptune_project jumutc/uterus \
    --neptune_dataset "UterUS (public)"
```

**Additional Arguments**:
- `--dataset_id`: nnUNet dataset ID (default: `502`)
- `--dataset_name`: Dataset name for nnUNet (default: `UterUSSegmentation`)

#### 4. nnUNet v2 on Private Dataset

```bash
python Uterus_Segmentation_Private_Exp_nnUNet.py \
    --image_path /path/to/images \
    --seg_path /path/to/segmentations \
    --dataset_id 501 \
    --dataset_name UterusSegmentation \
    --csv_output ./data/input_nnunet.csv \
    --sacred_runs ./runs \
    --neptune_project jumutc/uterus \
    --neptune_dataset "TVUS (private)"
```

### Video Inference

Process video files with segmentation overlays:

```bash
python inference_video.py \
    ./models/model_tvus.pt \
    /path/to/video.mp4 \
    segmented
```

**Arguments**:
- `model_path`: Path to trained model file
- `video_path_or_folder`: Path to video file or folder containing videos
- `postfix`: Optional postfix for output filename (default: `segmented`)

**Examples**:
```bash
# Single video file
python inference_video.py model_tvus.pt video1.mp4

# Process all videos in a folder
python inference_video.py model_tvus.pt /path/to/videos/

# Custom output postfix
python inference_video.py model_tvus.pt video1.mp4 segmented_output
```

### Image Preprocessing

Preprocess images by removing red-colored pixels (common in medical imaging):

```bash
# Single image
python preprocess_image.py input.jpg output.jpg 15

# Process all images in a folder recursively
python preprocess_image.py /path/to/images 15
```

**Arguments**:
- `input_path`: Input image file or directory
- `output_path`: Output image path (for single file mode)
- `window_size`: Window size for pixel replacement (default: `15`)

### Inspect Image-Segmentation Pairs

Visualize image-segmentation pairs for quality control:

```bash
# From CSV file
python inspect_pairs.py input.csv

# Single pair
python inspect_pairs.py image.jpg mask.jpg

# From folders
python inspect_pairs.py /path/to/images /path/to/masks
```

**Navigation**:
- `Left/Right` arrows or `P/N` keys: Navigate between pairs
- `Q` key: Quit

## Configuration

### Model Architecture

The model architecture is fully configurable through Sacred experiment configuration. You can modify the model settings in the experiment scripts:

**Configurable Parameters**:
- `model_name`: Model architecture (e.g., `'MAnet'`, `'DeepLabV3Plus'`, `'Unet'`, `'FPN'`, `'Linknet'`, `'PSPNet'`)
- `encoder_name`: Encoder backbone (e.g., `'efficientnet-b7'`, `'inceptionresnetv2'`, `'resnet50'`, `'resnet101'`)
- `model_params`: Dictionary of model-specific parameters:
  - `encoder_weights`: Pre-trained weights (e.g., `'imagenet'`, `'imagenet+background'`)
  - `decoder_channels`: Number of decoder channels (for DeepLabV3Plus)
  - `activation`: Activation function (e.g., `None`, `'sigmoid'`, `'softmax'`)
  - `classes`: Number of output classes (typically `1` for binary segmentation)

**Example Configurations**:

```python
# MAnet with InceptionResNetV2
'model_name': 'MAnet',
'encoder_name': 'inceptionresnetv2',
'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1}

# DeepLabV3Plus with EfficientNet-B7
'model_name': 'DeepLabV3Plus',
'encoder_name': 'efficientnet-b7',
'model_params': {'encoder_weights': 'imagenet', 'decoder_channels': 60, 'activation': None, 'classes': 1}
```

### Loss Functions

The training scripts support multiple loss functions configured via Sacred:

- `TverskyLoss`: Good for imbalanced datasets
- `FocalLoss`: Focuses on hard examples
- `SoftBCEWithLogitsLoss`: Smooth binary cross-entropy

### Data Augmentation

- **Training**: Resize, HorizontalFlip, VerticalFlip, Rotate, MotionBlur, ZoomBlur, Defocus, GaussNoise, Normalize (min_max)
- **Validation**: Resize, Normalize (min_max)
- **Normalization**: Uses `min_max` normalization (normalizes to [0, 1] range)

### Training Parameters

- **Input Size**: Configurable (default: 224x192 for public dataset, 512x768 for private dataset)
- **Batch Size**: 16 for training, 1 for validation (public dataset); 2 for training, 1 for validation (private dataset)
- **Learning Rate**: 1e-4 (max_lr for OneCycleLR)
- **Epochs**: 200
- **Weight Decay**: 1e-4
- **Early Stopping**: Stops if Dice doesn't improve for 50 epochs
- **Cross-Validation**: 3 folds with 15% validation split

## Data Format

### Public Dataset (NIfTI)

The public dataset used in this repository is the **UterUS dataset**, a publicly available dataset for uterine segmentation in transvaginal ultrasound images.

- **Dataset**: [UterUS Dataset](https://github.com/UL-FRI-LGM/UterUS)
- **Images**: NIfTI files (`.nii.gz`) containing 3D volumes
- **Segmentations**: NIfTI files (`.nii.gz`) with matching filenames
- **Structure**: Each volume is processed slice-by-slice

For more information about the dataset, including download instructions and citation, please visit the [UterUS GitHub repository](https://github.com/UL-FRI-LGM/UterUS).

### Private TVUS Dataset

- **Images**: Standard image formats (`.jpg`, `.png`, etc.)
- **Segmentations**: Grayscale masks (`.png`)
- **Structure**: Organized by volume ID with subdirectories

## Evaluation Metrics

The models are evaluated using:

- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **NSD (Normalized Surface Distance)**: Measures boundary accuracy
- **Dice Coefficient**: Measures volume overlap

Results are logged to:
- Neptune.ai for experiment tracking
- Sacred for local experiment management
- Console output with detailed metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={AI-Powered Uterine Segmentation in 2D and 3D TVUS: A Multi-Group Study},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

- **[UterUS Dataset](https://github.com/UL-FRI-LGM/UterUS)**: Public dataset for uterine segmentation in transvaginal ultrasound images
- **nnUNet framework developers**: Medical image segmentation framework
- **Segmentation Models PyTorch contributors**: PyTorch segmentation models library
