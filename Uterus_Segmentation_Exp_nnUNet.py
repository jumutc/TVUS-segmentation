"""
Uterus Segmentation Experiment using nnUNet v2

This script reproduces Uterus_Segmentation_Exp.py but uses nnUNet v2 
instead of segmentation_models_pytorch (DeepLabV3Plus).

Key differences:
- Uses nnUNet v2 framework instead of smp.DeepLabV3Plus
- Converts data from NIfTI format to nnUNet format (imagesTr/labelsTr structure)
- Uses nnUNet's training API (run_training)
- Preserves the same GroupShuffleSplit train/val splitting logic
- Maintains Sacred and Neptune logging
- Computes same metrics (IoU, NSD, Dice) using MeshMetrics

Requirements:
- nnunetv2: pip install nnunetv2
- Set environment variables: nnUNet_raw, nnUNet_preprocessed, nnUNet_results
  (or they will be created in current directory)
"""

import argparse
import glob
import json
import os
import shutil
import ssl
import sys

# Set nnUNet paths BEFORE importing nnunetv2.paths (it reads env vars at import time)
# These can be overridden via environment variables or command-line arguments
if 'nnUNet_raw' not in os.environ:
    os.environ['nnUNet_raw'] = os.path.join(os.getcwd(), 'nnUNet_raw')
if 'nnUNet_preprocessed' not in os.environ:
    os.environ['nnUNet_preprocessed'] = os.path.join(os.getcwd(), 'nnUNet_preprocessed')
if 'nnUNet_results' not in os.environ:
    os.environ['nnUNet_results'] = os.path.join(os.getcwd(), 'nnUNet_results')

# Create directories
os.makedirs(os.environ['nnUNet_raw'], exist_ok=True)
os.makedirs(os.environ['nnUNet_preprocessed'], exist_ok=True)
os.makedirs(os.environ['nnUNet_results'], exist_ok=True)

import cv2
import neptune
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from MeshMetrics import DistanceMetrics

try:
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.run.run_training import run_training
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    
    # Fix PolyLRScheduler for PyTorch version compatibility
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    import torch.optim.lr_scheduler as lr_scheduler
    
    # Monkey patch PolyLRScheduler to fix __init__ signature issue
    original_init = PolyLRScheduler.__init__
    def fixed_init(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        # Only pass optimizer and last_epoch to avoid signature mismatch
        last_epoch = current_step if current_step is not None else -1
        lr_scheduler._LRScheduler.__init__(self, optimizer, last_epoch=last_epoch)
    
    PolyLRScheduler.__init__ = fixed_init
except ImportError as e:
    print(f"Warning: nnUNet imports failed: {e}")
    print("Please install nnUNet: pip install nnunetv2")
    raise
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ex = Experiment("uterus_exp_nnunet")

metrics = DistanceMetrics()

def parse_args():
    parser = argparse.ArgumentParser(description='Uterus Segmentation Experiment using nnUNet v2')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to directory containing NIfTI image volumes')
    parser.add_argument('--seg_path', type=str, required=True,
                        help='Path to directory containing NIfTI segmentation volumes')
    parser.add_argument('--dataset_id', type=int, default=502,
                        help='nnUNet dataset ID (default: 502)')
    parser.add_argument('--dataset_name', type=str, default='UterUSSegmentation',
                        help='Dataset name for nnUNet (default: UterUSSegmentation)')
    parser.add_argument('--csv_output', type=str, default='input_nnunet.csv',
                        help='Path to save the input CSV file (default: input_nnunet.csv)')
    parser.add_argument('--sacred_runs', type=str, default='uterus_runs',
                        help='Path to Sacred runs directory (default: uterus_runs)')
    parser.add_argument('--neptune_project', type=str, default='jumutc/uterus',
                        help='Neptune project name (default: jumutc/uterus)')
    parser.add_argument('--neptune_dataset', type=str, default='UterUS (public)',
                        help='Dataset name for Neptune logging (default: UterUS (public))')
    return parser.parse_args()


def create_df(image_path, seg_path):
    """Create dataframe with paths to NIfTI files and slice indices."""
    image_paths = sorted(glob.glob(f"{image_path}/*.nii.gz"))
    volume_ids, img_paths, seg_paths, slice_indices = [], [], [], []

    for img_path in tqdm(image_paths):
        volume_id = os.path.basename(img_path)
        seg_path_full = os.path.join(seg_path, volume_id)
        
        if not os.path.exists(seg_path_full):
            continue
            
        nii_img = nib.load(img_path)
        nii_img_data = nii_img.get_fdata()
        
        # Store paths and slice indices instead of loading all data
        for i in range(nii_img_data.shape[-1]):
            volume_ids.append(volume_id)
            img_paths.append(img_path)
            seg_paths.append(seg_path_full)
            slice_indices.append(i)
        
        if not os.path.exists(seg_path):
            continue
            
        nii_img = nib.load(image_path)
        nii_img_data = nii_img.get_fdata()
        
        # Store paths and slice indices instead of loading all data
        for i in range(nii_img_data.shape[-1]):
            volume_ids.append(volume_id)
            img_paths.append(image_path)
            seg_paths.append(seg_path)
            slice_indices.append(i)

    return pd.DataFrame({
        'volume_id': volume_ids,
        'img_path': img_paths,
        'seg_path': seg_paths,
        'slice_idx': slice_indices
    }, index=np.arange(0, len(volume_ids)))


def convert_to_nnunet_format(df, dataset_id, dataset_name, train_indices, val_indices):
    """
    Convert images from NIfTI format to nnUNet format and create dataset structure.
    nnUNet expects all images in imagesTr/labelsTr, validation is determined by splits.json.
    """
    dataset_name_full = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_path = os.path.join(os.environ['nnUNet_raw'], dataset_name_full)
    imagesTr_path = os.path.join(dataset_path, 'imagesTr')
    labelsTr_path = os.path.join(dataset_path, 'labelsTr')
    
    # Clean up if exists
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    
    os.makedirs(imagesTr_path, exist_ok=True)
    os.makedirs(labelsTr_path, exist_ok=True)
    
    train_cases = []
    val_cases = []
    
    # Convert all images (both train and val go into imagesTr/labelsTr)
    print("Converting images to nnUNet format...")
    all_indices = list(train_indices) + list(val_indices)
    
    for idx in tqdm(all_indices):
        img_path = df.loc[idx]['img_path']
        seg_path = df.loc[idx]['seg_path']
        volume_id = df.loc[idx]['volume_id']
        slice_idx = df.loc[idx]['slice_idx']
        
        # Load NIfTI files
        nii_img = nib.load(img_path)
        nii_seg = nib.load(seg_path)
        nii_img_data = nii_img.get_fdata()
        nii_seg_data = nii_seg.get_fdata()
        
        # Extract slice
        img_slice = nii_img_data[..., slice_idx]
        seg_slice = nii_seg_data[..., slice_idx]
        
        # Normalize image to 0-255 range
        img_slice = np.squeeze(img_slice)
        seg_slice = np.squeeze(seg_slice)
        
        # Ensure image and mask have the same dimensions
        # Resize mask to match image dimensions if they differ
        if img_slice.shape != seg_slice.shape:
            seg_slice = cv2.resize(seg_slice.astype(np.float32), (img_slice.shape[1], img_slice.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Keep as single-channel grayscale (nnUNet expects 1 channel for 2D)
        if len(img_slice.shape) == 2:
            img_slice = img_slice  # Already 2D, keep as is
        elif len(img_slice.shape) == 3:
            # If somehow 3D, take first channel to convert to grayscale
            img_slice = img_slice[:, :, 0]
        assert len(img_slice.shape) == 2, f"Expected 2D grayscale image, got shape {img_slice.shape}"
        
        # Binarize segmentation mask to exactly 0 and 1 (nnUNet requires labels to be 0, 1, 2, ...)
        # Do NOT multiply by 255 - nnUNet expects class indices, not pixel intensities
        seg_slice = (seg_slice > 0).astype(np.uint8)  # Values: 0 or 1
        
        # Verify dimensions match after all conversions
        assert img_slice.shape == seg_slice.shape, f"Image and mask dimensions still don't match: img={img_slice.shape}, mask={seg_slice.shape}"
        
        # Verify mask only contains 0 and 1
        unique_mask_values = np.unique(seg_slice)
        assert np.all(np.isin(unique_mask_values, [0, 1])), f"Mask contains invalid values: {unique_mask_values}. Expected only 0 and 1."
        
        # Create unique identifier
        unique_id = f"{volume_id.replace('.nii.gz', '')}_{slice_idx:05d}"
        
        # Save single-channel grayscale image (nnUNet expects separate channel files)
        # Channel 0 (grayscale)
        img_filename_0 = f"{unique_id}_0000.png"
        cv2.imwrite(os.path.join(imagesTr_path, img_filename_0), img_slice)
        
        # Save mask as uint8 with values exactly 0 and 1 (class indices, not pixel intensities)
        mask_filename = f"{unique_id}.png"
        cv2.imwrite(os.path.join(labelsTr_path, mask_filename), seg_slice)
        
        # Track which cases are train vs val
        if idx in train_indices:
            train_cases.append(unique_id)
        else:
            val_cases.append(unique_id)
    
    return dataset_path, imagesTr_path, labelsTr_path, train_cases, val_cases


def create_dataset_json(dataset_path, dataset_name, num_total_cases):
    """Create dataset.json file for nnUNet."""
    dataset_json = {
        "channel_names": {
            "0": "grayscale"
        },
        "labels": {
            "background": 0,
            "uterus": 1
        },
        "numTraining": num_total_cases,
        "numTest": 0,
        "file_ending": ".png",
        "name": dataset_name,
        "description": "UterUS segmentation dataset",
        "reference": "UterUS public dataset",
        "licence": "Public",
        "release": "1.0",
        "tensorImageSize": "2D",
        "modality": {
            "0": "grayscale"
        },
        "dimension": 2
    }
    
    json_path = os.path.join(dataset_path, 'dataset.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    return json_path


def create_splits_json(dataset_id, dataset_name, train_cases, val_cases):
    """Create splits.json file for nnUNet with custom train/val split."""
    # Splits are created in preprocessed folder after preprocessing
    preprocessed_path = os.path.join(os.environ['nnUNet_preprocessed'], f"Dataset{dataset_id:03d}_{dataset_name}")
    os.makedirs(preprocessed_path, exist_ok=True)
    splits_path = os.path.join(preprocessed_path, 'splits_final.json')
    
    splits = [
        {
            "train": train_cases,
            "val": val_cases
        }
    ]
    
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=4)
    
    return splits_path


def evaluate_predictions(dataset_id, dataset_name, val_cases, df, val_indices):
    """Evaluate nnUNet predictions and compute metrics."""
    results_path = os.path.join(
        os.environ['nnUNet_results'],
        f"Dataset{dataset_id:03d}_{dataset_name}",
        'nnUNetTrainer__nnUNetPlans__2d',
        'fold_0',
        'validation'
    )
    
    val_iou_scores = []
    val_nsd_scores = []
    val_dice_scores = []
    
    if not os.path.exists(results_path):
        print(f"Warning: Results path {results_path} does not exist. Skipping evaluation.")
        return 0.0, 0.0, 0.0
    
    # Load predictions
    for case_id, idx in zip(val_cases, val_indices):
        pred_path = os.path.join(results_path, f"{case_id}.png")
        if not os.path.exists(pred_path):
            continue
        
        # Load ground truth from original NIfTI
        img_path = df.loc[idx]['img_path']
        seg_path = df.loc[idx]['seg_path']
        slice_idx = df.loc[idx]['slice_idx']
        
        nii_seg = nib.load(seg_path)
        nii_seg_data = nii_seg.get_fdata()
        gt_mask = np.squeeze(nii_seg_data[..., slice_idx])
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # Load prediction
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            continue
        
        # Resize prediction to match GT if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize masks
        gt_binary = (gt_mask > 0).astype(bool)
        pred_binary = (pred_mask > 0).astype(bool)
        
        # Compute metrics
        metrics.set_input(pred_binary, gt_binary, spacing=(1, 1))
        val_iou_scores.append(metrics.iou())
        val_nsd_scores.append(metrics.nsd(2))
        val_dice_scores.append(metrics.dsc())
    
    if len(val_iou_scores) == 0:
        return 0.0, 0.0, 0.0
    
    mean_iou = np.mean(val_iou_scores)
    mean_nsd = np.mean(val_nsd_scores)
    mean_dice = np.mean(val_dice_scores)
    
    return mean_iou, mean_nsd, mean_dice


@ex.config
def config():
    dataset_id = 502
    dataset_name = "UterUSSegmentation"
    image_path = ''
    seg_path = ''
    csv_output = 'input_nnunet.csv'
    sacred_runs = 'uterus_runs'
    neptune_project = 'jumutc/uterus'
    neptune_dataset = 'UterUS (public)'


@ex.main
def run_experiment(_run, dataset_id, dataset_name, image_path, seg_path, csv_output, sacred_runs, neptune_project, neptune_dataset):
    best_iou_scores = {}
    best_nsd_scores = {}
    best_dice_scores = {}
    
    _neptune_run = neptune.init_run(
        project=neptune_project,
        api_token=os.environ.get('NEPTUNE_API_TOKEN'),
    )
    _neptune_run['dataset'] = neptune_dataset
    _neptune_run['parameters'] = {
        'framework': 'nnUNet',
    }
    
    # Create dataframe
    df = create_df(image_path, seg_path)
    print('Total Images: ', len(df))
    df.to_csv(csv_output, index=False)
    
    # Perform cross-validation with GroupShuffleSplit (same as original)
    for fold, (X_train, X_val) in enumerate(GroupShuffleSplit(n_splits=3, test_size=0.15, random_state=0).split(df.index, groups=df['volume_id'])):
        print(f'\n{"="*60}')
        print(f'Fold {fold + 1}/3')
        print('Train Size   : ', len(X_train))
        print('Val Size     : ', len(X_val))
        
        if set(df.loc[X_train]['volume_id'].values) & set(df.loc[X_val]['volume_id'].values):
            raise ValueError('Intersecting validation and train groups detected!')
        
        torch.manual_seed(fold)
        np.random.seed(fold)
        
        # Convert data to nnUNet format for this fold
        fold_dataset_id = dataset_id + fold
        dataset_path, imagesTr_path, labelsTr_path, train_cases, val_cases = convert_to_nnunet_format(
            df, fold_dataset_id, dataset_name, X_train, X_val
        )
        
        # Count actual converted cases
        num_total = len([f for f in os.listdir(imagesTr_path) if f.endswith('.png')])
        
        # Create dataset.json
        create_dataset_json(dataset_path, dataset_name, num_total)
        
        # Extract dataset fingerprint (required before experiment planning)
        print(f"\nExtracting dataset fingerprint for Dataset{fold_dataset_id:03d}...")
        fingerprint_extracted = False
        try:
            fingerprint_extractor = DatasetFingerprintExtractor(fold_dataset_id, num_processes=4, verbose=True)
            fingerprint_extractor.run(overwrite_existing=True)
            print("Dataset fingerprint extracted successfully.")
            fingerprint_extracted = True
        except Exception as e:
            print(f"ERROR: Dataset fingerprint extraction failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot proceed without dataset fingerprint. Extraction failed: {e}")
        
        if not fingerprint_extracted:
            raise RuntimeError("Dataset fingerprint extraction failed. Cannot proceed.")
        
        # Run experiment planning
        print(f"\nRunning experiment planning for Dataset{fold_dataset_id:03d}...")
        fold_dataset_name = f"Dataset{fold_dataset_id:03d}_{dataset_name}"
        plans_file = os.path.join(nnUNet_preprocessed, fold_dataset_name, 'nnUNetPlans.json')
        
        try:
            planner = ExperimentPlanner(fold_dataset_id)
            planner.plan_experiment()
            print("Experiment planning completed successfully.")
            
            # Verify plans file was created
            if not os.path.exists(plans_file):
                raise FileNotFoundError(f"Plans file was not created at {plans_file}")
            print(f"Verified plans file exists: {plans_file}")
            
            # Copy dataset.json to preprocessed folder (required for preprocessing)
            preprocessed_dataset_folder = os.path.join(nnUNet_preprocessed, fold_dataset_name)
            os.makedirs(preprocessed_dataset_folder, exist_ok=True)
            dataset_json_src = os.path.join(dataset_path, 'dataset.json')
            dataset_json_dst = os.path.join(preprocessed_dataset_folder, 'dataset.json')
            if os.path.exists(dataset_json_src):
                shutil.copy2(dataset_json_src, dataset_json_dst)
                print(f"Copied dataset.json to preprocessed folder: {dataset_json_dst}")
        except Exception as e:
            print(f"ERROR: Experiment planning failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot proceed without experiment plans. Planning failed: {e}")
        
        # Create splits.json (after preprocessing, but we create it early)
        create_splits_json(fold_dataset_id, dataset_name, train_cases, val_cases)
        
        # Run preprocessing (required before training)
        print(f"\nRunning preprocessing for Dataset{fold_dataset_id:03d} configuration '2d'...")
        try:
            preprocessor = DefaultPreprocessor(verbose=True)
            preprocessor.run(
                dataset_name_or_id=fold_dataset_id,
                configuration_name='2d',
                plans_identifier='nnUNetPlans',
                num_processes=10
            )
            print("Preprocessing completed successfully.")
        except Exception as e:
            print(f"ERROR: Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot proceed without preprocessing. Preprocessing failed: {e}")
        
        # Run training using nnUNet's training API
        print(f"\nStarting nnUNet training for fold {fold + 1}...")
        try:
            # fold_dataset_name already defined above
            # Note: nnUNet v2 API - adjust parameters as needed for your version
            run_training(
                dataset_name_or_id=fold_dataset_name,
                configuration='2d',
                fold=0,  # Use fold 0 since we're using custom splits in splits_final.json
                trainer_class_name='nnUNetTrainer',
                num_gpus=1,
                use_compressed_data=False,
                export_validation_probabilities=False,
                continue_training=False,
                only_run_validation=False,
                disable_checkpointing=False,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing to evaluation...")
        
        # After training, evaluate and log metrics
        print(f"Evaluating predictions for fold {fold + 1}...")
        mean_iou, mean_nsd, mean_dice = evaluate_predictions(
            fold_dataset_id, dataset_name, val_cases, df, X_val
        )
        
        best_iou_scores[fold] = mean_iou
        best_nsd_scores[fold] = mean_nsd
        best_dice_scores[fold] = mean_dice
        
        _neptune_run[f"validation.{fold}.mIoU"] = float(best_iou_scores[fold])
        _neptune_run[f"validation.{fold}.NSD"] = float(best_nsd_scores[fold])
        _neptune_run[f"validation.{fold}.Dice"] = float(best_dice_scores[fold])
        
        _run.log_scalar(f"validation.{fold}.mIoU", float(best_iou_scores[fold]))
        _run.log_scalar(f"validation.{fold}.NSD", float(best_nsd_scores[fold]))
        _run.log_scalar(f"validation.{fold}.Dice", float(best_dice_scores[fold]))
        
        # Clean up for next fold (optional)
        # You might want to keep the data for analysis
    
    best_iou_scores = np.array(list(best_iou_scores.values()))
    best_nsd_scores = np.array(list(best_nsd_scores.values()))
    best_dice_scores = np.array(list(best_dice_scores.values()))
    print(f"TOTAL AVERAGE CV mIOU: {np.nanmean(best_iou_scores)}")
    print(f"TOTAL AVERAGE CV NSD: {np.nanmean(best_nsd_scores)}")
    print(f"TOTAL AVERAGE CV Dice: {np.nanmean(best_dice_scores)}")
    
    _neptune_run["mIoU.average"] = float(np.nanmean(best_iou_scores))
    _neptune_run["NSD.average"] = float(np.nanmean(best_nsd_scores))
    _neptune_run["Dice.average"] = float(np.nanmean(best_dice_scores))
    _neptune_run["mIoU.std"] = float(np.nanstd(best_iou_scores))
    _neptune_run["NSD.std"] = float(np.nanstd(best_nsd_scores))
    _neptune_run["Dice.std"] = float(np.nanstd(best_dice_scores))
    
    _run.log_scalar("average.mIoU", float(np.nanmean(best_iou_scores)))
    _run.log_scalar("average.NSD", float(np.nanmean(best_nsd_scores)))
    _run.log_scalar("average.Dice", float(np.nanmean(best_dice_scores)))
    _run.log_scalar("std.mIoU", float(np.nanstd(best_iou_scores)))
    _run.log_scalar("std.NSD", float(np.nanstd(best_nsd_scores)))
    _run.log_scalar("std.Dice", float(np.nanstd(best_dice_scores)))
    
    _neptune_run.stop()


if __name__ == '__main__':
    args = parse_args()
    
    # Set up Sacred observer
    ex.observers.append(FileStorageObserver(args.sacred_runs))
    
    # Run experiment with command-line arguments
    ex.run(config_updates={
        'dataset_id': args.dataset_id,
        'dataset_name': args.dataset_name,
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'neptune_dataset': args.neptune_dataset,
    })
