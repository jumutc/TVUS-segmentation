"""
Uterus Segmentation Private Experiment using MONAI

This script reproduces Uterus_Segmentation_Private_Exp_no_neptune.py but uses the
MONAI library instead of segmentation_models_pytorch.

Key differences:
- Uses MONAI networks (FlexibleUNet, BasicUNetPlusPlus, BasicUNet, AttentionUnet)
- Supports MONAI TverskyLoss and BoundaryWeightedTverskyLoss (Tversky + boundary-distance regression)
- Loads labeled data from Supervisely video exports (ann/*.json + matching videos)
- Uses a separate control_path for unlabeled control videos (negative samples)
- Uses MONAI transforms for preprocessing and augmentation (with optional extra augmentations)
- Preserves the same data loading, GroupShuffleSplit CV, Sacred logging, and MONAI GPU metrics

Requirements:
- monai: pip install monai
"""

import argparse
import gc
import json
import os
import ssl
import time

import cv2
import numpy as np
import pandas as pd
import torch
from monai.losses import TverskyLoss
from monai_segmentation_losses import BoundaryWeightedTverskyLoss, resolve_model_out_channels
from monai_segmentation_metrics import MonaiValidationMetrics
from monai.networks.nets import AttentionUnet, BasicUNet, BasicUNetPlusPlus, FlexibleUNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
)
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ex = Experiment("uterus_exp_monai")

MONAI_MODELS = {
    "FlexibleUNet",
    "BasicUNetPlusPlus",
    "BasicUNet",
    "AttentionUnet",
}

NICHE_CLASS_NAMES = frozenset({"niche"})
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV")


def parse_args():
    parser = argparse.ArgumentParser(description="Uterus Segmentation Private Experiment using MONAI")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Supervisely export root (contains dataset */ann/ and video files)",
    )
    parser.add_argument(
        "--control_path",
        type=str,
        default=None,
        help="Path to root folder with unlabeled control videos; searched recursively in subfolders (optional)",
    )
    parser.add_argument(
        "--control_balance_ratio",
        type=float,
        default=0.3,
        help="Ratio of control (negative) frames to add for balancing (default: 0.3)",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="model_tvus.pt",
        help="Path to save the trained model (default: model_tvus.pt)",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default="input.csv",
        help="Path to save the input CSV file (default: input.csv)",
    )
    parser.add_argument(
        "--sacred_runs",
        type=str,
        default="uterus_runs_monai",
        help="Path to Sacred runs directory (default: uterus_runs_monai)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TVUS (private)",
        help="Dataset name for logging (default: TVUS (private))",
    )
    return parser.parse_args()


def _is_niche_class(class_title):
    title = (class_title or "").lower()
    return title in NICHE_CLASS_NAMES or any(name in title for name in NICHE_CLASS_NAMES)


def _video_name_from_ann_path(ann_path):
    return os.path.basename(ann_path)[:-5] if ann_path.endswith(".json") else os.path.basename(ann_path)


def _find_supervisely_ann_files(data_path):
    ann_files = []
    for root, _, files in os.walk(data_path):
        if os.path.basename(root) != "ann":
            continue
        for fname in files:
            if fname.endswith(".json"):
                ann_files.append(os.path.join(root, fname))
    return sorted(ann_files)


def _build_video_index(search_roots):
    video_index = {}
    for search_root in search_roots:
        if not search_root or not os.path.isdir(search_root):
            continue
        for dirpath, _, filenames in os.walk(search_root):
            for fname in filenames:
                if fname.lower().endswith(VIDEO_EXTENSIONS):
                    video_index.setdefault(fname, os.path.join(dirpath, fname))
    return video_index


def _resolve_video_path(video_name, video_index, ann_path):
    if video_name in video_index:
        return video_index[video_name]

    ann_dir = os.path.dirname(ann_path)
    video_dir = os.path.join(os.path.dirname(ann_dir), "video")
    candidate = os.path.join(video_dir, video_name)
    if os.path.isfile(candidate):
        return candidate

    stem = os.path.splitext(video_name)[0]
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(video_dir, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
        indexed = video_index.get(f"{stem}{ext}")
        if indexed:
            return indexed
    return None


def _collect_control_videos(control_path):
    """Collect video files from control_path, including any nested subfolders."""
    if not control_path or not os.path.isdir(control_path):
        return []
    control_videos = []
    for dirpath, _, filenames in os.walk(control_path):
        for fname in filenames:
            if fname.lower().endswith(VIDEO_EXTENSIONS):
                control_videos.append(os.path.join(dirpath, fname))
    return sorted(set(control_videos))


def _points_to_contour(points):
    if not points:
        return None
    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    if contour.shape[0] < 3:
        return None
    return contour


def _supervisely_polygon_to_mask(exterior, interior, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    exterior_contour = _points_to_contour(exterior)
    if exterior_contour is None:
        return mask

    cv2.fillPoly(mask, [exterior_contour], 255)
    for hole in interior or []:
        hole_contour = _points_to_contour(hole)
        if hole_contour is not None:
            cv2.fillPoly(mask, [hole_contour], 0)
    return mask


def _get_niche_object_keys(annotation):
    return {
        obj["key"]
        for obj in annotation.get("objects", [])
        if _is_niche_class(obj.get("classTitle"))
    }


def _scale_points(points, scale_x, scale_y):
    scaled = []
    for x, y in points:
        scaled.append([int(round(x * scale_x)), int(round(y * scale_y))])
    return scaled


def _create_mask_from_supervisely_frame(frame, niche_keys, ann_width, ann_height, mask_width, mask_height):
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    scale_x = mask_width / ann_width if ann_width else 1.0
    scale_y = mask_height / ann_height if ann_height else 1.0

    for figure in frame.get("figures", []):
        if figure.get("geometryType") != "polygon":
            continue
        if figure.get("objectKey") not in niche_keys:
            continue
        points = figure.get("geometry", {}).get("points", {})
        exterior = _scale_points(points.get("exterior", []), scale_x, scale_y)
        interior = [_scale_points(hole, scale_x, scale_y) for hole in points.get("interior", [])]
        polygon_mask = _supervisely_polygon_to_mask(exterior, interior, mask_height, mask_width)
        mask = np.maximum(mask, polygon_mask)
    return mask


def create_df(data_path, control_path=None, control_balance_ratio=0.3):
    """
    Build dataframe from a Supervisely video export:
    - {data_path}/dataset */ann/{video_name}.mp4.json
    - matching videos under sibling video/ folders or anywhere under data_path
    - optional control_path: root folder tree of unlabeled videos (searched recursively)
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    ann_files = _find_supervisely_ann_files(data_path)
    if not ann_files:
        raise FileNotFoundError(f"No Supervisely ann/*.json files found under {data_path}")

    video_index = _build_video_index([data_path])
    rows = []
    missing_videos = 0
    skipped_empty = 0

    for ann_path in tqdm(ann_files, desc="Loading labeled data"):
        video_name = _video_name_from_ann_path(ann_path)
        video_path = _resolve_video_path(video_name, video_index, ann_path)
        if not video_path:
            missing_videos += 1
            continue

        with open(ann_path, encoding="utf-8") as f:
            annotation = json.load(f)

        niche_keys = _get_niche_object_keys(annotation)
        if not niche_keys:
            skipped_empty += 1
            continue

        frames = annotation.get("frames", [])
        if not frames:
            skipped_empty += 1
            continue

        ann_width = int(annotation["size"]["width"])
        ann_height = int(annotation["size"]["height"])

        cap = cv2.VideoCapture(video_path)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if vh <= 0 or vw <= 0:
            missing_videos += 1
            continue

        volume_id = os.path.splitext(os.path.basename(video_name))[0]
        for frame_data in frames:
            frame_idx = int(frame_data["index"])
            mask = _create_mask_from_supervisely_frame(
                frame_data,
                niche_keys,
                ann_width,
                ann_height,
                vw,
                vh,
            )
            if not mask.any():
                continue
            rows.append(
                {
                    "video_path": video_path,
                    "frame_idx": frame_idx,
                    "seg": mask,
                    "volume_id": volume_id,
                }
            )

    n_labeled = len(rows)
    n_control_target = max(0, int(n_labeled * control_balance_ratio))
    print("Total Labeled Images: ", n_labeled)
    if missing_videos:
        print(f"Skipped {missing_videos} annotation files with missing or unreadable videos")
    if skipped_empty:
        print(f"Skipped {skipped_empty} annotation files without Niche labels")

    control_videos = _collect_control_videos(control_path)
    if n_control_target > 0 and control_videos:
        added = 0
        for video_path in control_videos:
            if added >= n_control_target:
                break
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            if n_frames <= 0 or vh <= 0 or vw <= 0:
                continue
            volume_id = os.path.splitext(os.path.basename(video_path))[0]
            indices = np.linspace(0, n_frames - 1, min(n_control_target - added, max(1, n_frames // 5)), dtype=int)
            for fi in indices:
                if added >= n_control_target:
                    break
                rows.append(
                    {
                        "video_path": video_path,
                        "frame_idx": int(fi),
                        "seg": np.zeros((vh, vw), dtype=np.uint8),
                        "volume_id": volume_id,
                    }
                )
                added += 1
        print(f"Added {added} control frames from {len(control_videos)} control videos")
    elif n_control_target > 0:
        print("No control videos found; skipping control balancing")

    if not rows:
        raise ValueError("No training samples found. Check data_path, video files, and annotations.")

    return pd.DataFrame(rows, index=np.arange(len(rows)))


def _read_frame_from_video(video_path, frame_idx):
    """Load a single frame from video as RGB."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_monai_transforms(height, width, train=True, use_extra_augmentations=False):
    """Build MONAI preprocessing/augmentation pipeline for image and label."""
    keys = ["image", "label"]
    transforms = [
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Resized(keys=keys, spatial_size=(height, width), mode=["bilinear", "nearest"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ]

    if train:
        transforms.extend(
            [
                RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
            ]
        )
        if use_extra_augmentations:
            transforms.extend(
                [
                    RandRotated(keys=keys, range_x=np.pi / 12, prob=0.2, mode=["bilinear", "nearest"], padding_mode="zeros"),
                    RandRotate90d(keys=keys, prob=0.2, spatial_axes=(0, 1)),
                    RandZoomd(keys=keys, prob=0.3, min_zoom=0.9, max_zoom=1.1, mode=["bilinear", "nearest"]),
                    RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
                    RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),
                ]
            )

    transforms.append(EnsureTyped(keys=keys, data_type="tensor"))
    return Compose(transforms)


class TVUSMONAIDataset(Dataset):
    """Dataset that loads frames from videos and applies MONAI transforms."""

    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        video_path = row["video_path"]
        frame_idx = int(row["frame_idx"])
        mask_1 = row["seg"]

        img = _read_frame_from_video(video_path, frame_idx)
        mask = (mask_1 > 0).astype(np.float32)

        data = {"image": img, "label": mask}
        if self.transform is not None:
            data = self.transform(data)

        return data["image"], data["label"], idx


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def create_model(model_name, encoder_name, model_params):
    """Create a MONAI segmentation model based on configuration."""
    if model_name not in MONAI_MODELS:
        raise ValueError(f"Unsupported model_name '{model_name}'. Choose from {sorted(MONAI_MODELS)}")

    params = model_params.copy()
    in_channels = params.pop("in_channels", 3)
    out_channels = params.pop("out_channels", 1)
    spatial_dims = params.pop("spatial_dims", 2)

    if model_name == "FlexibleUNet":
        decoder_channels = params.pop("decoder_channels", (256, 128, 64, 32, 16))
        pretrained = params.pop("pretrained", True)
        if params:
            raise ValueError(f"Unknown FlexibleUNet params: {params}")
        return FlexibleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            backbone=encoder_name,
            pretrained=pretrained,
            decoder_channels=tuple(decoder_channels),
            spatial_dims=spatial_dims,
        )

    if model_name == "BasicUNetPlusPlus":
        features = params.pop("features", (64, 128, 256, 512, 1024, 128))
        deep_supervision = params.pop("deep_supervision", False)
        if params:
            raise ValueError(f"Unknown BasicUNetPlusPlus params: {params}")
        return BasicUNetPlusPlus(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=tuple(features),
            deep_supervision=deep_supervision,
        )

    if model_name == "BasicUNet":
        features = params.pop("features", (64, 128, 256, 512, 1024, 128))
        if params:
            raise ValueError(f"Unknown BasicUNet params: {params}")
        return BasicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=tuple(features),
        )

    if model_name == "AttentionUnet":
        channels = params.pop("channels", (16, 32, 64, 128, 256))
        strides = params.pop("strides", (2, 2, 2, 2))
        if params:
            raise ValueError(f"Unknown AttentionUnet params: {params}")
        return AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(channels),
            strides=tuple(strides),
        )

    raise ValueError(f"Model factory missing implementation for {model_name}")


def fit(
    _run,
    epochs,
    model,
    train_loader,
    val_loader,
    losses,
    optimizer,
    scheduler,
    best_iou_scores,
    best_nsd_scores,
    best_dice_scores,
    fold,
    model_output_path,
    val_df,
):
    train_losses = []
    test_losses = []
    val_iou = []
    val_nsd = []
    val_dice = []
    lrs = []
    min_dice = -np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()

        since = time.time()
        running_loss = 0
        model.train()
        for _, data in enumerate(tqdm(train_loader)):
            image, mask, _ = data

            image = image.to(device)
            mask = mask.to(device)

            output = model(image)
            loss = 0
            for _loss in losses:
                loss += _loss(output, mask)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            val_metrics = MonaiValidationMetrics(nsd_tolerance=6.0)

            with torch.no_grad():
                for _, data in enumerate(tqdm(val_loader)):
                    image, mask, idx = data

                    image = image.to(device)
                    mask = mask.to(device)
                    output = model(image)
                    val_metrics.update(output, val_df, idx, device)

                    for _loss in losses:
                        test_loss += _loss(output, mask).item()

            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))
            test_iou_score, test_nsd_score, test_dice_score, test_iou_scores, test_nsd_scores, test_dice_scores = (
                val_metrics.aggregate()
            )

            if min_dice < test_dice_score:
                print(
                    "Validation Dice increasing.. {:.3f} >> {:.3f}, per class >> {:s}".format(
                        min_dice,
                        test_dice_score,
                        str(np.mean(test_dice_scores, axis=0)),
                    )
                )
                best_iou_scores[fold] = test_iou_score
                best_nsd_scores[fold] = test_nsd_score
                best_dice_scores[fold] = test_dice_score
                min_dice = test_dice_score
                decrease += 1
                not_improve = 0
                print("saving model...")
                torch.save(model, model_output_path)

            if test_dice_score < min_dice:
                not_improve += 1
                print(f"Dice not increased for {not_improve} time")
                if not_improve == 50:
                    print("Dice not increased for 50 times, Stop Training")
                    break

            val_iou.append(test_iou_score)
            val_nsd.append(test_nsd_score)
            val_dice.append(test_dice_score)

            _run.log_scalar(f"training.{fold}.loss", float(running_loss / len(train_loader)))
            _run.log_scalar(f"validation.{fold}.loss", float(test_loss / len(val_loader)))
            _run.log_scalar(f"validation.{fold}.mIoU", float(val_iou[-1]))
            _run.log_scalar(f"validation.{fold}.NSD", float(val_nsd[-1]))
            _run.log_scalar(f"validation.{fold}.Dice", float(val_dice[-1]))

            print(
                "Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                "Val mIoU: {:.3f}..".format(val_iou[-1]),
                "Val NSD: {:.3f}..".format(val_nsd[-1]),
                "Val Dice: {:.3f}..".format(val_dice[-1]),
                "Time: {:.2f}m".format((time.time() - since) / 60),
            )

    history = {
        "train_loss": train_losses,
        "val_loss": test_losses,
        "val_miou": val_iou,
        "val_nsd": val_nsd,
        "val_dice": val_dice,
        "lrs": lrs,
    }
    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    return history


@ex.config
def config():
    losses = []
    encoder_name = ""
    model_name = "FlexibleUNet"
    model_params = {
        "in_channels": 3,
        "out_channels": 1,
        "pretrained": True,
        "decoder_channels": (256, 128, 64, 32, 16),
    }
    data_path = ""
    control_path = ""
    control_balance_ratio = 0.3
    model_output = "model_tvus.pt"
    csv_output = "input.csv"
    sacred_runs = "uterus_runs_monai"
    dataset_name = "TVUS (private)"
    use_extra_augmentations = False


@ex.capture
def get_losses(losses):
    return losses


@ex.capture
def get_encoder_name(encoder_name):
    return encoder_name


@ex.capture
def get_model_name(model_name):
    return model_name


@ex.capture
def get_model_params(model_params):
    return model_params


@ex.capture
def get_use_extra_augmentations(use_extra_augmentations):
    return use_extra_augmentations


@ex.main
def run_experiment(
    _run,
    data_path,
    control_path,
    control_balance_ratio,
    model_output,
    csv_output,
    use_extra_augmentations,
):
    max_lr = 1e-4
    epochs = 200
    weight_decay = 1e-4
    best_iou_scores = {}
    best_nsd_scores = {}
    best_dice_scores = {}
    height, width = 512, 768

    df = create_df(data_path, control_path or None, control_balance_ratio)
    print("Total Images: ", len(df))
    print(df.head())
    df[["volume_id", "video_path", "frame_idx"]].to_csv(csv_output, index=False)

    for i, (X_train, X_val) in enumerate(
        GroupShuffleSplit(n_splits=3, test_size=0.15, random_state=0).split(df.index, groups=df["volume_id"])
    ):
        print("Train Size   : ", len(X_train))
        print("Val Size     : ", len(X_val))

        if set(df.loc[X_train]["volume_id"].values) & set(df.loc[X_val]["volume_id"].values):
            raise ValueError("Intersecting validation and train groups detected!")

        torch.manual_seed(i)
        np.random.seed(i)

        losses = eval(get_losses())
        model_params = get_model_params().copy()
        resolved_out_channels = resolve_model_out_channels(losses, model_params.get("out_channels", 1))
        if resolved_out_channels != model_params.get("out_channels", 1):
            print(
                f"Adjusting out_channels from {model_params.get('out_channels', 1)} "
                f"to {resolved_out_channels} for boundary distance regression."
            )
            model_params["out_channels"] = resolved_out_channels

        model = create_model(get_model_name(), get_encoder_name(), model_params)
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)

        t_train = build_monai_transforms(height, width, train=True, use_extra_augmentations=use_extra_augmentations)
        t_val = build_monai_transforms(height, width, train=False)

        train_df = df.loc[X_train].reset_index()
        train_set = TVUSMONAIDataset(train_df, t_train)
        val_df = df.loc[X_val].reset_index()
        val_set = TVUSMONAIDataset(val_df, t_val)

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
        )

        fit(
            _run,
            epochs,
            model,
            train_loader,
            val_loader,
            losses,
            optimizer,
            scheduler,
            best_iou_scores,
            best_nsd_scores,
            best_dice_scores,
            i,
            model_output,
            val_df,
        )

    best_iou_scores = np.array(list(best_iou_scores.values()))
    best_nsd_scores = np.array(list(best_nsd_scores.values()))
    best_dice_scores = np.array(list(best_dice_scores.values()))
    print(f"TOTAL AVERAGE CV mIOU: {np.nanmean(best_iou_scores)}")
    print(f"TOTAL AVERAGE CV NSD: {np.nanmean(best_nsd_scores)}")
    print(f"TOTAL AVERAGE CV Dice: {np.nanmean(best_dice_scores)}")

    _run.log_scalar("average.mIoU", float(np.nanmean(best_iou_scores)))
    _run.log_scalar("average.NSD", float(np.nanmean(best_nsd_scores)))
    _run.log_scalar("average.Dice", float(np.nanmean(best_dice_scores)))
    _run.log_scalar("std.mIoU", float(np.nanstd(best_iou_scores)))
    _run.log_scalar("std.NSD", float(np.nanstd(best_nsd_scores)))
    _run.log_scalar("std.Dice", float(np.nanstd(best_dice_scores)))


LOSS_PRESETS = {
    "boundary_tversky": {
        "losses": "[BoundaryWeightedTverskyLoss(sigmoid=True, include_background=True, boundary_lambda=0.5)]",
        "out_channels": 2,
    },
    "tversky": {
        "losses": "[TverskyLoss(sigmoid=True, include_background=True)]",
        "out_channels": 1,
    },
}

MODEL_PRESETS = {
    "FlexibleUNet": {
        "encoder_name": "efficientnet-b7",
        "model_params": {
            "in_channels": 3,
            "pretrained": True,
            "decoder_channels": (256, 128, 64, 32, 16),
        },
    },
    "BasicUNetPlusPlus": {
        "encoder_name": "",
        "model_params": {
            "in_channels": 3,
            "features": (64, 128, 256, 512, 1024, 128),
            "deep_supervision": False,
        },
    },
    "BasicUNet": {
        "encoder_name": "",
        "model_params": {
            "in_channels": 3,
            "features": (64, 128, 256, 512, 1024, 128),
        },
    },
    "AttentionUnet": {
        "encoder_name": "",
        "model_params": {
            "in_channels": 3,
            "channels": (16, 32, 64, 128, 256),
            "strides": (2, 2, 2, 2),
        },
    },
}


def get_model_output_path(base_path, model_name, encoder_name, has_aug, loss_name=""):
    """Generate a unique model output path with postfix."""
    base_name, ext = os.path.splitext(base_path)
    aug_suffix = "_aug" if has_aug else "_noaug"
    encoder_suffix = encoder_name or "default"
    loss_suffix = loss_name or "loss"
    postfix = f"_{model_name}_{encoder_suffix}_{loss_suffix}{aug_suffix}"
    return f"{base_name}{postfix}{ext}"


def _common_run_kwargs(args):
    return {
        "data_path": args.data_path,
        "control_path": args.control_path or "",
        "control_balance_ratio": args.control_balance_ratio,
        "csv_output": args.csv_output,
        "sacred_runs": args.sacred_runs,
        "dataset_name": args.dataset_name,
    }


def build_experiment_configs(args):
    """Build Sacred config updates for every model × loss × augmentation combination."""
    common = _common_run_kwargs(args)
    configs = []

    for model_name, model_preset in MODEL_PRESETS.items():
        encoder_name = model_preset["encoder_name"]
        for loss_name, loss_preset in LOSS_PRESETS.items():
            model_params = model_preset["model_params"].copy()
            model_params["out_channels"] = loss_preset["out_channels"]

            for use_extra_augmentations in (False, True):
                configs.append(
                    {
                        **common,
                        "losses": loss_preset["losses"],
                        "encoder_name": encoder_name,
                        "model_name": model_name,
                        "model_params": model_params,
                        "use_extra_augmentations": use_extra_augmentations,
                        "model_output": get_model_output_path(
                            args.model_output,
                            model_name,
                            encoder_name,
                            use_extra_augmentations,
                            loss_name,
                        ),
                    }
                )

    return configs


if __name__ == "__main__":
    args = parse_args()

    ex.observers.append(FileStorageObserver(args.sacred_runs))

    for config_updates in build_experiment_configs(args):
        print(
            "Running experiment:",
            config_updates["model_name"],
            config_updates["encoder_name"] or "default",
            config_updates["losses"],
            "aug" if config_updates["use_extra_augmentations"] else "noaug",
        )
        ex.run(config_updates=config_updates)
