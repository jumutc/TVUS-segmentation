"""
Uterus Segmentation Private Experiment using MONAI (image/seg directory layout)

This script reproduces Uterus_Segmentation_Private_Exp.py but uses the MONAI library
instead of segmentation_models_pytorch.

Key differences from Uterus_Segmentation_Private_Exp_niches_MONAI.py:
- Uses image_path/seg_path directory layout (preprocessed images + masked segmentations)
- Loads image/mask pairs from disk via create_df(), not from videos/Encord JSON

MONAI components:
- Networks: FlexibleUNet, BasicUNetPlusPlus, BasicUNet, AttentionUnet
- Loss: BoundaryWeightedTverskyLoss (Tversky + boundary-distance regression)
- Transforms: MONAI Compose pipeline
- Metrics: MONAI DiceMetric, MeanIoU, SurfaceDiceMetric (GPU)
- Logging: Sacred (no Neptune)

Requirements:
- monai: pip install monai
- Recommended: CUDA GPU with >=16GB VRAM (24GB comfortable for 512x768, batch_size=2).
  This variant preloads images into memory; prefer >=32–64GB system RAM for large datasets.
"""

import argparse
import gc
import glob
import os
import ssl
import time

import cv2
import numpy as np
import pandas as pd
import torch
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

ex = Experiment("uterus_exp_monai_images")

MONAI_MODELS = {
    "FlexibleUNet",
    "BasicUNetPlusPlus",
    "BasicUNet",
    "AttentionUnet",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uterus Segmentation Private Experiment using MONAI (image/seg directories)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to directory containing image volumes",
    )
    parser.add_argument(
        "--seg_path",
        type=str,
        required=True,
        help="Path to directory containing segmentation masks",
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
        default="uterus_runs_monai_images",
        help="Path to Sacred runs directory (default: uterus_runs_monai_images)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TVUS (private)",
        help="Dataset name for logging (default: TVUS (private))",
    )
    return parser.parse_args()


def find_in_paths(p, image_paths):
    filename = os.path.basename(p).split("_")[0]
    in_paths = [_p for _p in image_paths if filename in _p]
    return in_paths[0] if in_paths else p


def create_df(image_path, seg_path):
    images, segmentations, volume_ids, img_paths, seg_paths = [], [], [], [], []

    for volume_id in tqdm(os.listdir(image_path)):
        image_paths = sorted(glob.glob(os.path.join(image_path, volume_id, "*", volume_id + "*")))
        preprocessed_paths = sorted(glob.glob(os.path.join(seg_path, volume_id, "*", volume_id + "*_preprocessed*")))
        seg_masks = sorted(glob.glob(os.path.join(seg_path, volume_id, "*", "masked_" + volume_id + "*")))
        preprocessed_paths = [find_in_paths(p, image_paths) for p in preprocessed_paths]

        for _image_path, _seg_path in zip(preprocessed_paths, seg_masks):
            images.append(cv2.cvtColor(cv2.imread(_image_path), cv2.COLOR_BGR2RGB))
            segmentations.append(cv2.imread(_seg_path, cv2.IMREAD_GRAYSCALE))
            volume_ids.append(volume_id)
            img_paths.append(_image_path)
            seg_paths.append(_seg_path)

    return pd.DataFrame(
        {
            "img": images,
            "seg": segmentations,
            "volume_id": volume_ids,
            "img_path": img_paths,
            "seg_path": seg_paths,
        },
        index=np.arange(0, len(images)),
    )


def build_monai_transforms(height, width, train=True, use_extra_augmentations=False):
    """Build MONAI preprocessing/augmentation pipeline for image and label.

    Spatial size is fixed to (height, width). RandRotate90d is only used when the
    canvas is square; on non-square inputs (e.g. 512x768) a 90° rotate swaps H/W
    and breaks torch.stack in the DataLoader collate.
    """
    keys = ["image", "label"]
    spatial_size = (height, width)
    transforms = [
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Resized(keys=keys, spatial_size=spatial_size, mode=["bilinear", "nearest"]),
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
            extra = [
                RandRotated(
                    keys=keys,
                    range_x=np.pi / 12,
                    prob=0.2,
                    keep_size=True,
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                ),
            ]
            # 90° rotates swap axes; only safe when height == width.
            if height == width:
                extra.append(RandRotate90d(keys=keys, prob=0.2, spatial_axes=(0, 1)))
            extra.extend(
                [
                    RandZoomd(
                        keys=keys,
                        prob=0.3,
                        min_zoom=0.9,
                        max_zoom=1.1,
                        keep_size=True,
                        mode=["bilinear", "nearest"],
                    ),
                    RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
                    RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),
                ]
            )
            transforms.extend(extra)
            # Guarantee uniform spatial size before batch collation.
            transforms.append(Resized(keys=keys, spatial_size=spatial_size, mode=["bilinear", "nearest"]))

    transforms.append(EnsureTyped(keys=keys, data_type="tensor"))
    return Compose(transforms)


class TVUSMONAIDataset(Dataset):
    """Dataset that loads preloaded images/masks from dataframe and applies MONAI transforms."""

    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = row["img"]
        mask_1 = row["seg"]
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
        "out_channels": 2,
        "pretrained": True,
        "decoder_channels": (256, 128, 64, 32, 16),
    }
    image_path = ""
    seg_path = ""
    model_output = "model_tvus.pt"
    csv_output = "input.csv"
    sacred_runs = "uterus_runs_monai_images"
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
def run_experiment(_run, image_path, seg_path, model_output, csv_output, use_extra_augmentations):
    max_lr = 1e-4
    epochs = 200
    weight_decay = 1e-4
    best_iou_scores = {}
    best_nsd_scores = {}
    best_dice_scores = {}
    height, width = 512, 768

    df = create_df(image_path, seg_path)
    print("Total Images: ", len(df))
    print(df.head())
    df[["volume_id", "img_path", "seg_path"]].to_csv(csv_output, index=False)

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

        train_loader = DataLoader(
            train_set, batch_size=2, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_set, batch_size=1, shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available()
        )

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


def get_model_output_path(base_path, model_name, encoder_name, has_aug):
    """Generate a unique model output path with postfix."""
    base_name, ext = os.path.splitext(base_path)
    aug_suffix = "_aug" if has_aug else "_noaug"
    postfix = f"_{model_name}_{encoder_name}_{aug_suffix}"
    return f"{base_name}{postfix}{ext}"


if __name__ == "__main__":
    args = parse_args()

    ex.observers.append(FileStorageObserver(args.sacred_runs))

    ex.run(
        config_updates={
            "losses": "[BoundaryWeightedTverskyLoss(sigmoid=True, include_background=True, boundary_lambda=0.5)]",
            "encoder_name": "efficientnet-b7",
            "model_name": "FlexibleUNet",
            "model_params": {
                "in_channels": 3,
                "out_channels": 2,
                "pretrained": True,
                "decoder_channels": (256, 128, 64, 32, 16),
            },
            "use_extra_augmentations": False,
            "image_path": args.image_path,
            "seg_path": args.seg_path,
            "model_output": get_model_output_path(args.model_output, "FlexibleUNet", "efficientnet-b7", False),
            "csv_output": args.csv_output,
            "sacred_runs": args.sacred_runs,
            "dataset_name": args.dataset_name,
        }
    )
    # ex.run(
    #     config_updates={
    #         "losses": "[BoundaryWeightedTverskyLoss(sigmoid=True, include_background=True, boundary_lambda=0.5, boundary_sigma=6.0)]",
    #         "encoder_name": "efficientnet-b7",
    #         "model_name": "FlexibleUNet",
    #         "model_params": {
    #             "in_channels": 3,
    #             "out_channels": 1,
    #             "pretrained": True,
    #             "decoder_channels": (256, 128, 64, 32, 16),
    #         },
    #         "use_extra_augmentations": True,
    #         "image_path": args.image_path,
    #         "seg_path": args.seg_path,
    #         "model_output": get_model_output_path(args.model_output, "FlexibleUNet", "efficientnet-b7", True),
    #         "csv_output": args.csv_output,
    #         "sacred_runs": args.sacred_runs,
    #         "dataset_name": args.dataset_name,
    #     }
    # )
    # ex.run(
    #     config_updates={
    #         "losses": "[BoundaryWeightedTverskyLoss(sigmoid=True, include_background=True, boundary_lambda=0.5, boundary_sigma=6.0)]",
    #         "encoder_name": "",
    #         "model_name": "BasicUNetPlusPlus",
    #         "model_params": {
    #             "in_channels": 3,
    #             "out_channels": 1,
    #             "features": (64, 128, 256, 512, 1024, 128),
    #             "deep_supervision": False,
    #         },
    #         "use_extra_augmentations": False,
    #         "image_path": args.image_path,
    #         "seg_path": args.seg_path,
    #         "model_output": get_model_output_path(args.model_output, "BasicUNetPlusPlus", "default", False),
    #         "csv_output": args.csv_output,
    #         "sacred_runs": args.sacred_runs,
    #         "dataset_name": args.dataset_name,
    #     }
    # )
