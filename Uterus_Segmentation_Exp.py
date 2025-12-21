import argparse
import gc
import glob
import os
import random
import ssl
import sys
import time

import albumentations as A
import cv2
import neptune
import nibabel as nib
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

from MeshMetrics import DistanceMetrics
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sacred import Experiment
from sacred.observers import FileStorageObserver
from skimage.color import label2rgb
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ex = Experiment("uterus_exp")

metrics = DistanceMetrics()

def parse_args():
    parser = argparse.ArgumentParser(description='Uterus Segmentation Experiment')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to directory containing NIfTI image volumes')
    parser.add_argument('--seg_path', type=str, required=True,
                        help='Path to directory containing NIfTI segmentation volumes')
    parser.add_argument('--model_output', type=str, default='model.pt',
                        help='Path to save the trained model (default: model.pt)')
    parser.add_argument('--csv_output', type=str, default='input.csv',
                        help='Path to save the input CSV file (default: input.csv)')
    parser.add_argument('--sacred_runs', type=str, default='uterus_runs',
                        help='Path to Sacred runs directory (default: uterus_runs)')
    parser.add_argument('--neptune_project', type=str, default='jumutc/uterus',
                        help='Neptune project name (default: jumutc/uterus)')
    parser.add_argument('--dataset_name', type=str, default='UterUS (public)',
                        help='Dataset name for logging (default: UterUS (public))')
    return parser.parse_args()

def create_df(image_path, seg_path):
    image_paths = sorted(glob.glob(f"{image_path}/*.nii.gz"))
    images, segmentations, volume_ids = [], [], []

    for img_path in tqdm(image_paths):
        volume_id = os.path.basename(img_path)
        nii_img = nib.load(img_path)
        nii_seg = nib.load(os.path.join(seg_path, volume_id))
        nii_img_data = nii_img.get_fdata()
        nii_seg_data = nii_seg.get_fdata()

        for i in range(0,nii_img_data.shape[-1]):
            images.append(nii_img_data[..., i:i+1])
            segmentations.append(nii_seg_data[..., i:i+1])
            volume_ids.append(volume_id)


    return pd.DataFrame({'img': images, 'seg': segmentations, 'volume_id': volume_ids},
                        index=np.arange(0, len(images)))


class UterusDataset(Dataset):
    def __init__(self, X, transform):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = np.repeat(self.X.loc[idx]['img'], 3, axis=-1)
        mask = self.X.loc[idx]['seg']
        
        # Ensure consistent dtype for OpenCV compatibility (float32 for albumentations)
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        aug = self.transform(image=img, mask=mask)

        img = aug['image']
        mask = aug['mask']

        mask = torch.from_numpy(np.transpose(mask, axes=(2, 0, 1))).round().float()
        t = T.Compose([T.ToTensor()])
        img = t(img).float()

        return img, mask


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(_run, _neptune_run, epochs, model, train_loader, val_loader, losses, optimizer, scheduler, best_iou_scores, best_nsd_scores, best_dice_scores, fold, model_output_path):
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
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image, mask = data

            image = image.to(device)
            mask = mask.to(device)

            # forward
            output = model(image)
            loss = 0

            for _loss in losses:
                loss += _loss(output, mask)

            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            val_iou_scores = []
            val_nsd_scores = []
            val_dice_scores = []
            images, gts, outputs = [], [], []

            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)

                    images.append(image.cpu())
                    outputs.append(output.cpu())
                    gts.append(mask.cpu())

                    gt = np.squeeze(gts[-1].detach().numpy(), axis=(0, 1))
                    out = np.squeeze(outputs[-1].detach().numpy(), axis=(0, 1))
                    metrics.set_input(out > 0, gt > 0, spacing=(1, 1))
                    val_iou_scores.append(metrics.iou())
                    val_nsd_scores.append(metrics.nsd(2))
                    val_dice_scores.append(metrics.dsc())

                    for _loss in losses:
                        test_loss += _loss(output, mask).item()

            # calculate mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))
            test_iou_scores = np.vstack(val_iou_scores)
            test_iou_score = np.mean(test_iou_scores)

            # NSD scores
            test_nsd_scores = np.vstack(val_nsd_scores)
            test_nsd_score = np.mean(test_nsd_scores)
            
            # Dice scores
            test_dice_scores = np.vstack(val_dice_scores)
            test_dice_score = np.mean(test_dice_scores)

            if min_dice < test_dice_score:
                print('Validation Dice increasing.. {:.3f} >> {:.3f}, per class >> {:s}'.format(min_dice, test_dice_score,
                                                                                               str(np.mean(
                                                                                                   test_dice_scores,
                                                                                                   axis=0))))
                best_iou_scores[fold] = test_iou_score
                best_nsd_scores[fold] = test_nsd_score
                best_dice_scores[fold] = test_dice_score
                min_dice = test_dice_score
                decrease += 1
                not_improve = 0
                print('saving model...')
                torch.save(model, model_output_path)

            if test_dice_score < min_dice:
                not_improve += 1
                print(f'Dice not increased for {not_improve} time')
                if not_improve == 50:
                    print('Dice not increased for 50 times, Stop Training')
                    break

            # iou, nsd and dice
            val_iou.append(test_iou_score)
            val_nsd.append(test_nsd_score)
            val_dice.append(test_dice_score)

            _neptune_run[f"training.{fold}.lr"].append(get_lr(optimizer))
            _neptune_run[f"training.{fold}.loss"].append(float(running_loss / len(train_loader)))
            _neptune_run[f"validation.{fold}.loss"].append(float(test_loss / len(val_loader)))
            _neptune_run[f"validation.{fold}.mIoU"].append(float(val_iou[-1]))
            _neptune_run[f"validation.{fold}.NSD"].append(float(val_nsd[-1]))
            _neptune_run[f"validation.{fold}.Dice"].append(float(val_dice[-1]))

            _run.log_scalar(f"training.{fold}.loss", float(running_loss / len(train_loader)))
            _run.log_scalar(f"validation.{fold}.loss", float(test_loss / len(val_loader)))
            _run.log_scalar(f"validation.{fold}.mIoU", float(val_iou[-1]))
            _run.log_scalar(f"validation.{fold}.NSD", float(val_nsd[-1]))
            _run.log_scalar(f"validation.{fold}.Dice", float(val_dice[-1]))

            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou[-1]),
                  "Val NSD: {:.3f}..".format(val_nsd[-1]),
                  "Val Dice: {:.3f}..".format(val_dice[-1]),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'val_miou': val_iou, 'val_nsd': val_nsd, 'val_dice': val_dice, 'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


@ex.config
def config():
    losses = []
    encoder_name = ''
    model_name = 'MAnet'
    model_params = {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1}
    image_path = ''
    seg_path = ''
    model_output = 'model.pt'
    csv_output = 'input.csv'
    sacred_runs = 'uterus_runs'
    neptune_project = 'jumutc/uterus'
    dataset_name = 'UterUS (public)'


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
def get_config(image_path, seg_path, model_output, csv_output, sacred_runs, neptune_project, dataset_name):
    return image_path, seg_path, model_output, csv_output, sacred_runs, neptune_project, dataset_name


def create_model(model_name, encoder_name, model_params):
    """Create a segmentation model based on configuration."""
    model_class = getattr(smp, model_name)
    # Merge encoder_name into model_params
    params = model_params.copy()
    params['encoder_name'] = encoder_name
    return model_class(**params)


@ex.main
def run_experiment(_run, image_path, seg_path, model_output, csv_output, sacred_runs, neptune_project, dataset_name):
    max_lr = 1e-4
    epochs = 200
    weight_decay = 1e-4
    best_iou_scores = {}
    best_nsd_scores = {}
    best_dice_scores = {}
    height, width = 224, 192

    # Create dataframe
    df = create_df(image_path, seg_path)
    print('Total Images: ', len(df))
    print(df.head())
    df[['volume_id']].to_csv(csv_output, index=False)

    _neptune_run = neptune.init_run(
        project=neptune_project,
        api_token=os.environ.get('NEPTUNE_API_TOKEN'),
    )
    _neptune_run['dataset'] = dataset_name
    _neptune_run['parameters'] = {
        'max_lr': max_lr,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'img_height': height,
        'img_width': width,
        'encoder': get_encoder_name(),
        'model': get_model_name(),
        'model_params': get_model_params(),
        'losses': get_losses()
    }

    for i, (X_train, X_val) in enumerate(GroupShuffleSplit(n_splits=3, test_size=0.15, random_state=0).split(df.index, groups=df['volume_id'])):
        print('Train Size   : ', len(X_train))
        print('Val Size     : ', len(X_val))

        if set(df.loc[X_train]['volume_id'].values) & set(df.loc[X_val]['volume_id'].values):
            raise ValueError('Intersecting validation and train groups detected!')

        torch.manual_seed(i)
        np.random.seed(i)

        model = create_model(get_model_name(), get_encoder_name(), get_model_params())
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)

        t_train = A.Compose(
            [A.Resize(height, width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
             A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise(),
             A.Normalize(normalization="min_max", p=1.0)], is_check_shapes=False)

        t_val = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST),
                           A.Normalize(normalization="min_max", p=1.0)], is_check_shapes=False)
        train_set = UterusDataset(df.loc[X_train].reset_index(), t_train)
        val_set = UterusDataset(df.loc[X_val].reset_index(), t_val)

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                        steps_per_epoch=len(train_loader),
                                                        pct_start=0.2)

        fit(_run, _neptune_run, epochs, model, train_loader, val_loader, eval(get_losses()), optimizer, scheduler,
            best_iou_scores, best_nsd_scores, best_dice_scores, i, model_output)

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
    
    # Run experiments with command-line arguments
    ex.run(config_updates={
        'losses': "[smp.losses.TverskyLoss('binary')]",
        'encoder_name': 'inceptionresnetv2',
        'model_name': 'MAnet',
        'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1},
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'model_output': args.model_output,
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'dataset_name': args.dataset_name,
    })
