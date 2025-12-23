import argparse
import gc
import glob
import os
import ssl
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
    parser = argparse.ArgumentParser(description='Uterus Segmentation Private Experiment')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to directory containing image volumes')
    parser.add_argument('--seg_path', type=str, required=True,
                        help='Path to directory containing segmentation masks')
    parser.add_argument('--model_output', type=str, default='model_tvus.pt',
                        help='Path to save the trained model (default: model_tvus.pt)')
    parser.add_argument('--csv_output', type=str, default='input.csv',
                        help='Path to save the input CSV file (default: input.csv)')
    parser.add_argument('--sacred_runs', type=str, default='uterus_runs',
                        help='Path to Sacred runs directory (default: uterus_runs)')
    parser.add_argument('--neptune_project', type=str, default='jumutc/uterus',
                        help='Neptune project name (default: jumutc/uterus)')
    parser.add_argument('--dataset_name', type=str, default='TVUS (private)',
                        help='Dataset name for logging (default: TVUS (private))')
    return parser.parse_args()

def find_in_paths(p, image_paths):
    filename = os.path.basename(p).split('_')[0]
    in_paths = [_p for _p in image_paths if filename in _p]
    return in_paths[0] if in_paths else p


def create_df(image_path, seg_path):
    images, segmentations, volume_ids, img_paths, seg_paths = [], [], [], [], []

    for volume_id in tqdm(os.listdir(image_path)):
        image_paths = sorted(glob.glob(os.path.join(image_path, volume_id, '*', volume_id + '*')))
        preprocessed_paths = sorted(glob.glob(os.path.join(seg_path, volume_id, '*', volume_id + '*_preprocessed*')))
        seg_masks = sorted(glob.glob(os.path.join(seg_path, volume_id, '*', 'masked_' + volume_id + '*')))
        preprocessed_paths = [find_in_paths(p, image_paths) for p in preprocessed_paths]

        for _image_path, _seg_path in zip(preprocessed_paths, seg_masks):
            images.append(cv2.cvtColor(cv2.imread(_image_path), cv2.COLOR_BGR2RGB))
            segmentations.append(cv2.imread(_seg_path, cv2.IMREAD_GRAYSCALE))
            volume_ids.append(volume_id)
            img_paths.append(_image_path)
            seg_paths.append(_seg_path)

    return pd.DataFrame({'img': images, 'seg': segmentations, 'volume_id': volume_ids, 'img_path': img_paths, 'seg_path': seg_paths},
                        index=np.arange(0, len(images)))


class TVUSDataset(Dataset):
    def __init__(self, X, transform):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X.loc[idx]['img']
        mask_1 = self.X.loc[idx]['seg']

        # Ensure consistent dtype for OpenCV compatibility (float32 for albumentations)
        img = img.astype(np.float32)

        mask = np.zeros(list(mask_1.shape) + [1], dtype=np.float32)
        mask[mask_1 > 0, 0] = 1

        aug = self.transform(image=img, mask=mask)

        img = aug['image']
        mask = aug['mask']

        mask = torch.from_numpy(np.transpose(mask, axes=(2, 0, 1))).round().float()
        t = T.Compose([T.ToTensor()])
        img = t(img).float()

        return img, mask, idx


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(_run, _neptune_run, epochs, model, train_loader, val_loader, losses, optimizer, scheduler, best_iou_scores, best_nsd_scores, best_dice_scores, fold, model_output_path, val_df):
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
            image, mask, _ = data

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
                    image, mask, idx = data

                    image = image.to(device)
                    output = model(image).cpu()
                    out = np.squeeze(output.detach().numpy(), axis=(0, 1))
                    
                    # Convert idx to scalar if it's a tensor/array
                    if isinstance(idx, torch.Tensor):
                        idx = idx.item()
                    elif isinstance(idx, (np.ndarray, list)):
                        idx = idx[0] if len(idx) > 0 else idx
                    
                    gt = val_df.loc[idx]['seg']
                    gt = np.squeeze(gt)
                    
                    # Resize prediction to match GT if needed
                    if out.shape != gt.shape:
                        out = cv2.resize(out, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

                    # Binarize masks
                    gt_binary = (gt > 0).astype(bool)
                    out_binary = (out > 0).astype(bool)

                    # Compute metrics
                    metrics.set_input(out_binary, gt_binary, spacing=(1, 1))
                    val_iou_scores.append(metrics.iou())
                    val_nsd_scores.append(metrics.nsd(6))
                    val_dice_scores.append(metrics.dsc())

                    mask = mask.cpu()
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
    model_name = 'DeepLabV3Plus'
    model_params = {'encoder_weights': 'imagenet', 'decoder_channels': 60, 'activation': None, 'classes': 1}
    image_path = ''
    seg_path = ''
    model_output = 'model_tvus.pt'
    csv_output = 'input.csv'
    sacred_runs = 'uterus_runs'
    neptune_project = 'jumutc/uterus'
    dataset_name = 'TVUS (private)'
    augmentations = '[A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise()]'


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
def get_augmentations(augmentations):
    return augmentations


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
    height, width = 512, 768
    # height, width = 768, 1024

    # Create dataframe
    df = create_df(image_path, seg_path)
    print('Total Images: ', len(df))
    print(df.head())
    df[['volume_id', 'img_path', 'seg_path']].to_csv(csv_output, index=False)

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
        'losses': get_losses(),
        'augmentations': get_augmentations(),
        'dataset': 'private'
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

        augmentation_list = eval(get_augmentations())
        t_train = A.Compose(
            [A.Resize(height, width, interpolation=cv2.INTER_LINEAR), A.HorizontalFlip(), A.VerticalFlip()]
            + augmentation_list
            + [A.Normalize(normalization="min_max", p=1.0)], is_check_shapes=False)

        t_val = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_LINEAR),
                           A.Normalize(normalization="min_max", p=1.0)], is_check_shapes=False)
        train_df = df.loc[X_train].reset_index()
        train_set = TVUSDataset(train_df, t_train)
        val_df = df.loc[X_val].reset_index()
        val_set = TVUSDataset(val_df, t_val)

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                        steps_per_epoch=len(train_loader),
                                                        pct_start=0.2)

        fit(_run, _neptune_run, epochs, model, train_loader, val_loader, eval(get_losses()), optimizer, scheduler,
            best_iou_scores, best_nsd_scores, best_dice_scores, i, model_output, val_df)

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


def get_model_output_path(base_path, model_name, encoder_name, has_aug):
    """Generate a unique model output path with postfix."""
    base_name, ext = os.path.splitext(base_path)
    aug_suffix = '_aug' if has_aug else '_noaug'
    postfix = f"_{model_name}_{encoder_name}_{aug_suffix}"
    return f"{base_name}{postfix}{ext}"


if __name__ == '__main__':
    args = parse_args()
    
    # Set up Sacred observer
    ex.observers.append(FileStorageObserver(args.sacred_runs))
    
    # Run experiments with command-line arguments
    ex.run(config_updates={
        'losses': "[smp.losses.TverskyLoss('binary')]",
        'encoder_name': 'efficientnet-b7',
        'model_name': 'DeepLabV3Plus',
        'model_params': {'encoder_weights': 'imagenet', 'decoder_channels': 256, 'activation': None, 'classes': 1},
        'augmentations': '[]',
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'model_output': get_model_output_path(args.model_output, 'DeepLabV3Plus', 'efficientnet-b7', False),
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'dataset_name': args.dataset_name,
    })
    ex.run(config_updates={
        'losses': "[smp.losses.TverskyLoss('binary')]",
        'encoder_name': 'efficientnet-b7',
        'model_name': 'DeepLabV3Plus',
        'model_params': {'encoder_weights': 'imagenet', 'decoder_channels': 256, 'activation': None, 'classes': 1},
        'augmentations': '[A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise()]',
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'model_output': get_model_output_path(args.model_output, 'DeepLabV3Plus', 'efficientnet-b7', True),
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'dataset_name': args.dataset_name,
    })
    ex.run(config_updates={
        'losses': "[smp.losses.TverskyLoss('binary')]",
        'encoder_name': 'inceptionresnetv2',
        'model_name': 'MAnet',
        'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
                         'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32)},
        'augmentations': '[]',
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'model_output': get_model_output_path(args.model_output, 'MAnet', 'inceptionresnetv2', False),
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'dataset_name': args.dataset_name,
    })
    ex.run(config_updates={
        'losses': "[smp.losses.TverskyLoss('binary')]",
        'encoder_name': 'inceptionresnetv2',
        'model_name': 'MAnet',
        'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
                         'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32)},
        'augmentations': '[A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise()]',
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'model_output': get_model_output_path(args.model_output, 'MAnet', 'inceptionresnetv2', True),
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'dataset_name': args.dataset_name,
    })
    ex.run(config_updates={
        'losses': "[smp.losses.TverskyLoss('binary')]",
        'encoder_name': 'inceptionresnetv2',
        'model_name': 'UnetPlusPlus',
        'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
                         'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32), 'in_channels': 1},
        'augmentations': '[]',
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'model_output': get_model_output_path(args.model_output, 'UnetPlusPlus', 'inceptionresnetv2', False),
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'dataset_name': args.dataset_name,
    })
    ex.run(config_updates={
        'losses': "[smp.losses.TverskyLoss('binary')]",
        'encoder_name': 'inceptionresnetv2',
        'model_name': 'UnetPlusPlus',
        'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
                         'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32), 'in_channels': 1},
        'augmentations': '[A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise()]',
        'image_path': args.image_path,
        'seg_path': args.seg_path,
        'model_output': get_model_output_path(args.model_output, 'UnetPlusPlus', 'inceptionresnetv2', True),
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'neptune_project': args.neptune_project,
        'dataset_name': args.dataset_name,
    })
