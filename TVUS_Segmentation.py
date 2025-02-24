import gc
import glob
import os
import time

import albumentations as A
import cv2
import neptune
import nibabel as nib
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ex = Experiment("tvus_exp")
ex.observers.append(FileStorageObserver('tvus_runs'))

IMAGE_PATH = 'data/dataset/annotated_volumes'
SEG_PATH = 'data/dataset/annotations'


def create_df():
    image_paths = sorted(glob.glob(f"{IMAGE_PATH}/*.nii.gz"))
    images, segmentations, volume_ids = [], [], []

    for image_path in tqdm(image_paths):
        volume_id = os.path.basename(image_path)
        nii_img = nib.load(image_path)
        nii_seg = nib.load(os.path.join(SEG_PATH, volume_id))
        nii_img_data = nii_img.get_fdata()
        nii_seg_data = nii_seg.get_fdata()

        for i in range(0, nii_img_data.shape[-1]):
            images.append(nii_img_data[..., i:i + 1])
            segmentations.append(nii_seg_data[..., i:i + 1])
            volume_ids.append(volume_id)

    return pd.DataFrame({'img': images, 'seg': segmentations, 'volume_id': volume_ids},
                        index=np.arange(0, len(images)))


df = create_df()
print('Total Images: ', len(df))


class GlendaDataset(Dataset):
    def __init__(self, X, transform, second_transforms):
        self.X = X
        self.transform = transform
        self.second_transforms = second_transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = np.repeat(self.X.loc[idx]['img'], 3, axis=-1)
        mask = self.X.loc[idx]['seg']

        aug = self.transform(image=img, mask=mask)

        img = aug['image']
        mask = aug['mask']

        if self.second_transforms:
            masks = [second_transform(image=img, mask=mask)['mask'] for second_transform in self.second_transforms]
            masks = [torch.from_numpy(np.transpose(m, axes=(2, 0, 1))).round().float() for m in masks]
        else:
            masks = []

        mask = torch.from_numpy(np.transpose(mask, axes=(2, 0, 1))).round().float()
        t = T.Compose([T.ToTensor()])
        img = t(img).float()

        return img, mask, masks


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(_run, _neptune_run, epochs, model, train_loader, val_loader, losses, optimizer, scheduler, best_iou_scores,
        fold):
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_iou = -np.inf
    decrease = 1
    not_improve = 100

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()

        since = time.time()
        running_loss = 0
        iou_scores = []
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image, mask, masks = data

            image = image.to(device)
            mask = mask.to(device)
            masks = [m.to(device) for m in masks]

            # forward
            output = model(image)
            loss = losses[0](output, mask)

            if len(losses) > 1:
                encoder_output = model.encoder(image)

                for j in range(0, len(masks)):
                    # please adapt this part depending on the decoder being used
                    encoder_output_mean = encoder_output[j + 1].mean(dim=1, keepdim=True)
                    loss += losses[-1](encoder_output_mean, masks[j]).mean()

            # evaluation metrics
            tp, fp, fn, tn = smp.metrics.get_stats(F.sigmoid(output), mask.long(), mode='binary', threshold=0.5)

            iou_scores.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction=None).cpu())
            accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro').cpu()

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
            test_accuracy = 0
            val_iou_scores = []
            images, gts, outputs = [], [], []
            tps, fps, fns, tns = [], [], [], []

            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles, _ = data

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)

                    images.append(image.cpu())
                    outputs.append(output.cpu())
                    gts.append(mask.cpu())

                    # evaluation metrics
                    tp, fp, fn, tn = smp.metrics.get_stats(F.sigmoid(output), mask.long(), mode='binary',
                                                           threshold=0.5)
                    tps.append(tp)
                    fps.append(fp)
                    fns.append(fn)
                    tns.append(tn)

                    val_iou_scores.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction=None).cpu())
                    test_accuracy += smp.metrics.accuracy(tp, fp, fn, tn, reduction='micro').cpu()

                    loss = losses[0](output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))
            test_iou_scores = np.vstack(val_iou_scores)
            test_iou_score = np.nanmean(test_iou_scores)
            train_iou_score = np.nanmean(np.vstack(iou_scores))

            if min_iou < test_iou_score:
                print('Validation IoU increasing.. {:.3f} >> {:.3f}, per class >> {:s}'.format(min_iou, test_iou_score,
                                                                                               str(np.nanmean(
                                                                                                   test_iou_scores,
                                                                                                   axis=0))))
                best_iou_scores[fold] = test_iou_scores
                min_iou = test_iou_score
                decrease += 1
                not_improve = 0
                print('saving model...')
                torch.save(model, 'model.pt')

            if test_iou_score < min_iou:
                not_improve += 1
                print(f'IoU not increased for {not_improve} time')
                if not_improve == 100:
                    print('IoU not increased for 100 times, Stop Training')
                    break

            # iou
            val_iou.append(test_iou_score)
            train_iou.append(train_iou_score)
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))

            _neptune_run[f"training.{fold}.lr"].append(get_lr(optimizer))
            _neptune_run[f"training.{fold}.loss"].append(float(running_loss / len(train_loader)))
            _neptune_run[f"validation.{fold}.loss"].append(float(test_loss / len(val_loader)))
            _neptune_run[f"training.{fold}.mIoU"].append(float(train_iou[-1]))
            _neptune_run[f"validation.{fold}.mIoU"].append(float(val_iou[-1]))
            _neptune_run[f"training.{fold}.accuracy"].append(float(accuracy / len(train_loader)))
            _neptune_run[f"validation.{fold}.accuracy"].append(float(test_accuracy / len(val_loader)))

            _run.log_scalar(f"training.{fold}.loss", float(running_loss / len(train_loader)))
            _run.log_scalar(f"validation.{fold}.loss", float(test_loss / len(val_loader)))
            _run.log_scalar(f"training.{fold}.mIoU", float(train_iou[-1]))
            _run.log_scalar(f"validation.{fold}.mIoU", float(val_iou[-1]))
            _run.log_scalar(f"training.{fold}.accuracy", float(accuracy / len(train_loader)))
            _run.log_scalar(f"validation.{fold}.accuracy", float(test_accuracy / len(val_loader)))

            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(train_iou[-1]),
                  "Val mIoU: {:.3f}..".format(val_iou[-1]),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


@ex.config
def config():
    losses = []
    encoder_name = ''


@ex.capture
def get_losses(losses):
    return losses


@ex.capture
def get_encoder_name(encoder_name):
    return encoder_name


@ex.main
def run_experiment(_run):
    max_lr = 1e-4
    epochs = 200
    weight_decay = 1e-4
    best_iou_scores = {}
    height, width = 224, 192

    _neptune_run = neptune.init_run(
        project="{your_project_here}",
        api_token="{your_token_here}",
    )
    _neptune_run['parameters'] = {
        'max_lr': max_lr,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'img_height': height,
        'img_width': width,
        'encoder': get_encoder_name(),
        'decoder': 'MAnet',
        'dataset': 'public',
        'losses': get_losses()
    }

    for i, (X_train, X_val) in enumerate(
            GroupShuffleSplit(n_splits=3, test_size=0.15, random_state=0).split(df.index, groups=df['volume_id'])):
        print('Train Size   : ', len(X_train))
        print('Val Size     : ', len(X_val))

        if set(df.loc[X_train]['volume_id'].values) & set(df.loc[X_val]['volume_id'].values):
            raise ValueError('Intersecting validation and train groups detected!')

        # model = smp.DeepLabV3Plus(get_encoder_name(), encoder_weights='imagenet', decoder_channels=60, activation=None, classes=1)
        model = smp.MAnet(get_encoder_name(), encoder_weights='imagenet', activation=None, classes=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)

        t_train = A.Compose(
            [A.Resize(height, width, interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(),
             A.Rotate(p=0.2)], is_check_shapes=False)

        t_val = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST)], is_check_shapes=False)
        t_mask = [
            A.Compose([A.Resize(height // 2, width // 2, interpolation=cv2.INTER_NEAREST)], is_check_shapes=False),
            A.Compose([A.Resize(height // 4, width // 4, interpolation=cv2.INTER_NEAREST)], is_check_shapes=False),
            A.Compose([A.Resize(height // 8, width // 8, interpolation=cv2.INTER_NEAREST)], is_check_shapes=False),
            A.Compose([A.Resize(height // 16, width // 16, interpolation=cv2.INTER_NEAREST)], is_check_shapes=False)
        ]

        train_set = GlendaDataset(df.loc[X_train].reset_index(), t_train, t_mask)
        val_set = GlendaDataset(df.loc[X_val].reset_index(), t_val, None)

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                        steps_per_epoch=len(train_loader),
                                                        pct_start=0.2)

        fit(_run, _neptune_run, epochs, model, train_loader, val_loader, eval(get_losses()), optimizer, scheduler,
            best_iou_scores, i)

    best_iou_scores = np.vstack(best_iou_scores.values())
    print(f"TOTAL AVERAGE CV mIOU: {np.nanmean(best_iou_scores)}")

    _neptune_run["average.mIoU"] = float(np.nanmean(best_iou_scores))
    _run.log_scalar("average.mIoU", float(np.nanmean(best_iou_scores)))

    _neptune_run.stop()


ex.run(config_updates={
    'losses': "[smp.losses.TverskyLoss('binary')]",
    'encoder_name': 'efficientnet-b1'
})
ex.run(config_updates={
    'losses': "[smp.losses.TverskyLoss('binary'), smp.losses.TverskyLoss('binary')]",
    'encoder_name': 'efficientnet-b1'
})
ex.run(config_updates={
    'losses': "[smp.losses.TverskyLoss('binary'), smp.losses.FocalLoss('binary')]",
    'encoder_name': 'efficientnet-b1'
})
