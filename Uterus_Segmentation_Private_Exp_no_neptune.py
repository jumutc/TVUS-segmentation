import argparse
import gc
import json
import os
import ssl
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from MeshMetrics import DistanceMetrics
from sacred import Experiment
from sacred.observers import FileStorageObserver
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
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory (contains videos/, labels/, metadata_labels.json)')
    parser.add_argument('--control_balance_ratio', type=float, default=0.3,
                        help='Ratio of control (negative) frames to add for balancing (default: 0.3)')
    parser.add_argument('--model_output', type=str, default='model_tvus.pt',
                        help='Path to save the trained model (default: model_tvus.pt)')
    parser.add_argument('--csv_output', type=str, default='input.csv',
                        help='Path to save the input CSV file (default: input.csv)')
    parser.add_argument('--sacred_runs', type=str, default='uterus_runs',
                        help='Path to Sacred runs directory (default: uterus_runs)')
    parser.add_argument('--dataset_name', type=str, default='TVUS (private)',
                        help='Dataset name for logging (default: TVUS (private))')
    return parser.parse_args()

def _is_control_folder(folder_name):
    """Folders with 'control' in name contain videos without labels (negative samples)."""
    return 'control' in folder_name.lower()


def _collect_video_paths(videos_root, include_control=False):
    """Collect video path by filename. Excludes control folders unless include_control=True."""
    video_by_name = {}
    for folder in os.listdir(videos_root):
        folder_path = os.path.join(videos_root, folder)
        if not os.path.isdir(folder_path):
            continue
        if _is_control_folder(folder) and not include_control:
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.mp4', '.avi', '.mov')):
                video_by_name[fname] = os.path.join(folder_path, fname)
    return video_by_name


def _polygon_to_mask(polygon_data, height, width):
    """Convert Encord polygon (normalized 0-1) to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon_data:
        return mask
    # polygon can be dict like {"0": {"x": 0.4, "y": 0.2}, ...} or polygons array
    pts = []
    if isinstance(polygon_data, dict):
        keys = sorted(polygon_data.keys(), key=lambda k: int(k) if k.isdigit() else k)
        for k in keys:
            pt = polygon_data[k]
            x = int(pt['x'] * width)
            y = int(pt['y'] * height)
            pts.append([x, y])
    elif isinstance(polygon_data, list) and len(polygon_data) > 0:
        # polygons: [[[x,y,x,y,...]]] - flattened xy pairs
        flat = polygon_data[0][0] if isinstance(polygon_data[0][0], (list, tuple)) else polygon_data[0]
        for i in range(0, len(flat), 2):
            x, y = int(flat[i] * width), int(flat[i + 1] * height)
            pts.append([x, y])
    if len(pts) >= 3:
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _create_mask_from_encord_objects(objects, height, width):
    """Create binary mask from Encord frame objects (polygon annotations)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for obj in objects:
        if obj.get('shape') != 'polygon':
            continue
        poly = obj.get('polygon')
        if isinstance(poly, dict):
            single = _polygon_to_mask(poly, height, width)
            mask = np.maximum(mask, single)
        elif obj.get('polygons'):
            # polygons: [[[x,y,x,y,...]]] or [[x,y,x,y,...]]
            for poly_contour in obj['polygons']:
                flat = poly_contour[0] if poly_contour and isinstance(poly_contour[0], (list, tuple)) else poly_contour
                if not isinstance(flat, (list, tuple)) or len(flat) < 6:
                    continue
                pts = []
                for i in range(0, len(flat), 2):
                    pts.append([int(flat[i] * width), int(flat[i + 1] * height)])
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask


def create_df(data_path, control_balance_ratio=0.3):
    """
    Build dataframe from new data format:
    - data/metadata_labels.json: label_hash -> video filename
    - data/labels/{label_hash}.json: Encord format, frame -> polygon objects
    - data/videos/: videos; folders with 'control' excluded from labels, used for balancing
    """
    metadata_path = os.path.join(data_path, 'metadata_labels.json')
    labels_dir = os.path.join(data_path, 'labels')
    videos_dir = os.path.join(data_path, 'videos')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # label_hash -> video filename (only entries with title)
    label_to_video = {m['label_hash']: m['title'] for m in metadata if m.get('title')}

    # Video path resolution: labeled videos (exclude control), control videos (for balancing)
    labeled_videos = _collect_video_paths(videos_dir, include_control=False)
    control_videos = {}
    for folder in os.listdir(videos_dir):
        folder_path = os.path.join(videos_dir, folder)
        if not os.path.isdir(folder_path) or not _is_control_folder(folder):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.mp4', '.avi', '.mov')):
                control_videos[fname] = os.path.join(folder_path, fname)

    rows = []
    for label_hash, video_filename in tqdm(label_to_video.items(), desc='Loading labeled data'):
        video_path = labeled_videos.get(video_filename)
        if not video_path or not os.path.isfile(video_path):
            continue
        label_path = os.path.join(labels_dir, f'{label_hash}.json')
        if not os.path.isfile(label_path):
            continue
        with open(label_path) as f:
            label_data = json.load(f)
        if not isinstance(label_data, dict):
            continue

        cap = cv2.VideoCapture(video_path)
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        volume_id = os.path.splitext(video_filename)[0]
        for frame_key, frame_data in label_data.items():
            if not frame_key.isdigit():
                continue
            frame_idx = int(frame_key)
            objs = frame_data.get('objects', [])
            if not objs:
                continue
            mask = _create_mask_from_encord_objects(objs, vh, vw)
            rows.append({
                'video_path': video_path,
                'frame_idx': frame_idx,
                'seg': mask,
                'volume_id': volume_id,
            })

    # Add control frames for balancing (negative samples)
    n_labeled = len(rows)
    n_control_target = max(0, int(n_labeled * control_balance_ratio))
    print('Total Labeled Images: ', n_labeled)

    if n_control_target > 0 and control_videos:
        control_paths = list(control_videos.values())
        added = 0
        for video_path in control_paths:
            if added >= n_control_target:
                break
            cap = cv2.VideoCapture(video_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            if n_frames <= 0:
                continue
            volume_id = os.path.splitext(os.path.basename(video_path))[0]
            # Sample frames uniformly
            indices = np.linspace(0, n_frames - 1, min(n_control_target - added, max(1, n_frames // 5)), dtype=int)
            for fi in indices:
                if added >= n_control_target:
                    break
                rows.append({
                    'video_path': video_path,
                    'frame_idx': int(fi),
                    'seg': np.zeros((vh, vw), dtype=np.uint8),
                    'volume_id': volume_id,
                })
                added += 1

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


class TVUSDataset(Dataset):
    """Dataset that loads frames from videos and uses precomputed Encord masks."""

    def __init__(self, X, transform):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.loc[idx]
        video_path = row['video_path']
        frame_idx = int(row['frame_idx'])
        mask_1 = row['seg']

        img = _read_frame_from_video(video_path, frame_idx)

        mask = np.zeros(list(mask_1.shape) + [1])
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


def fit(_run, epochs, model, train_loader, val_loader, losses, optimizer, scheduler, best_iou_scores, best_nsd_scores, best_dice_scores, fold, model_output_path, val_df):
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
    data_path = ''
    control_balance_ratio = 0.3
    model_output = 'model_tvus.pt'
    csv_output = 'input.csv'
    sacred_runs = 'uterus_runs'
    dataset_name = 'Niches (private)'
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
def run_experiment(_run, data_path, control_balance_ratio, model_output, csv_output):
    max_lr = 1e-4
    epochs = 200
    weight_decay = 1e-4
    best_iou_scores = {}
    best_nsd_scores = {}
    best_dice_scores = {}
    height, width = 512, 768

    # Create dataframe from new data format (videos, labels, metadata)
    df = create_df(data_path, control_balance_ratio)
    print('Total Images: ', len(df))
    print(df.head())
    df[['volume_id', 'video_path', 'frame_idx']].to_csv(csv_output, index=False)

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
            + augmentation_list, is_check_shapes=False)
        t_val = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_LINEAR)], is_check_shapes=False)

        train_df = df.loc[X_train].reset_index()
        train_set = TVUSDataset(train_df, t_train)
        val_df = df.loc[X_val].reset_index()
        val_set = TVUSDataset(val_df, t_val)

        train_loader = DataLoader(train_set, batch_size=2, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=False)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                        steps_per_epoch=len(train_loader),
                                                        pct_start=0.2)

        fit(_run, epochs, model, train_loader, val_loader, eval(get_losses()), optimizer, scheduler,
            best_iou_scores, best_nsd_scores, best_dice_scores, i, model_output, val_df)

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
        'data_path': args.data_path,
        'control_balance_ratio': args.control_balance_ratio,
        'model_output': get_model_output_path(args.model_output, 'DeepLabV3Plus', 'efficientnet-b7', False),
        'csv_output': args.csv_output,
        'sacred_runs': args.sacred_runs,
        'dataset_name': args.dataset_name,
    })
    # ex.run(config_updates={
    #     'losses': "[smp.losses.TverskyLoss('binary')]",
    #     'encoder_name': 'efficientnet-b7',
    #     'model_name': 'DeepLabV3Plus',
    #     'model_params': {'encoder_weights': 'imagenet', 'decoder_channels': 256, 'activation': None, 'classes': 1},
    #     'augmentations': '[A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise()]',
    #     'data_path': args.data_path,
    #     'control_balance_ratio': args.control_balance_ratio,
    #     'model_output': get_model_output_path(args.model_output, 'DeepLabV3Plus', 'efficientnet-b7', True),
    #     'csv_output': args.csv_output,
    #     'sacred_runs': args.sacred_runs,
    #     'dataset_name': args.dataset_name,
    # })
    # ex.run(config_updates={
    #     'losses': "[smp.losses.TverskyLoss('binary')]",
    #     'encoder_name': 'inceptionresnetv2',
    #     'model_name': 'MAnet',
    #     'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
    #                      'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32)},
    #     'augmentations': '[]',
    #     'data_path': args.data_path,
    #     'control_balance_ratio': args.control_balance_ratio,
    #     'model_output': get_model_output_path(args.model_output, 'MAnet', 'inceptionresnetv2', False),
    #     'csv_output': args.csv_output,
    #     'sacred_runs': args.sacred_runs,
    #     'dataset_name': args.dataset_name,
    # })
    # ex.run(config_updates={
    #     'losses': "[smp.losses.TverskyLoss('binary')]",
    #     'encoder_name': 'inceptionresnetv2',
    #     'model_name': 'MAnet',
    #     'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
    #                      'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32)},
    #     'augmentations': '[A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise()]',
    #     'data_path': args.data_path,
    #     'control_balance_ratio': args.control_balance_ratio,
    #     'model_output': get_model_output_path(args.model_output, 'MAnet', 'inceptionresnetv2', True),
    #     'csv_output': args.csv_output,
    #     'sacred_runs': args.sacred_runs,
    #     'dataset_name': args.dataset_name,
    # })
    # ex.run(config_updates={
    #     'losses': "[smp.losses.TverskyLoss('binary')]",
    #     'encoder_name': 'inceptionresnetv2',
    #     'model_name': 'UnetPlusPlus',
    #     'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
    #                      'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32)},
    #     'augmentations': '[]',
    #     'data_path': args.data_path,
    #     'control_balance_ratio': args.control_balance_ratio,
    #     'model_output': get_model_output_path(args.model_output, 'UnetPlusPlus', 'inceptionresnetv2', False),
    #     'csv_output': args.csv_output,
    #     'sacred_runs': args.sacred_runs,
    #     'dataset_name': args.dataset_name,
    # })
    # ex.run(config_updates={
    #     'losses': "[smp.losses.TverskyLoss('binary')]",
    #     'encoder_name': 'inceptionresnetv2',
    #     'model_name': 'UnetPlusPlus',
    #     'model_params': {'encoder_weights': 'imagenet+background', 'activation': None, 'classes': 1,
    #                      'encoder_depth': 5, 'decoder_channels': (512, 256, 128, 64, 32)},
    #     'augmentations': '[A.Rotate(p=0.2), A.MotionBlur(), A.ZoomBlur(), A.Defocus(), A.GaussNoise()]',
    #     'data_path': args.data_path,
    #     'control_balance_ratio': args.control_balance_ratio,
    #     'model_output': get_model_output_path(args.model_output, 'UnetPlusPlus', 'inceptionresnetv2', True),
    #     'csv_output': args.csv_output,
    #     'sacred_runs': args.sacred_runs,
    #     'dataset_name': args.dataset_name,
    # })
