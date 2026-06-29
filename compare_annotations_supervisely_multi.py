import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from MeshMetrics import DistanceMetrics

NICHE_CLASS_NAMES = frozenset({'niche'})

CONTOUR_COLORS_BGR = [
    (0, 0, 255),      # red
    (255, 0, 0),      # blue
    (0, 200, 0),      # green
    (0, 200, 255),    # orange
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (128, 0, 255),    # pink
    (255, 128, 0),    # light blue
    (0, 128, 255),    # dark orange
]

CONTOUR_COLOR_NAMES = [
    "red", "blue", "green", "orange", "magenta",
    "cyan", "yellow", "pink", "light-blue", "dark-orange",
]


def _sanitize_path_component(name):
    return re.sub(r'[^\w\-.]', '_', name)


def load_blacklist(parent_folder, filename='blacklist.txt'):
    blacklist_path = Path(parent_folder) / filename
    if not blacklist_path.exists():
        return set()

    blacklist = set()
    with open(blacklist_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                blacklist.add(line)
    return blacklist


def video_stem_to_study_number(video_stem):
    """
    Map a video filename stem to the Study number used in frame_measurements.csv.

    Examples:
      VU0805_1_sag_cropped -> VU805_1
      VU0852_sag_cropped     -> VU0852
      VU1162_sag_1_cropped   -> VU1162_1
    """
    name = video_stem.replace('_sag_cropped', '').replace('_cropped', '')
    name = re.sub(r'_sag_(\d+)', r'_\1', name)
    name = re.sub(r'^VU080(\d)', r'VU80\1', name)
    return name


def load_frame_measurements(parent_folder, filename='frame_measurements.csv'):
    """
    Load per-study NSD tolerance (tau) from frame_measurements.csv.

    Returns:
        Dict mapping study number -> tau (Pixels 1 mm column value)
    """
    csv_path = Path(parent_folder) / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Frame measurements CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    if 'Study number' not in df.columns or 'Pixels 1 mm' not in df.columns:
        raise ValueError(
            f"{csv_path} must contain 'Study number' and 'Pixels 1 mm' columns"
        )

    measurements = {}
    for _, row in df.iterrows():
        study = str(row['Study number']).strip()
        tau = row['Pixels 1 mm']
        if pd.isna(tau):
            print(f"Warning: missing Pixels 1 mm for study {study}, skipping row")
            continue
        measurements[study] = float(tau)
    return measurements


def get_nsd_tau_for_video(video_name, measurements):
    stem = Path(video_name).stem
    study = video_stem_to_study_number(stem)
    if study not in measurements:
        raise KeyError(
            f"No Pixels 1 mm entry for video {video_name} "
            f"(study number {study})"
        )
    return measurements[study]


def load_supervisely_annotation(annotation_path):
    with open(annotation_path, encoding='utf-8') as f:
        return json.load(f)


def _is_niche_class(class_title):
    return (class_title or '').lower() in NICHE_CLASS_NAMES


def get_niche_object_keys(annotation):
    return {
        obj['key']
        for obj in annotation.get('objects', [])
        if _is_niche_class(obj.get('classTitle'))
    }


def _points_to_contour(points):
    if not points:
        return None
    contour = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    if contour.shape[0] < 3:
        return None
    return contour


def supervisely_polygon_to_mask(exterior, interior, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)

    exterior_contour = _points_to_contour(exterior)
    if exterior_contour is None:
        return mask.astype(bool)

    cv2.fillPoly(mask, [exterior_contour], 255)

    for hole in interior or []:
        hole_contour = _points_to_contour(hole)
        if hole_contour is not None:
            cv2.fillPoly(mask, [hole_contour], 0)

    return mask > 0


def get_niche_frame_indices(annotation):
    niche_keys = get_niche_object_keys(annotation)
    if not niche_keys:
        return set()

    frame_indices = set()
    for frame in annotation.get('frames', []):
        for figure in frame.get('figures', []):
            if figure.get('geometryType') != 'polygon':
                continue
            if figure.get('objectKey') not in niche_keys:
                continue
            frame_indices.add(frame['index'])
            break
    return frame_indices


def extract_niche_mask_at_frame(annotation, frame_index):
    height = annotation['size']['height']
    width = annotation['size']['width']
    niche_keys = get_niche_object_keys(annotation)

    mask = np.zeros((height, width), dtype=bool)
    for frame in annotation.get('frames', []):
        if frame['index'] != frame_index:
            continue
        for figure in frame.get('figures', []):
            if figure.get('geometryType') != 'polygon':
                continue
            if figure.get('objectKey') not in niche_keys:
                continue
            geometry = figure.get('geometry', {})
            points = geometry.get('points', {})
            polygon_mask = supervisely_polygon_to_mask(
                points.get('exterior', []),
                points.get('interior', []),
                height,
                width,
            )
            mask |= polygon_mask
    return mask


def read_video_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise ValueError(
            f"Could not read frame {frame_index} from {video_path}"
        )
    return frame


def compute_segmentation_metrics(mask1, mask2, spacing=(1, 1), nsd_tau=10):
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(
            mask2.astype(np.uint8),
            (mask1.shape[1], mask1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    metrics = DistanceMetrics()
    metrics.set_input(mask1, mask2, spacing=spacing)

    return {
        'iou': float(metrics.iou()),
        'nsd': float(metrics.nsd(nsd_tau)),
        'dice': float(metrics.dsc()),
    }


def draw_contours_overlay(base_img, masks, labels, contour_thickness=2):
    overlay = base_img.copy()
    h, w = overlay.shape[:2]

    for idx, (mask, label) in enumerate(zip(masks, labels)):
        color = CONTOUR_COLORS_BGR[idx % len(CONTOUR_COLORS_BGR)]

        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)

    legend_y = 20
    for idx, label in enumerate(labels):
        color = CONTOUR_COLORS_BGR[idx % len(CONTOUR_COLORS_BGR)]
        color_name = CONTOUR_COLOR_NAMES[idx % len(CONTOUR_COLOR_NAMES)]
        text = f"{label} ({color_name})"
        cv2.putText(
            overlay, text, (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
        legend_y += 18

    return overlay


def _video_name_from_annotation_json(json_name):
    if json_name.endswith('.json'):
        return json_name[:-5]
    return json_name


def _resolve_video_path(video_dir, video_name):
    video_path = video_dir / video_name
    if video_path.exists():
        return video_path

    stem = Path(video_name).stem
    for ext in ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'):
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _resolve_comparison_frames(per_expert_frames, frame_strategy):
    """
    Resolve which frame to use per expert for a video.

    Returns a list of comparison specs. Each spec is a dict mapping
    expert name -> frame index. Multiple specs are returned when several
    frames are annotated by every expert (intersection mode).
    """
    common = (
        set.intersection(*per_expert_frames.values())
        if per_expert_frames else set()
    )
    if common:
        return [
            dict.fromkeys(per_expert_frames, frame_index)
            for frame_index in sorted(common)
        ]

    if frame_strategy == 'intersection':
        return []

    all_frames = sorted(
        idx for frames in per_expert_frames.values() for idx in frames
    )
    if not all_frames:
        return []

    median_idx = int(round(float(np.median(all_frames))))
    return [{
        expert: min(frames, key=lambda idx: (abs(idx - median_idx), idx))
        for expert, frames in per_expert_frames.items()
    }]


def find_annotation_groups(parent_folder, frame_strategy='intersection'):
    """
    Discover videos annotated by multiple Supervisely experts.

    Expected layout:
      parent_folder/
        annotators/<expert>/ann/<video>.mp4.json
        video/<video>.mp4

    Returns:
        List of dicts with keys:
          group_name, video_name, frame_index, entries [(expert, ann_path), ...]
    """
    parent_path = Path(parent_folder)
    annotators_dir = parent_path / 'annotators'
    video_dir = parent_path / 'video'

    if not annotators_dir.is_dir():
        raise ValueError(f"Annotators folder not found: {annotators_dir}")
    if not video_dir.is_dir():
        raise ValueError(f"Video folder not found: {video_dir}")

    expert_dirs = sorted(d for d in annotators_dir.iterdir() if d.is_dir())
    if len(expert_dirs) < 2:
        raise ValueError(
            f"Need at least 2 expert folders under {annotators_dir}"
        )

    per_expert_files = {}
    for expert_dir in expert_dirs:
        ann_dir = expert_dir / 'ann'
        if not ann_dir.is_dir():
            print(f"Warning: no ann/ folder for expert {expert_dir.name}, skipping")
            continue
        per_expert_files[expert_dir.name] = {
            f.name: f for f in sorted(ann_dir.glob('*.json'))
        }

    if len(per_expert_files) < 2:
        raise ValueError("Fewer than 2 experts with ann/ folders found")

    common_json_names = set.intersection(
        *(set(files.keys()) for files in per_expert_files.values())
    )

    for expert, files in per_expert_files.items():
        only_here = set(files.keys()) - common_json_names
        if only_here:
            print(
                f"Warning: annotations only in {expert}: "
                f"{sorted(only_here)[:5]}{'...' if len(only_here) > 5 else ''}"
            )

    groups = []
    for json_name in sorted(common_json_names):
        video_name = _video_name_from_annotation_json(json_name)
        per_expert_frames = {}
        per_expert_paths = {}

        for expert, files in per_expert_files.items():
            ann_path = files[json_name]
            annotation = load_supervisely_annotation(ann_path)
            niche_frames = get_niche_frame_indices(annotation)
            if not niche_frames:
                print(
                    f"Warning: no niche polygons in {expert}/{json_name}, skipping video"
                )
                per_expert_frames = None
                break
            per_expert_frames[expert] = niche_frames
            per_expert_paths[expert] = ann_path

        if per_expert_frames is None:
            continue

        comparison_frames = _resolve_comparison_frames(
            per_expert_frames, frame_strategy,
        )
        if not comparison_frames:
            print(
                f"Warning: no common niche frame for {video_name} "
                f"({ {e: sorted(v) for e, v in per_expert_frames.items()} }); skipping"
            )
            continue

        for frame_map in comparison_frames:
            entries = []
            skip_group = False
            for expert in sorted(per_expert_files.keys()):
                ann_path = per_expert_paths[expert]
                actual_frame = frame_map[expert]
                annotation = load_supervisely_annotation(ann_path)
                mask = extract_niche_mask_at_frame(annotation, actual_frame)
                if not np.any(mask):
                    print(
                        f"Warning: empty niche mask for {expert}/{video_name} "
                        f"frame {actual_frame}, skipping group"
                    )
                    skip_group = True
                    break
                entries.append((expert, str(ann_path), actual_frame))

            if skip_group or len(entries) < 2:
                continue

            unique_frames = sorted({frame for frame in frame_map.values()})
            if len(unique_frames) == 1:
                frame_label = str(unique_frames[0])
                group_name = (
                    f"{video_name}#frame{unique_frames[0]}"
                    if len(comparison_frames) > 1
                    else video_name
                )
            else:
                frame_label = ','.join(
                    f"{expert}:{frame}"
                    for expert, frame in sorted(frame_map.items())
                )
                group_name = f"{video_name}#frames({frame_label})"

            groups.append({
                'group_name': group_name,
                'video_name': video_name,
                'frame_index': unique_frames[0],
                'frame_map': frame_map,
                'entries': entries,
            })

    return groups, video_dir


def _save_group_outputs(masks, entries, group, group_idx, tmp_dir, video_dir):
    tmp_dir = Path(tmp_dir)
    subdir_name = (
        f"{group_idx:03d}_{_sanitize_path_component(group['group_name'])}"
    )
    group_dir = tmp_dir / subdir_name
    group_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for i, (expert, ann_path, frame_index) in enumerate(entries):
        labels.append(f"{expert}/frame{frame_index}")
        mask_uint8 = masks[i].astype(np.uint8) * 255
        mask_path = group_dir / f"mask_{i}_{expert}_frame{frame_index}.png"
        cv2.imwrite(str(mask_path), mask_uint8)

    video_path = _resolve_video_path(video_dir, group['video_name'])
    if video_path is not None:
        overlay_frame = group['frame_map'].get(
            group['entries'][0][0], group['frame_index'],
        )
        base_img = read_video_frame(video_path, overlay_frame)
    else:
        h, w = masks[0].shape
        base_img = np.full((h, w, 3), 128, dtype=np.uint8)
        print(f"Warning: video not found for {group['video_name']}, using grey base")

    overlay = draw_contours_overlay(base_img, masks, labels)
    overlay_path = group_dir / "overlay_contours.png"
    cv2.imwrite(str(overlay_path), overlay)

    return str(group_dir)


def _expert_from_annotation_label(label):
    return label.split('/', 1)[0]


def _normalize_annotator_pair(label_a, label_b):
    expert_a = _expert_from_annotation_label(label_a)
    expert_b = _expert_from_annotation_label(label_b)
    if expert_a <= expert_b:
        return expert_a, expert_b
    return expert_b, expert_a


def build_average_results(results_df):
    """
    Build per-annotator-pair and total summary with mean and std per metric.
    """
    metric_cols = ['iou', 'nsd', 'dice']
    working = results_df.copy()
    pair_cols = working.apply(
        lambda row: _normalize_annotator_pair(
            row['annotation_a'], row['annotation_b'],
        ),
        axis=1,
        result_type='expand',
    )
    working['annotator_a'] = pair_cols[0]
    working['annotator_b'] = pair_cols[1]

    summary_rows = []
    for (annotator_a, annotator_b), group in working.groupby(
        ['annotator_a', 'annotator_b'], sort=True,
    ):
        row = {
            'annotator_a': annotator_a,
            'annotator_b': annotator_b,
            'annotator_pair': f"{annotator_a} vs {annotator_b}",
            'n': len(group),
        }
        for metric in metric_cols:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        summary_rows.append(row)

    total_row = {
        'annotator_a': 'TOTAL',
        'annotator_b': 'TOTAL',
        'annotator_pair': 'TOTAL',
        'n': len(working),
    }
    for metric in metric_cols:
        total_row[f'{metric}_mean'] = working[metric].mean()
        total_row[f'{metric}_std'] = (
            working[metric].std(ddof=1) if len(working) > 1 else 0.0
        )
    summary_rows.append(total_row)

    column_order = [
        'annotator_pair', 'annotator_a', 'annotator_b', 'n',
        'iou_mean', 'iou_std', 'nsd_mean', 'nsd_std', 'dice_mean', 'dice_std',
    ]
    return pd.DataFrame(summary_rows)[column_order]


def _average_output_path(output_path):
    path = Path(output_path)
    return path.with_name(f"{path.stem}_avg{path.suffix}")


def _print_summary_block(summary_df):
    print('=' * 60)
    for _, row in summary_df.iterrows():
        print(
            f"{row['annotator_pair']:>24}  (n={int(row['n']):>3})  "
            f"IoU={row['iou_mean']:.4f} +/- {row['iou_std']:.4f}  "
            f"NSD={row['nsd_mean']:.4f} +/- {row['nsd_std']:.4f}  "
            f"Dice={row['dice_mean']:.4f} +/- {row['dice_std']:.4f}"
        )
    print('=' * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Pairwise comparison of Supervisely video polygon annotations '
                    'from multiple experts (niche class only)'
    )
    parser.add_argument(
        '--parent-folder', type=str, required=True,
        help='Root folder containing annotators/ and video/ subfolders',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save results CSV (default: print to stdout)',
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Print detailed information for each group',
    )
    parser.add_argument(
        '--save-masks', action='store_true', dest='save_masks',
        help='Save extracted masks and overlay images',
    )
    parser.add_argument(
        '--no-save-masks', action='store_false', dest='save_masks',
        help='Do not save masks or overlay images',
    )
    parser.set_defaults(save_masks=True)
    parser.add_argument(
        '--tmp-dir', type=str, default=None,
        help='Directory for saved masks/overlays '
             '(default: tmp/compare_annotations_supervisely_multi)',
    )
    parser.add_argument(
        '--frame-strategy', type=str, default='intersection',
        choices=['intersection', 'median'],
        help='How to pick the frame when experts annotated different indices '
             '(default: intersection — skip if none)',
    )
    parser.add_argument(
        '--measurements-csv', type=str, default='frame_measurements.csv',
        help='CSV with per-study NSD tau in the "Pixels 1 mm" column '
             '(default: frame_measurements.csv in parent folder)',
    )

    args = parser.parse_args()

    print(f"Scanning parent folder: {args.parent_folder}")
    frame_measurements = load_frame_measurements(
        args.parent_folder, filename=args.measurements_csv,
    )
    print(
        f"Loaded NSD tau for {len(frame_measurements)} studies "
        f"from {args.measurements_csv}"
    )
    groups, video_dir = find_annotation_groups(
        args.parent_folder, frame_strategy=args.frame_strategy,
    )

    if not groups:
        print("No annotation groups found!")
        sys.exit(1)

    blacklist = load_blacklist(args.parent_folder)
    if blacklist:
        print(f"Blacklist loaded: {len(blacklist)} file(s) excluded")
        original_count = len(groups)
        groups = [
            g for g in groups
            if Path(g['video_name']).name not in blacklist
            and f"{Path(g['video_name']).name}.json" not in blacklist
        ]
        skipped = original_count - len(groups)
        if skipped:
            print(f"Skipped {skipped} group(s) due to blacklist")

    if not groups:
        print("No annotation groups remaining after applying blacklist!")
        sys.exit(1)

    total_entries = sum(len(g['entries']) for g in groups)
    print(
        f"Found {len(groups)} group(s) with {total_entries} total "
        f"expert annotations (class: niche -> index 0)"
    )

    tmp_dir = (
        Path(args.tmp_dir)
        if args.tmp_dir
        else Path("tmp") / "compare_annotations_supervisely_multi"
    )
    if args.save_masks:
        tmp_dir = tmp_dir.resolve()
        print(f"Masks and overlays will be saved to: {tmp_dir}")

    all_pairwise = []
    results = []

    for group_idx, group in enumerate(groups):
        try:
            entries = group['entries']
            n = len(entries)
            nsd_tau = get_nsd_tau_for_video(
                group['video_name'], frame_measurements,
            )
            if args.verbose:
                print(
                    f"\nGroup '{group['group_name']}' "
                    f"(frame {group['frame_index']}, {n} experts, "
                    f"NSD tau={nsd_tau:.0f} px):"
                )
                for expert, ann_path, frame_index in entries:
                    print(f"  - {expert}: frame {frame_index} ({ann_path})")

            masks = []
            for expert, ann_path, frame_index in entries:
                annotation = load_supervisely_annotation(ann_path)
                mask = extract_niche_mask_at_frame(annotation, frame_index)
                masks.append(mask)
                if args.verbose:
                    print(
                        f"  Mask {expert}@frame{frame_index}: "
                        f"shape={mask.shape}, pixels={np.sum(mask)}"
                    )

            if args.save_masks:
                saved_dir = _save_group_outputs(
                    masks, entries, group, group_idx, tmp_dir, video_dir,
                )
                if args.verbose:
                    print(f"  Saved to: {saved_dir}")

            for (i, j) in combinations(range(n), 2):
                m = compute_segmentation_metrics(
                    masks[i], masks[j], nsd_tau=nsd_tau,
                )
                all_pairwise.append(m)

                expert_i, _, frame_i = entries[i]
                expert_j, _, frame_j = entries[j]

                results.append({
                    'group': group['group_name'],
                    'video': group['video_name'],
                    'frame_index': group['frame_index'],
                    'nsd_tau': nsd_tau,
                    'annotation_a': f"{expert_i}/frame{frame_i}",
                    'annotation_b': f"{expert_j}/frame{frame_j}",
                    'iou': m['iou'],
                    'nsd': m['nsd'],
                    'dice': m['dice'],
                })

                if args.verbose:
                    print(
                        f"  {expert_i}/frame{frame_i} vs {expert_j}/frame{frame_j}: "
                        f"IoU={m['iou']:.4f}  NSD={m['nsd']:.4f}  "
                        f"Dice={m['dice']:.4f}  (tau={nsd_tau:.0f})"
                    )

        except Exception as e:
            print(f"Error processing group '{group['group_name']}': {e}")
            continue

    if not all_pairwise:
        print("No pairs were successfully processed!")
        sys.exit(1)

    df = pd.DataFrame(results)
    avg_df = build_average_results(df)

    print(f"\n{'=' * 60}")
    print(
        f"Aggregated Pairwise Metrics ({len(all_pairwise)} pairs "
        f"across {len(groups)} groups):"
    )
    _print_summary_block(avg_df)

    if args.output:
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")

        avg_output_path = _average_output_path(output_path)
        avg_df.to_csv(avg_output_path, index=False)
        print(f"Average results saved to: {avg_output_path}")
    elif args.verbose:
        print("\nDetailed Results:")
        print(df.to_string(index=False))
        print("\nAverage Results:")
        print(avg_df.to_string(index=False))


if __name__ == "__main__":
    main()
