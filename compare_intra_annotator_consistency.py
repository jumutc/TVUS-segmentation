"""
Intra-annotator consistency analysis for repeated Supervisely annotation sessions.

Expects a parent folder with multiple repeat exports from the same annotator, e.g.:

  annotators_v2/
    1619371_380709_Repeat 1/Repeat 1/dataset .../ann/*.json
    1619371_380709_Repeat 1/Repeat 1/dataset .../video/*.mp4
    1619372_380758_Repeat 2/...
    1619373_380759_Repeat 3/...

For each video, niche polygon masks are compared pairwise across repeats
using Dice, NSD, and IoU — the same metrics as compare_annotations_supervisely_multi.py.

When all repeats share a common annotated frame, all pairwise combinations are
compared (3 pairs for 3 repeats). When they do not, available repeat pairs
with a shared frame are compared instead (e.g. Repeat 2 vs Repeat 3 only).

Additional consistency statistics (std, variance, range, mask-area CV) are
reported per video and in aggregate summaries.
"""

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

from compare_annotations_supervisely_multi import (
    _average_output_path,
    _resolve_video_path,
    _sanitize_path_component,
    _video_name_from_annotation_json,
    compute_segmentation_metrics,
    draw_contours_overlay,
    extract_niche_mask_at_frame,
    get_niche_frame_indices,
    get_nsd_tau_for_video,
    load_frame_measurements,
    load_supervisely_annotation,
    read_video_frame,
    video_stem_to_study_number,
)

METRIC_COLS = ['iou', 'nsd', 'dice']
REPEAT_DIR_PATTERN = re.compile(r'repeat\s*(\d+)', re.IGNORECASE)


def _repeat_sort_key(repeat_label):
    match = REPEAT_DIR_PATTERN.search(repeat_label)
    if match:
        return (0, int(match.group(1)), repeat_label)
    return (1, repeat_label)


def discover_repeat_datasets(parent_folder):
    """
    Find Supervisely dataset folders (ann/ + video/) under repeat exports.

    Returns:
        List of dicts: repeat_label, dataset_dir, ann_dir, video_dir
    """
    parent_path = Path(parent_folder)
    if not parent_path.is_dir():
        raise ValueError(f"Parent folder not found: {parent_path}")

    datasets = []
    seen = set()
    for ann_dir in sorted(parent_path.rglob('ann')):
        if not ann_dir.is_dir():
            continue
        dataset_dir = ann_dir.parent
        video_dir = dataset_dir / 'video'
        if not video_dir.is_dir():
            continue

        key = str(dataset_dir.resolve())
        if key in seen:
            continue
        seen.add(key)

        repeat_label = _infer_repeat_label(dataset_dir, parent_path)
        datasets.append({
            'repeat_label': repeat_label,
            'dataset_dir': dataset_dir,
            'ann_dir': ann_dir,
            'video_dir': video_dir,
        })

    datasets.sort(key=lambda d: _repeat_sort_key(d['repeat_label']))
    if len(datasets) < 2:
        raise ValueError(
            f"Need at least 2 repeat datasets under {parent_path}; "
            f"found {len(datasets)}"
        )
    return datasets


def _infer_repeat_label(dataset_dir, parent_path):
    """Derive a human-readable repeat label from the folder path."""
    rel_parts = dataset_dir.relative_to(parent_path).parts
    for part in rel_parts:
        if REPEAT_DIR_PATTERN.search(part):
            return part.strip()
    return rel_parts[0] if rel_parts else dataset_dir.name


def _reference_frame_for_video(video_name, frame_measurements_df):
    if frame_measurements_df is None:
        return None
    stem = Path(video_name).stem
    study = video_stem_to_study_number(stem)
    ref_rows = frame_measurements_df[
        frame_measurements_df['Study number'].astype(str).str.strip() == study
    ]
    if ref_rows.empty or 'Frame' not in ref_rows.columns:
        return None
    return int(ref_rows.iloc[0]['Frame'])


def resolve_frame_for_repeats(repeat_frames, frame_measurements_df, video_name):
    """
    Pick the frame index used for a subset of repeats.

    Priority:
      1. Intersection of niche-annotated frames across the subset
      2. Reference frame from frame_measurements.csv (if annotated in every repeat)
      3. None — subset repeats disagree on frame
    """
    if not repeat_frames:
        return None, 'none', []

    common = set.intersection(*repeat_frames.values())
    if common:
        return sorted(common)[0], 'intersection', sorted(common)

    ref_frame = _reference_frame_for_video(video_name, frame_measurements_df)
    if ref_frame is not None and all(
        ref_frame in frames for frames in repeat_frames.values()
    ):
        return ref_frame, 'measurements_csv', [ref_frame]

    return None, 'none', sorted(set.union(*repeat_frames.values()))


def find_comparable_groups_for_video(
    video_name, per_repeat_frames, per_repeat_paths, all_repeat_labels,
    frame_measurements_df=None,
):
    """
    Build one full comparison group or partial pair groups for a video.

    Returns a list of group dicts (empty if no comparable repeat subset exists).
    """
    all_labels = sorted(per_repeat_frames.keys())
    all_frames = {label: per_repeat_frames[label] for label in all_labels}

    frame_index, frame_source, frame_candidates = resolve_frame_for_repeats(
        all_frames, frame_measurements_df, video_name,
    )
    if frame_index is not None:
        frame_sets = [frozenset(all_frames[label]) for label in all_labels]
        return [{
            'video_name': video_name,
            'comparison_mode': 'full',
            'n_repeats_compared': len(all_labels),
            'repeats_compared': all_labels,
            'excluded_repeats': [],
            'frame_index': frame_index,
            'frame_source': frame_source,
            'frame_candidates': frame_candidates,
            'frames_per_repeat': {
                label: sorted(all_frames[label]) for label in all_labels
            },
            'same_frame_across_repeats': len(set(frame_sets)) == 1,
            'entries': [
                (label, str(per_repeat_paths[label]), frame_index)
                for label in all_labels
            ],
        }]

    partial_groups = []
    seen_pairs = set()
    for pair_labels in combinations(all_labels, 2):
        pair_frames = {label: all_frames[label] for label in pair_labels}
        pair_frame_index, pair_frame_source, pair_frame_candidates = (
            resolve_frame_for_repeats(
                pair_frames, frame_measurements_df, video_name,
            )
        )
        if pair_frame_index is None:
            continue

        pair_key = tuple(sorted(pair_labels))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        excluded = sorted(set(all_labels) - set(pair_labels))
        partial_groups.append({
            'video_name': video_name,
            'comparison_mode': 'partial',
            'n_repeats_compared': 2,
            'repeats_compared': list(pair_labels),
            'excluded_repeats': excluded,
            'frame_index': pair_frame_index,
            'frame_source': pair_frame_source,
            'frame_candidates': pair_frame_candidates,
            'frames_per_repeat': {
                label: sorted(all_frames[label]) for label in all_labels
            },
            'same_frame_across_repeats': (
                all_frames[pair_labels[0]] == all_frames[pair_labels[1]]
            ),
            'entries': [
                (label, str(per_repeat_paths[label]), pair_frame_index)
                for label in pair_labels
            ],
        })

    return partial_groups


def find_video_groups(parent_folder, frame_measurements_df=None):
    """
    Build comparison groups for videos annotated in every repeat session.

    Returns:
        (groups, repeat_datasets, skipped) where each group contains video
        metadata, resolved frame index, and per-repeat annotation paths.
        Groups may be full (all repeats) or partial (available pairs only).
    """
    repeat_datasets = discover_repeat_datasets(parent_folder)
    all_repeat_labels = [ds['repeat_label'] for ds in repeat_datasets]

    per_repeat_ann = {}
    for ds in repeat_datasets:
        label = ds['repeat_label']
        per_repeat_ann[label] = {
            f.name: f for f in sorted(ds['ann_dir'].glob('*.json'))
        }

    common_json_names = set.intersection(
        *(set(files.keys()) for files in per_repeat_ann.values())
    )
    if not common_json_names:
        raise ValueError('No annotation JSON files common to all repeats')

    groups = []
    skipped = []

    for json_name in sorted(common_json_names):
        video_name = _video_name_from_annotation_json(json_name)
        per_repeat_frames = {}
        per_repeat_paths = {}
        skip = False

        for label, files in per_repeat_ann.items():
            ann_path = files[json_name]
            annotation = load_supervisely_annotation(ann_path)
            niche_frames = get_niche_frame_indices(annotation)
            if not niche_frames:
                skip = True
                skipped.append({
                    'video': video_name,
                    'reason': f'no niche polygons in {label}',
                })
                break
            per_repeat_frames[label] = niche_frames
            per_repeat_paths[label] = ann_path

        if skip:
            continue

        video_groups = find_comparable_groups_for_video(
            video_name,
            per_repeat_frames,
            per_repeat_paths,
            all_repeat_labels,
            frame_measurements_df=frame_measurements_df,
        )
        if not video_groups:
            skipped.append({
                'video': video_name,
                'reason': 'no comparable repeat pair',
                'per_repeat_frames': {
                    k: sorted(v) for k, v in per_repeat_frames.items()
                },
            })
            continue

        for group in video_groups:
            group['video_dir'] = repeat_datasets[0]['video_dir']
            groups.append(group)

    return groups, repeat_datasets, skipped


def _metric_stats(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {'n': 0, 'mean': np.nan, 'std': np.nan, 'var': np.nan,
                'min': np.nan, 'max': np.nan, 'range': np.nan}
    return {
        'n': n,
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        'var': float(np.var(arr, ddof=1)) if n > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'range': float(np.max(arr) - np.min(arr)),
    }


def build_per_video_consistency(pairwise_df):
    """Aggregate pairwise metrics into per-video consistency spread statistics."""
    rows = []
    for video, group in pairwise_df.groupby('video', sort=True):
        row = {
            'video': video,
            'comparison_mode': group['comparison_mode'].iloc[0],
            'n_repeats_compared': int(group['n_repeats_compared'].iloc[0]),
            'repeats_compared': group['repeats_compared'].iloc[0],
            'excluded_repeats': group['excluded_repeats'].iloc[0],
            'frame_index': group['frame_index'].iloc[0],
            'same_frame_across_repeats': group['same_frame_across_repeats'].iloc[0],
            'n_pairs': len(group),
        }
        for metric in METRIC_COLS:
            stats = _metric_stats(group[metric].tolist())
            row[f'{metric}_mean'] = stats['mean']
            row[f'{metric}_std'] = stats['std']
            row[f'{metric}_var'] = stats['var']
            row[f'{metric}_range'] = stats['range']
        rows.append(row)
    return pd.DataFrame(rows)


def build_pair_summary(pairwise_df):
    """Mean/std per repeat pair and overall total."""
    rows = []
    for (rep_a, rep_b), group in pairwise_df.groupby(
        ['repeat_a', 'repeat_b'], sort=True,
    ):
        row = {
            'repeat_pair': f"{rep_a} vs {rep_b}",
            'repeat_a': rep_a,
            'repeat_b': rep_b,
            'n': len(group),
            'n_partial': int((group['comparison_mode'] == 'partial').sum()),
            'n_full': int((group['comparison_mode'] == 'full').sum()),
        }
        for metric in METRIC_COLS:
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = (
                group[metric].std(ddof=1) if len(group) > 1 else 0.0
            )
        rows.append(row)

    total = {
        'repeat_pair': 'TOTAL',
        'repeat_a': 'TOTAL',
        'repeat_b': 'TOTAL',
        'n': len(pairwise_df),
        'n_partial': int((pairwise_df['comparison_mode'] == 'partial').sum()),
        'n_full': int((pairwise_df['comparison_mode'] == 'full').sum()),
    }
    for metric in METRIC_COLS:
        total[f'{metric}_mean'] = pairwise_df[metric].mean()
        total[f'{metric}_std'] = (
            pairwise_df[metric].std(ddof=1) if len(pairwise_df) > 1 else 0.0
        )
    rows.append(total)
    return pd.DataFrame(rows)


def _short_pair_label(repeat_pair):
    """Map a repeat_pair string to a compact label, e.g. R1_vs_R2."""
    matches = REPEAT_DIR_PATTERN.findall(repeat_pair)
    if len(matches) >= 2:
        return f"R{matches[0]}_vs_R{matches[1]}"
    return re.sub(r'[^\w.\-]+', '_', repeat_pair).strip('_')


def _metric_wide_by_video(pairwise_df, metric):
    return pairwise_df.pivot_table(
        index='video', columns='repeat_pair', values=metric, aggfunc='first',
    )


def paired_ttest(sample_a, sample_b) -> dict:
    """Two-sided paired t-test on matched videos."""
    a = pd.Series(sample_a, dtype=float)
    b = pd.Series(sample_b, dtype=float)
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]
    if len(a) < 2:
        return {
            'n_a': int(len(a)),
            'n_b': int(len(b)),
            'mean_a': float(a.mean()) if len(a) else float('nan'),
            'mean_b': float(b.mean()) if len(b) else float('nan'),
            'std_a': float(a.std(ddof=1)) if len(a) > 1 else float('nan'),
            'std_b': float(b.std(ddof=1)) if len(b) > 1 else float('nan'),
            't_statistic': float('nan'),
            'p_value': float('nan'),
            'df': float('nan'),
        }

    diff = a - b
    result = stats.ttest_rel(a, b, alternative='two-sided')
    return {
        'n_a': int(len(a)),
        'n_b': int(len(b)),
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'std_a': float(a.std(ddof=1)),
        'std_b': float(b.std(ddof=1)),
        't_statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'df': float(len(a) - 1),
        'mean_diff': float(diff.mean()),
        'std_diff': float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
    }


def _ttest_row(test_type, metric, group_a, group_b, stats_row, test_method):
    row = {
        'test_type': test_type,
        'test_method': test_method,
        'metric': metric,
        'group_a': group_a,
        'group_b': group_b,
        'n_paired': stats_row['n_a'],
        'mean_a': stats_row['mean_a'],
        'mean_b': stats_row['mean_b'],
        'std_a': stats_row['std_a'],
        'std_b': stats_row['std_b'],
        'mean_diff': stats_row.get('mean_diff', float('nan')),
        'std_diff': stats_row.get('std_diff', float('nan')),
        't_statistic': stats_row['t_statistic'],
        'df': stats_row['df'],
        'p_value': stats_row['p_value'],
        'alternative': 'two-sided',
    }
    return row


def build_within_study_ttests(pairwise_df):
    """
    Within-study significance tests on matched videos.

    - Paired t-tests between repeat-pair metric distributions
    - Friedman omnibus test across all repeat pairs (videos with every pair)
    """
    rows = []
    repeat_pairs = sorted(pairwise_df['repeat_pair'].unique())

    for metric in METRIC_COLS:
        wide = _metric_wide_by_video(pairwise_df, metric)

        for pair_a, pair_b in combinations(repeat_pairs, 2):
            if pair_a not in wide.columns or pair_b not in wide.columns:
                continue
            matched = wide[[pair_a, pair_b]].dropna()
            stats_row = paired_ttest(matched[pair_a], matched[pair_b])
            rows.append(_ttest_row(
                'repeat_pair_paired', metric, pair_a, pair_b, stats_row, 'paired',
            ))

        available_cols = [p for p in repeat_pairs if p in wide.columns]
        complete = wide[available_cols].dropna()
        if len(available_cols) >= 3 and len(complete) >= 2:
            friedman = stats.friedmanchisquare(
                *[complete[col] for col in available_cols]
            )
            rows.append({
                'test_type': 'repeat_pairs_omnibus',
                'test_method': 'friedman',
                'metric': metric,
                'group_a': '; '.join(available_cols),
                'group_b': '',
                'n_paired': len(complete),
                'mean_a': float('nan'),
                'mean_b': float('nan'),
                'std_a': float('nan'),
                'std_b': float('nan'),
                'mean_diff': float('nan'),
                'std_diff': float('nan'),
                't_statistic': float(friedman.statistic),
                'df': float('nan'),
                'p_value': float(friedman.pvalue),
                'alternative': 'two-sided',
            })

    return pd.DataFrame(rows)


def enrich_pair_summary_with_pvalues(pair_summary_df, ttest_df):
    """Add within-study paired p-values to the repeat-pair summary rows."""
    enriched = pair_summary_df.copy()
    paired_tests = ttest_df[ttest_df['test_type'] == 'repeat_pair_paired']

    for _, test in paired_tests.iterrows():
        metric = test['metric']
        pair_a, pair_b = test['group_a'], test['group_b']
        p = test['p_value']
        for target_pair, other_pair in ((pair_a, pair_b), (pair_b, pair_a)):
            col = f"{metric}_p_vs_{_short_pair_label(other_pair)}"
            mask = enriched['repeat_pair'] == target_pair
            if mask.any():
                enriched.loc[mask, col] = p

    omnibus = ttest_df[ttest_df['test_type'] == 'repeat_pairs_omnibus']
    for metric in METRIC_COLS:
        col = f'{metric}_p_omnibus'
        enriched[col] = np.nan
        metric_rows = omnibus[omnibus['metric'] == metric]
        if not metric_rows.empty:
            enriched.loc[enriched['repeat_pair'] != 'TOTAL', col] = (
                metric_rows.iloc[0]['p_value']
            )

    return enriched


def build_ttest_results(pairwise_df):
    """Build within-study significance tests for matched videos."""
    return build_within_study_ttests(pairwise_df)


def _ttest_output_path(output_path):
    path = Path(output_path)
    return path.with_name(f"{path.stem}_ttest{path.suffix}")


def _print_ttest_block(ttest_df):
    if ttest_df.empty:
        return

    print('-' * 72)
    print('Within-study significance tests (matched videos):')
    for test_type in sorted(ttest_df['test_type'].unique()):
        subset = ttest_df[ttest_df['test_type'] == test_type]
        print(f"  [{test_type}]")
        for _, row in subset.iterrows():
            if row['test_type'] == 'repeat_pairs_omnibus':
                print(
                    f"    {row['metric'].upper():>4}  Friedman across repeat pairs "
                    f"(n={int(row['n_paired'])}): "
                    f"chi2={row['t_statistic']:.4f}, "
                    f"p={row['p_value']:.6g}"
                )
                continue

            label = f"{row['group_a']} vs {row['group_b']}"
            print(
                f"    {row['metric'].upper():>4}  {label} "
                f"(n={int(row['n_paired'])}): "
                f"mean={row['mean_a']:.4f} vs {row['mean_b']:.4f}, "
                f"diff={row['mean_diff']:.4f}, "
                f"t={row['t_statistic']:.4f}, df={row['df']:.2f}, "
                f"p={row['p_value']:.6g}"
            )


def build_consistency_baseline(per_video_df):
    """
    Dataset-level baseline describing how much consistency varies across videos.

    Uses per-video mean pairwise metrics and their spread across the cohort.
    """
    rows = []
    for metric in METRIC_COLS:
        col = f'{metric}_mean'
        spread_col = f'{metric}_std'
        stats = _metric_stats(per_video_df[col].tolist())
        spread_stats = _metric_stats(per_video_df[spread_col].tolist())
        rows.append({
            'metric': metric,
            'cohort_mean_of_video_means': stats['mean'],
            'cohort_std_across_videos': stats['std'],
            'cohort_var_across_videos': stats['var'],
            'cohort_min_video_mean': stats['min'],
            'cohort_max_video_mean': stats['max'],
            'mean_within_video_pair_std': spread_stats['mean'],
            'mean_within_video_pair_var': spread_stats['var'],
        })
    frame_agreement = per_video_df['same_frame_across_repeats'].mean()
    rows.append({
        'metric': 'frame_agreement_rate',
        'cohort_mean_of_video_means': frame_agreement,
        'cohort_std_across_videos': np.nan,
        'cohort_var_across_videos': np.nan,
        'cohort_min_video_mean': np.nan,
        'cohort_max_video_mean': np.nan,
        'mean_within_video_pair_std': np.nan,
        'mean_within_video_pair_var': np.nan,
    })
    partial_rate = (per_video_df['comparison_mode'] == 'partial').mean()
    rows.append({
        'metric': 'partial_comparison_rate',
        'cohort_mean_of_video_means': partial_rate,
        'cohort_std_across_videos': np.nan,
        'cohort_var_across_videos': np.nan,
        'cohort_min_video_mean': np.nan,
        'cohort_max_video_mean': np.nan,
        'mean_within_video_pair_std': np.nan,
        'mean_within_video_pair_var': np.nan,
    })
    return pd.DataFrame(rows)


def _save_group_outputs(masks, entries, group, group_idx, tmp_dir, video_dir):
    tmp_dir = Path(tmp_dir)
    subdir_name = (
        f"{group_idx:03d}_{_sanitize_path_component(group['video_name'])}"
    )
    group_dir = tmp_dir / subdir_name
    group_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{label}/frame{group['frame_index']}" for label, _, _ in entries]
    for i, (label, _, frame_index) in enumerate(entries):
        mask_path = group_dir / f"mask_{i}_{label}_frame{frame_index}.png"
        cv2.imwrite(str(mask_path), masks[i].astype(np.uint8) * 255)

    video_path = _resolve_video_path(video_dir, group['video_name'])
    if video_path is not None:
        base_img = read_video_frame(video_path, group['frame_index'])
    else:
        h, w = masks[0].shape
        base_img = np.full((h, w, 3), 128, dtype=np.uint8)

    overlay = draw_contours_overlay(base_img, masks, labels)
    cv2.imwrite(str(group_dir / "overlay_contours.png"), overlay)
    return str(group_dir)


def _print_summary(pair_summary, per_video_df, baseline_df, skipped, ttest_df=None):
    print('=' * 72)
    print('Pairwise repeat comparison (mean +/- std):')
    for _, row in pair_summary.iterrows():
        print(
            f"  {row['repeat_pair']:>28}  (n={int(row['n']):>2})  "
            f"Dice={row['dice_mean']:.4f} +/- {row['dice_std']:.4f}  "
            f"NSD={row['nsd_mean']:.4f} +/- {row['nsd_std']:.4f}  "
            f"IoU={row['iou_mean']:.4f} +/- {row['iou_std']:.4f}"
        )

    print('-' * 72)
    print('Per-video consistency spread (mean pairwise std within each video):')
    for metric in METRIC_COLS:
        col = f'{metric}_std'
        stats = _metric_stats(per_video_df[col].tolist())
        print(
            f"  {metric}: mean={stats['mean']:.4f}  "
            f"std={stats['std']:.4f}  var={stats['var']:.4f}"
        )

    print('-' * 72)
    print('Cohort consistency baseline:')
    for _, row in baseline_df.iterrows():
        if row['metric'] in ('frame_agreement_rate', 'partial_comparison_rate'):
            print(f"  {row['metric']}: {row['cohort_mean_of_video_means']:.1%}")
        else:
            m = row['metric']
            print(
                f"  {m}: video-mean {row['cohort_mean_of_video_means']:.4f} "
                f"+/- {row['cohort_std_across_videos']:.4f} (var={row['cohort_var_across_videos']:.6f}); "
                f"within-video pair std avg={row['mean_within_video_pair_std']:.4f}"
            )

    if ttest_df is not None:
        _print_ttest_block(ttest_df)

    if skipped:
        print('-' * 72)
        print(f'Skipped videos ({len(skipped)}):')
        for item in skipped:
            detail = item.get('per_repeat_frames', item['reason'])
            print(f"  - {item['video']}: {detail}")
    print('=' * 72)


def load_frame_measurements_with_frame(parent_folder, filename='frame_measurements.csv'):
    """Load measurements CSV including the reference Frame column."""
    csv_path = Path(parent_folder) / filename
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    required = {'Study number', 'Pixels 1 mm', 'Frame'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Warning: {csv_path} missing columns {missing}; "
              f"reference-frame fallback disabled")
        return None
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Intra-annotator consistency across repeated Supervisely sessions'
    )
    parser.add_argument(
        '--parent-folder', type=str, required=True,
        help='Root folder containing repeat export subfolders',
    )
    parser.add_argument(
        '--measurements-root', type=str, default=None,
        help='Folder containing frame_measurements.csv for NSD tau and reference frames '
             '(default: parent-folder, then parent of parent-folder)',
    )
    parser.add_argument(
        '--measurements-csv', type=str, default='frame_measurements.csv',
        help='CSV filename with Study number, Frame, and Pixels 1 mm columns',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path for detailed pairwise results CSV',
    )
    parser.add_argument(
        '--save-masks', action='store_true', default=False,
        help='Save extracted masks and overlay images',
    )
    parser.add_argument(
        '--tmp-dir', type=str, default=None,
        help='Directory for saved masks/overlays',
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
    )
    args = parser.parse_args()

    measurements_root = Path(
        args.measurements_root or args.parent_folder
    )
    if not (measurements_root / args.measurements_csv).exists():
        alt = Path(args.parent_folder).parent / args.measurements_csv
        if alt.exists():
            measurements_root = alt.parent

    frame_measurements = load_frame_measurements(
        measurements_root, filename=args.measurements_csv,
    )
    frame_measurements_df = load_frame_measurements_with_frame(
        measurements_root, filename=args.measurements_csv,
    )
    print(f"Loaded NSD tau for {len(frame_measurements)} studies")

    groups, repeat_datasets, skipped = find_video_groups(
        args.parent_folder, frame_measurements_df=frame_measurements_df,
    )
    repeat_labels = [ds['repeat_label'] for ds in repeat_datasets]
    n_full = sum(1 for g in groups if g['comparison_mode'] == 'full')
    n_partial = sum(1 for g in groups if g['comparison_mode'] == 'partial')
    print(
        f"Found {len(repeat_datasets)} repeats: {repeat_labels}; "
        f"{len(groups)} comparable group(s) "
        f"({n_full} full, {n_partial} partial pair), {len(skipped)} skipped"
    )

    if not groups:
        print('No comparable videos found!')
        sys.exit(1)

    tmp_dir = (
        Path(args.tmp_dir) if args.tmp_dir
        else Path('tmp') / 'compare_intra_annotator_consistency'
    )

    pairwise_rows = []
    for group_idx, group in enumerate(groups):
        try:
            nsd_tau = get_nsd_tau_for_video(
                group['video_name'], frame_measurements,
            )
            masks = []
            labels = []
            areas = []
            for repeat_label, ann_path, frame_index in group['entries']:
                annotation = load_supervisely_annotation(ann_path)
                mask = extract_niche_mask_at_frame(annotation, frame_index)
                masks.append(mask)
                labels.append(repeat_label)
                areas.append(int(np.sum(mask)))

            area_stats = _metric_stats(areas)
            area_cv = (
                area_stats['std'] / area_stats['mean']
                if area_stats['mean'] > 0 else np.nan
            )

            if args.save_masks:
                saved = _save_group_outputs(
                    masks, group['entries'], group, group_idx,
                    tmp_dir, group['video_dir'],
                )
                if args.verbose:
                    print(f"Saved {group['video_name']} -> {saved}")

            n = len(labels)
            for i, j in combinations(range(n), 2):
                metrics = compute_segmentation_metrics(
                    masks[i], masks[j], nsd_tau=nsd_tau,
                )
                repeats_compared = '; '.join(group['repeats_compared'])
                excluded_repeats = '; '.join(group['excluded_repeats'])
                pairwise_rows.append({
                    'video': group['video_name'],
                    'comparison_mode': group['comparison_mode'],
                    'n_repeats_compared': group['n_repeats_compared'],
                    'repeats_compared': repeats_compared,
                    'excluded_repeats': excluded_repeats,
                    'frame_index': group['frame_index'],
                    'frame_source': group['frame_source'],
                    'same_frame_across_repeats': group['same_frame_across_repeats'],
                    'frames_per_repeat': str(group['frames_per_repeat']),
                    'nsd_tau': nsd_tau,
                    'repeat_a': labels[i],
                    'repeat_b': labels[j],
                    'repeat_pair': f"{labels[i]} vs {labels[j]}",
                    'mask_area_a': areas[i],
                    'mask_area_b': areas[j],
                    'mask_area_cv_all_repeats': area_cv,
                    **metrics,
                })

            if args.verbose:
                mode_note = (
                    f"partial pair [{', '.join(group['repeats_compared'])}]"
                    if group['comparison_mode'] == 'partial'
                    else 'full'
                )
                print(
                    f"\n{group['video_name']} @ frame {group['frame_index']} "
                    f"({group['frame_source']}, {mode_note}, "
                    f"same_frame={group['same_frame_across_repeats']})"
                )
                if group['excluded_repeats']:
                    print(f"  excluded repeats: {', '.join(group['excluded_repeats'])}")
                for i, j in combinations(range(n), 2):
                    m = compute_segmentation_metrics(
                        masks[i], masks[j], nsd_tau=nsd_tau,
                    )
                    print(
                        f"  {labels[i]} vs {labels[j]}: "
                        f"Dice={m['dice']:.4f}  NSD={m['nsd']:.4f}  IoU={m['iou']:.4f}"
                    )
                    print(f"    areas: {areas[i]} vs {areas[j]} px")

        except Exception as exc:
            print(f"Error processing {group['video_name']}: {exc}")
            continue

    if not pairwise_rows:
        print('No pairs processed!')
        sys.exit(1)

    pairwise_df = pd.DataFrame(pairwise_rows)
    per_video_df = build_per_video_consistency(pairwise_df)
    pair_summary_df = build_pair_summary(pairwise_df)
    baseline_df = build_consistency_baseline(per_video_df)

    ttest_df = build_ttest_results(pairwise_df)
    pair_summary_df = enrich_pair_summary_with_pvalues(pair_summary_df, ttest_df)

    print()
    _print_summary(pair_summary_df, per_video_df, baseline_df, skipped, ttest_df)

    if args.output:
        out = Path(args.output)
        pairwise_df.to_csv(out, index=False)
        per_video_df.to_csv(out.with_name(f"{out.stem}_per_video{out.suffix}"), index=False)
        pair_summary_df.to_csv(_average_output_path(out), index=False)
        baseline_df.to_csv(out.with_name(f"{out.stem}_baseline{out.suffix}"), index=False)
        if not ttest_df.empty:
            ttest_path = _ttest_output_path(out)
            ttest_df.to_csv(ttest_path, index=False)
            print(
                f"\nSaved:\n  {out}\n  {out.with_name(f'{out.stem}_per_video{out.suffix}')}"
                f"\n  {_average_output_path(out)}\n  {out.with_name(f'{out.stem}_baseline{out.suffix}')}"
                f"\n  {ttest_path}"
            )
        else:
            print(
                f"\nSaved:\n  {out}\n  {out.with_name(f'{out.stem}_per_video{out.suffix}')}"
                f"\n  {_average_output_path(out)}\n  {out.with_name(f'{out.stem}_baseline{out.suffix}')}"
            )


if __name__ == '__main__':
    main()
