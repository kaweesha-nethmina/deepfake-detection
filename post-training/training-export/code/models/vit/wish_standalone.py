"""Independent Wish experiment: content-grouped splits and sealed ViT evaluation.

This does not replace the team's existing split contract. Other models must use
the exported manifest before their results can form a controlled comparison.
"""
from __future__ import annotations
import io
from models.vit.audit_progress import discover_images, inspect_inventory, verify_inventory, timed_split_assignment
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps
from models.vit.manifest import DATASET, FIELDS, load_audited_manifest, sha256_file, source_from_name, validate_rows
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

def find_root(directory):
    roots = []
    for current, directories, _ in os.walk(directory):
        if 'Real' in directories and 'Fake' in directories:
            roots.append(Path(current).resolve())
            directories[:] = []
    if len(roots) != 1:
        raise ValueError(f'Expected one Real/Fake image root under {directory}; found {roots}')
    return roots[0]

def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def inspect_image(item):
    root, relative = item
    label, source = source_from_name(relative)
    if Path(relative).parts[0] != ('Real' if label == 0 else 'Fake'):
        raise ValueError(f'Folder and filename labels disagree: {relative}')
    path = root / relative
    try:
        content = path.read_bytes()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            small = list(rgb.convert('L').resize((9, 8)).tobytes())
            dhash = sum((int(small[y * 9 + x] > small[y * 9 + x + 1]) << y * 8 + x for y in range(8) for x in range(8)))
    except Exception as error:
        raise ValueError(f'Unreadable image {relative}: {error}') from error
    return {'filepath': relative, 'label': label, 'source': source, 'identity': '', 'sha256': hashlib.sha256(content).hexdigest(), 'pixel_sha256': pixels, 'dhash': f'{dhash:016x}'}

def group_content(rows, distance=4):
    """Union all dHash neighbors; multi-index lookup avoids all-pairs comparison."""
    if not 0 <= distance <= 8:
        raise ValueError('Near-duplicate distance must be in 0..8')
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        a, b = (find(a), find(b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen, buckets, pairs = ({}, defaultdict(list), [])
    for index, row in enumerate(rows):
        for kind in ('sha256', 'pixel_sha256'):
            key = (kind, row[kind])
            if key in seen:
                previous = seen[key]
                if rows[previous]['label'] != row['label']:
                    raise ValueError(f"Exact duplicate has conflicting labels: {row['filepath']}")
                union(index, previous)
            seen[key] = index
        value = int(row['dhash'], 16)
        keys, candidates = ([], set())
        for part in range(distance + 1):
            start, end = (64 * part // (distance + 1), 64 * (part + 1) // (distance + 1))
            key = (part, value >> start & (1 << end - start) - 1)
            keys.append(key)
            candidates.update(buckets[key])
        for previous in sorted(candidates):
            difference = (value ^ int(rows[previous]['dhash'], 16)).bit_count()
            if difference <= distance:
                union(index, previous)
                pairs.append({'filepath_a': rows[previous]['filepath'], 'filepath_b': row['filepath'], 'distance': difference})
        for key in keys:
            buckets[key].append(index)
        if (index + 1) % 10000 == 0:
            print(f'Grouped {index + 1:,}/{len(rows):,} images', flush=True)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(dict(row))
    return (list(groups.values()), pairs)

def take_groups(groups, target, leave=0):
    selected, count = ([], 0)
    limit = len(groups) - leave
    while len(selected) < limit and count < target:
        group = groups[len(selected)]
        selected.append(group)
        count += len(group)
    del groups[:len(selected)]
    return selected

def assign_splits(rows, seed=42, distance=4):
    """Deduplicate before splitting; keep perceptual components in one domain."""
    rows = sorted(rows, key=lambda row: row['filepath'])
    excluded = [dict(row, reason='unspecified_ai_source') for row in rows if row['source'] == 'AiGenImage']
    groups, pairs = group_content([r for r in rows if r['source'] != 'AiGenImage'], distance)
    strata = defaultdict(list)
    assigned = []
    for group in groups:
        sources, labels = ({r['source'] for r in group}, {r['label'] for r in group})
        if len(labels) > 1 or {'StyleGAN', 'StableDiffusion'}.issubset(sources):
            excluded.extend((dict(row, reason='ambiguous_cross_source_perceptual_group') for row in group))
            continue
        unique, seen = ([], set())
        for row in group:
            key = row['pixel_sha256']
            if key in seen:
                excluded.append(dict(row, reason='exact_duplicate_removed_before_split'))
            else:
                seen.add(key)
                unique.append(row)
        group_id = hashlib.sha256('\n'.join((r['filepath'] for r in group)).encode()).hexdigest()
        for row in unique:
            row['group_id'] = group_id
        if sources == {'StableDiffusion'}:
            assigned.extend((dict(row, split='cross_gen') for row in unique))
        else:
            stratum = Counter((r['source'] for r in unique)).most_common(1)[0][0]
            strata[stratum].append(unique)
    if not assigned or not all((strata[key] for key in ('FFHQ', 'CelebA', 'StyleGAN'))):
        raise ValueError('Need FFHQ, CelebA, StyleGAN and StableDiffusion after duplicate filtering.')
    rng = random.Random(seed)
    real_count = sum((len(g) for key in ('FFHQ', 'CelebA') for g in strata[key]))
    cross_fraction = len(assigned) / real_count
    if cross_fraction >= 0.5:
        raise ValueError('Too few real controls for this cross-generator protocol.')
    for source in ('FFHQ', 'CelebA', 'StyleGAN'):
        groups = strata[source]
        rng.shuffle(groups)
        if source != 'StyleGAN':
            controls = take_groups(groups, round(sum(map(len, groups)) * cross_fraction), leave=3)
            assigned.extend((dict(row, split='cross_gen') for group in controls for row in group))
        if len(groups) < 3:
            raise ValueError(f'Too few independent content groups for {source}')
        total = sum(map(len, groups))
        train = take_groups(groups, round(total * 0.7), leave=2)
        val = take_groups(groups, round(total * 0.15), leave=1)
        for split, items in (('train', train), ('val', val), ('test', groups)):
            assigned.extend((dict(row, split=split) for group in items for row in group))
    assigned.sort(key=lambda row: row['filepath'])
    validate_rows(assigned)
    by_group = defaultdict(set)
    for row in assigned:
        by_group[row['group_id']].add(row['split'])
    if any((len(parts) != 1 for parts in by_group.values())):
        raise AssertionError('Content group leaked between splits')
    return (assigned, excluded, pairs)

def prepare(root, output, seed=42, provenance=None, workers=4):
    root, output = (Path(root), Path(output))
    files = discover_images(root, EXTENSIONS)
    if not files:
        raise ValueError(f'No images under {root}')
    provenance = provenance or {'version': 'unreported_attached_snapshot'}
    settings = {'seed': seed, 'near_distance': 4, 'provenance': provenance, 'protocol': 'wish_independent_v1'}
    if output.exists():
        if not (output / 'audit.json').is_file():
            raise ValueError('Incomplete audit directory. Use a new OUTPUT_ROOT.')
        audit = json.loads((output / 'audit.json').read_text())
        if audit.get('independent_settings') != settings:
            raise ValueError('Data preparation settings changed. Use a new OUTPUT_ROOT.')
        with (output / 'inventory.csv').open() as handle:
            inventory = list(csv.DictReader(handle))
        if sha256_file(output / 'inventory.csv') != audit['inventory_sha256']:
            raise ValueError('Inventory checksum mismatch')
        if [r['filepath'] for r in inventory] != files:
            raise ValueError('Dataset file list changed. Use a new OUTPUT_ROOT.')
        verify_inventory(root, inventory, sha256_file, workers)
        load_audited_manifest(output / 'manifest.csv', root, verify_splits=())
        print('Reusing verified, unchanged split manifest', flush=True)
        return audit
    print(f'Reading and hashing {len(files):,} images; no model evaluation occurs here.', flush=True)
    inventory = inspect_inventory(root, files, inspect_image, workers)
    rows, excluded, pairs = timed_split_assignment(assign_splits, inventory, seed)
    output.mkdir(parents=True)
    write_csv(output / 'inventory.csv', inventory, FIELDS)
    write_csv(output / 'manifest.csv', rows, (*FIELDS, 'group_id'))
    write_csv(output / 'excluded.csv', excluded, (*FIELDS, 'reason'))
    write_csv(output / 'near_duplicate_candidates.csv', pairs, ('filepath_a', 'filepath_b', 'distance'))
    for split in ('train', 'val', 'test', 'cross_gen'):
        write_csv(output / f'{split}.csv', [r for r in rows if r['split'] == split], (*FIELDS, 'group_id'))
    write_csv(output / 'near_duplicates.csv', [], ('filepath_a', 'filepath_b', 'distance'))
    snapshot = sha256_file(output / 'inventory.csv')
    audit = {'dataset': DATASET, 'dataset_version': f'content-sha256:{snapshot}', 'manifest_sha256': sha256_file(output / 'manifest.csv'), 'inventory_sha256': snapshot, 'independent_settings': settings, 'near_duplicate_pairs': 0, 'detected_near_pairs_before_grouping': len(pairs), 'near_distance': 4, 'identity_metadata_available': False, 'status': 'passed', 'counts': dict(Counter((f"{r['split']}:{r['source']}:{r['label']}" for r in rows))), 'exclusions': dict(Counter((r['reason'] for r in excluded))), 'limitations': 'dHash is heuristic, not face identity verification; source/compression confounding remains.'}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    return audit

def train_or_resume(cfg):
    """Entry for every torchrun process; one process per GPU joins the same run."""
    from models.vit import distributed as dist_utils
    from models.vit.train import train
    ctx = dist_utils.setup(cfg.get('device', 'auto'))
    try:
        output = Path(cfg['output']['results_dir']) / cfg['run_name']
        complete = (output / 'training_complete.json').exists()
        last = output / 'checkpoints/last.pt'
        resume = str(last) if last.is_file() else None
        dist_utils.barrier(ctx)  # every rank has read the run state before anything is written
        if complete:
            if json.loads((output / 'run_config_used.json').read_text()) != cfg:
                raise ValueError('Completed run has different settings. Choose a new RUN_NAME.')
            if ctx.main:
                load_audited_manifest(cfg['data']['manifest'], cfg['data']['root'], verify_splits=('train', 'val'))
                if not (output / 'checkpoints/best_model.pt').is_file():
                    raise FileNotFoundError('Completed run is missing its best checkpoint.')
                print(f'Reusing completed run: {output}', flush=True)
            dist_utils.barrier(ctx)
            return output
        return train(cfg, resume=resume, ctx=ctx)
    finally:
        dist_utils.cleanup(ctx)

def evaluate_vit(cfg, checkpoint, output, device='cuda'):
    """Freeze one ViT before either held-out set; never imply a four-model result."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from models.vit.evaluate_crossgen import predict_loader, single_image_latency, validate_metadata
    from models.vit.reporting import compute_metrics, make_figures
    from models.vit.runtime import ManifestDataset, device_for, load_checkpoint
    output, checkpoint = (Path(output), Path(checkpoint))
    if output.exists():
        raise FileExistsError('Evaluation already started here. Inspect saved results; do not rerun tests for tuning.')
    rows, audit = load_audited_manifest(cfg['data']['manifest'], cfg['data']['root'])
    device = device_for(device)
    model, metadata = load_checkpoint(cfg, checkpoint, device)
    validate_metadata(metadata, cfg, audit['manifest_sha256'])
    output.mkdir(parents=True)
    frozen = {'checkpoint_sha256': sha256_file(checkpoint), 'manifest_sha256': audit['manifest_sha256'], 'config': cfg, 'metadata': metadata, 'threshold': 0.5, 'scope': 'independent_vit_only_not_four_model_comparison'}
    (output / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    status = output / 'status.json'
    status.write_text(json.dumps({'status': 'running'}))
    predictions, metrics, timings = ([], {}, {})
    try:
        for split in ('test', 'cross_gen'):
            dataset = ManifestDataset(rows, cfg['data']['root'], split, metadata['preprocessing'])
            loader = DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=min(4, os.cpu_count() or 1), pin_memory=device.type == 'cuda')
            scores, batch_ms = predict_loader(model, loader, device, 'vit_b16', cfg['run_name'])
            predictions.extend(scores)
            metrics[split] = compute_metrics([r['label'] for r in scores], [r['probability'] for r in scores])
            timings[split] = {'batch_ms_per_image': batch_ms, 'single_image_ms': single_image_latency(model, dataset, device)}
            print(f'Completed sealed {split}: {len(scores):,} images', flush=True)
        pd.DataFrame(predictions).to_csv(output / 'predictions.csv', index=False)
        make_figures(output / 'predictions.csv', output / 'figures')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for axis, split in zip(axes, ('test', 'cross_gen')):
            cm = metrics[split]['confusion_matrix']
            axis.imshow(cm, cmap='Blues')
            maximum = max(map(max, cm))
            for y in (0, 1):
                for x in (0, 1):
                    axis.text(x, y, str(cm[y][x]), ha='center', va='center', color='white' if cm[y][x] > maximum / 2 else 'black')
            axis.set(title=split, xlabel='Predicted', ylabel='True', xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        fig.tight_layout()
        fig.savefig(output / 'figures/confusion_matrices_side_by_side.png', dpi=180)
        plt.close(fig)
        result = {'scope': frozen['scope'], 'metrics': metrics, 'timing': timings, 'timing_scope': 'FP32 model forward only, synchronized; excludes disk IO and preprocessing', 'accuracy_drop_pp': 100 * (metrics['test']['accuracy'] - metrics['cross_gen']['accuracy']), 'f1_drop': metrics['test']['f1_score'] - metrics['cross_gen']['f1_score'], **{key: metadata[key] for key in ('total_parameters', 'trainable_parameters', 'training_time_seconds')}}
        (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        status.write_text(json.dumps({'status': 'complete'}))
        return result
    except Exception as error:
        status.write_text(json.dumps({'status': 'failed_not_final', 'error': str(error)}))
        raise
