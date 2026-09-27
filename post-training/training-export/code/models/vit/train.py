"""Manifest-based training on one device or several GPUs (DDP via torchrun).

This command never evaluates either test set. With WORLD_SIZE > 1 each GPU trains
on a disjoint shard; gradients are averaged by DistributedDataParallel, so the
effective batch is per-GPU microbatch x GPUs x accumulation, exactly as configured.
"""
from __future__ import annotations
import os
import argparse
import copy
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from models.vit import distributed as dist_utils
from models.vit.manifest import load_audited_manifest
from models.vit.presize_cache import open_cache
from models.vit.reporting import compute_metrics, learning_curves
from models.vit.runtime import ManifestDataset, binary_loss, build_model, load_config, per_sample_loss, preprocessing_for, probabilities, set_seed


class ShardSampler(Sampler):
    """Disjoint, unpadded validation shards: every validation image is scored exactly once."""

    def __init__(self, size, rank, world):
        self.indices = list(range(rank, size, world))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap(model, ctx):
    """DDP after staged unfreezing; rewrapped whenever the trainable set changes."""
    if ctx.world == 1:
        return model
    ids = [ctx.device.index] if ctx.device.type == 'cuda' else None
    return DistributedDataParallel(model, device_ids=ids, output_device=ids[0] if ids else None)

def phase_parameters(model, cfg, warmup):
    """Keep teammates' heads/architectures; only apply their staged unfreezing."""
    for parameter in model.parameters():
        parameter.requires_grad = True
    if not cfg['train'].get('warmup_epochs', 0):
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    backbone = getattr(model, 'backbone', model)
    head = getattr(model, 'head', None)
    if head is None:
        head = getattr(backbone, 'fc', getattr(backbone, 'classifier', None))
    if head is None:
        raise ValueError('Cannot identify classifier for staged training.')
    for parameter in head.parameters():
        parameter.requires_grad = True
    if not warmup:
        if hasattr(backbone, 'layer4'):
            blocks = [backbone.layer3, backbone.layer4]
        elif hasattr(backbone, 'blocks'):
            blocks = list(backbone.blocks.children())[-2:]
        elif hasattr(backbone, 'features'):
            blocks = list(backbone.features.children())[-2:]
        else:
            raise ValueError('Cannot identify backbone blocks for fine-tuning.')
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = True

def run_epoch(model, loader, device, ctx, optimizer=None, scheduler=None, scaler=None, accumulation=1, grad_clip=1.0, amp=False, log_every=0):
    """One pass over this rank's shard; losses/scores are gathered so every rank sees whole-dataset values.

    Per-step host synchronisation is limited to the GradScaler check, so the CPU can queue the
    next step while the GPU is still busy. Nonfinite losses are checked at log points and at the end.
    """
    training = optimizer is not None
    model.train(training)
    network = model if training else unwrap(model)  # evaluation never enters a DDP collective
    if training:
        for module in unwrap(model).modules():
            parameters = list(module.parameters(recurse=False))
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and parameters and (not any((p.requires_grad for p in parameters))):
                module.eval()
        optimizer.zero_grad(set_to_none=True)
    steps, local_samples = (len(loader), len(loader.sampler))
    losses, scores, labels_out, indices_out = ([], [], [], [])
    started = fetched = time.perf_counter()
    waiting = 0.0
    with torch.set_grad_enabled(training):
        for step, (images, labels, indices) in enumerate(loader):
            waiting += time.perf_counter() - fetched
            images, labels = (images.to(device, non_blocking=True), labels.to(device, non_blocking=True))
            boundary = (step + 1) % accumulation == 0 or step + 1 == steps
            sync = network.no_sync() if training and not boundary and isinstance(network, DistributedDataParallel) else nullcontext()
            with sync:
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = network(images)
                    loss = binary_loss(logits, labels)
                if training:
                    start = step // accumulation * accumulation * loader.batch_size
                    window_samples = min(accumulation * loader.batch_size, local_samples - start)
                    scaler.scale(loss * len(labels) / window_samples).backward()
            if training and boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), grad_clip)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                detached = logits.detach().float()
                losses.append(per_sample_loss(detached, labels))
                scores.append(probabilities(detached))
            labels_out.append(labels.detach())
            indices_out.append(indices)
            if log_every and ((step + 1) % log_every == 0 or step + 1 == steps):
                seen = torch.cat(losses)
                if not torch.isfinite(seen).all():
                    raise ValueError('Nonfinite loss; refusing to save a misleading result.')
                if ctx.main:
                    print(f"{('train' if training else 'val')} batch={step + 1}/{steps} loss={float(seen.mean()):.4f} (rank 0 shard)", flush=True)
            fetched = time.perf_counter()
    seconds = time.perf_counter() - started
    local = {'index': torch.cat(indices_out).tolist() if indices_out else [],
             'score': torch.cat(scores).cpu().tolist() if scores else [],
             'label': torch.cat(labels_out).cpu().tolist() if labels_out else [],
             'loss': torch.cat(losses).cpu().tolist() if losses else [],
             'seconds': seconds, 'waiting': waiting}
    merged = {}
    parts = dist_utils.all_gather_object(ctx, local)
    for part in parts:
        for index, score, label, value in zip(part['index'], part['score'], part['label'], part['loss']):
            merged.setdefault(index, (score, label, value))  # drops DistributedSampler padding repeats
    if not merged:
        raise ValueError('Empty loader.')
    order = sorted(merged)
    values = np.asarray([merged[i][2] for i in order], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite loss; refusing to save a misleading result.')
    metrics = compute_metrics([merged[i][1] for i in order], [merged[i][0] for i in order]) if ctx.main else None
    slowest = max((part['seconds'] for part in parts))
    stats = {'images': len(order), 'images_per_second': len(order) / max(slowest, 1e-09),
             'data_wait_fraction': max((part['waiting'] / max(part['seconds'], 1e-09) for part in parts))}
    return (float(values.mean()), metrics, stats)

def rng_state(generator, ctx=None):
    np_state = np.random.get_state()
    state = {'python': random.getstate(), 'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]], 'torch': torch.get_rng_state(), 'loader': generator.get_state()}
    if ctx is not None and ctx.device.type == 'cuda':
        state['cuda_device'] = torch.cuda.get_rng_state(ctx.device)  # this rank's GPU only
    elif ctx is None and torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state['mps'] = torch.mps.get_rng_state()
    return state

def restore_rng(state, generator, ctx=None):
    if isinstance(state, list):
        if ctx is None or len(state) != ctx.world:
            raise ValueError('Saved RNG states do not match the number of training processes.')
        state = state[ctx.rank]
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype='uint32'), *n[2:]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if 'cuda_device' in state:
        torch.cuda.set_rng_state(state['cuda_device'], ctx.device)
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'])

def train(cfg, resume=None, ctx=None):
    invocation_start = time.perf_counter()
    cfg = copy.deepcopy(cfg)
    ctx = ctx or dist_utils.setup(cfg.get('device', 'auto'))
    t = cfg['train']
    if not 1 <= t['epochs'] <= 30 or t['early_stopping_patience'] != 6:
        raise ValueError('Team protocol: 1..30 epochs and patience=6.')
    expected_world = int(cfg.get('distributed', {}).get('world_size', 1))
    if ctx.world != expected_world:
        raise ValueError(f'Config expects {expected_world} training process(es) but {ctx.world} were started.')
    batch, effective = (cfg['data']['batch_size'], t.get('effective_batch_size', 32))
    per_step = batch * ctx.world
    if batch < 1 or effective < per_step or effective % per_step:
        raise ValueError('Effective batch size must be a positive multiple of per-GPU microbatch x GPU count.')
    accumulation = effective // per_step
    out = Path(cfg['output']['results_dir']) / cfg['run_name']
    if out.exists() and (not resume):
        raise FileExistsError(f'Use a new run_name or --resume: {out}')
    # Rank 0 re-hashes train/val images (the audit's tamper check); the others only parse the manifest.
    rows, audit = load_audited_manifest(cfg['data']['manifest'], cfg['data']['root'], cfg['data'].get('near_duplicate_review'), ('train', 'val') if ctx.main else ())
    dist_utils.barrier(ctx)
    device = ctx.device
    set_seed(cfg['seed'])
    if ctx.main:
        model = build_model(cfg, pretrained=False if resume else None)  # rank 0 downloads weights once
    dist_utils.barrier(ctx)
    if not ctx.main:
        model = build_model(cfg, pretrained=False if resume else None)  # others read the local cache
    if ctx.world > 1 and device.type == 'cuda' and any((isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if ctx.world > 1:
        torch.manual_seed(cfg['seed'] + 1000 * ctx.rank)  # independent dropout/augmentation noise per GPU
    preproc = preprocessing_for(model, cfg)
    cache = open_cache(preproc, audit['manifest_sha256'])
    if ctx.main:
        print(f"Processes: {ctx.world} | per-GPU microbatch: {batch} | accumulation: {accumulation} | effective batch: {effective} | presize cache: {'on' if cache else 'off'}", flush=True)
    generator = torch.Generator().manual_seed(cfg['seed'] + ctx.rank)
    workers = cfg['data'].get('num_workers', 0)
    limit = 32 if cfg.get('smoke') else None

    def loader_for(part):
        dataset = ManifestDataset(rows, cfg['data']['root'], part, preproc, train=part == 'train', limit=limit, cache=cache)
        if part == 'train':
            sampler = DistributedSampler(dataset, num_replicas=ctx.world, rank=ctx.rank, shuffle=True, seed=cfg['seed'], drop_last=False) if ctx.world > 1 else None
        else:
            sampler = ShardSampler(len(dataset), ctx.rank, ctx.world)
        extra = {'prefetch_factor': 4} if workers else {}
        return DataLoader(dataset, batch_size=batch, sampler=sampler, shuffle=part == 'train' and sampler is None, generator=generator, num_workers=workers, pin_memory=device.type == 'cuda', drop_last=False, **extra)
    loaders = {part: loader_for(part) for part in ('train', 'val')}
    amp = bool(t.get('mixed_precision', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, best_loss, stale, elapsed, start_epoch = ([], float('inf'), 0, 0.0, 1)
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        if checkpoint['config'] != cfg or checkpoint['metadata']['manifest_sha256'] != audit['manifest_sha256']:
            raise ValueError('Resume config/manifest mismatch.')
        model.load_state_dict(checkpoint['model_state'])
        history, best_loss, stale = (checkpoint['history'], checkpoint['best_loss'], checkpoint['stale'])
        elapsed, start_epoch = (checkpoint['elapsed'], checkpoint['epoch'] + 1)
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'], generator, ctx)
        if not (out / 'checkpoints/best_model.pt').exists():
            raise FileNotFoundError('Resume needs the original best checkpoint as well as last.pt.')
        if checkpoint['phase'] == 'finetune' and stale >= t['early_stopping_patience']:
            if ctx.main:
                print('This run already reached early stopping; no further updates are allowed.')
            dist_utils.barrier(ctx)
            return out
    ckpt_dir = out / 'checkpoints'
    gpu_names = dist_utils.all_gather_object(ctx, torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine())
    if ctx.main:
        out.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
        (out / 'run_config_used.json').write_text(json.dumps(cfg, indent=2))
        versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'scikit-learn', 'Pillow', 'PyYAML')}
        environment = {'python': platform.python_version(), 'packages': versions, 'device': str(device), 'cuda': torch.version.cuda, 'gpu': gpu_names[0], 'gpus': gpu_names, 'world_size': ctx.world, 'per_device_batch': batch, 'gradient_accumulation': accumulation, 'presize_cache': cache is not None}
        try:
            environment['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
            environment['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError):
            environment['git_commit'] = None
        (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    metadata = {'model_name': cfg['model_name'], 'run_name': cfg['run_name'], 'manifest_sha256': audit['manifest_sha256'], 'dataset_version': audit['dataset_version'], 'preprocessing': preproc, 'label_mapping': {'real': 0, 'fake': 1}, 'threshold': 0.5, 'selection': 'validation_loss', 'smoke': cfg.get('smoke', False), 'model_config': cfg['model'], 'module': cfg['module'], 'seed': cfg['seed'], 'world_size': ctx.world}
    current_phase, optimizer, scheduler, network = (None, None, None, None)
    epoch_wall = []
    reason = dist_utils.broadcast_object(ctx, preflight_reason(history, start_epoch, t['epochs'], time.perf_counter() - invocation_start) if ctx.main else None)
    if reason:
        if ctx.main:
            save_pause(out, ckpt_dir, start_epoch - 1, reason, time.perf_counter() - invocation_start)
        dist_utils.barrier(ctx)
        return out
    for epoch in range(start_epoch, t['epochs'] + 1):
        warmup = epoch <= t.get('warmup_epochs', 0)
        phase = 'warmup' if warmup else 'finetune'
        if phase != current_phase:
            phase_parameters(model, cfg, warmup)
            network = None
            network = wrap(model, ctx)
            lr = t.get('warmup_lr', t['lr']) if warmup else t['lr']
            cls = torch.optim.Adam if t.get('optimizer') == 'adam' else torch.optim.AdamW
            optimizer = cls(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=t['weight_decay'])
            total_steps = max(1, t['epochs'] * math.ceil(len(loaders['train']) / accumulation))
            warm_steps = int(total_steps * t.get('warmup_steps_pct', 0))

            def schedule(step):
                if step < warm_steps:
                    return (step + 1) / max(1, warm_steps)
                return max(0.0, (total_steps - step) / max(1, total_steps - warm_steps))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
            if checkpoint and checkpoint['phase'] == phase:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            elif current_phase is not None or checkpoint:
                stale = 0
            current_phase = phase
        if isinstance(loaders['train'].sampler, DistributedSampler):
            loaders['train'].sampler.set_epoch(epoch)
        tick = time.perf_counter()
        train_loss, train_metrics, train_stats = run_epoch(network, loaders['train'], device, ctx, optimizer, scheduler, scaler, accumulation, t.get('grad_clip_norm', 1.0), amp, log_every=t.get('progress_every', 0))
        val_loss, val_metrics, val_stats = run_epoch(network, loaders['val'], device, ctx, amp=amp, log_every=t.get('progress_every', 0))
        elapsed += time.perf_counter() - tick
        improved = val_loss < best_loss
        if improved:
            best_loss, stale = (val_loss, 0)
        else:
            stale += 1
        rng_all = dist_utils.all_gather_object(ctx, rng_state(generator, ctx))
        stop = dist_utils.broadcast_object(ctx, not warmup and stale >= t['early_stopping_patience'])
        if ctx.main:
            history.append({'epoch': epoch, 'phase': phase, 'train_loss': train_loss, 'val_loss': val_loss, 'train_accuracy': train_metrics['accuracy'], 'val_accuracy': val_metrics['accuracy'], 'train_f1': train_metrics['f1_score'], 'val_f1': val_metrics['f1_score'], 'trainable_parameters': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'lr': optimizer.param_groups[0]['lr'], 'elapsed_seconds': elapsed, 'train_precision': train_metrics['precision'], 'train_recall': train_metrics['recall'], 'train_roc_auc': train_metrics['roc_auc'], 'train_average_precision': train_metrics['average_precision'], 'train_balanced_accuracy': train_metrics['balanced_accuracy'], 'train_specificity': train_metrics['specificity'], 'train_mcc': train_metrics['mcc'], 'val_precision': val_metrics['precision'], 'val_recall': val_metrics['recall'], 'val_roc_auc': val_metrics['roc_auc'], 'val_average_precision': val_metrics['average_precision'], 'val_balanced_accuracy': val_metrics['balanced_accuracy'], 'val_specificity': val_metrics['specificity'], 'val_mcc': val_metrics['mcc'], 'train_images_per_second': train_stats['images_per_second'], 'train_data_wait_fraction': train_stats['data_wait_fraction'], 'world_size': ctx.world})
            metadata.update(training_time_seconds=elapsed, total_parameters=sum((p.numel() for p in model.parameters())), trainable_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)))
            (out / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
            if improved:
                torch.save({'model_state': model.state_dict(), 'metadata': dict(metadata, best_epoch=epoch)}, ckpt_dir / 'best_model.tmp')
                (ckpt_dir / 'best_model.tmp').replace(ckpt_dir / 'best_model.pt')
            state = {'model_state': model.state_dict(), 'metadata': metadata, 'config': cfg, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'phase': phase, 'history': history, 'best_loss': best_loss, 'stale': stale, 'elapsed': elapsed, 'rng': rng_all if ctx.world > 1 else rng_all[0]}
            temporary = ckpt_dir / 'last.tmp'
            torch.save(state, temporary)
            temporary.replace(ckpt_dir / 'last.pt')
            pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
            print(f"epoch={epoch} phase={phase} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} train_img/s={train_stats['images_per_second']:.1f} data_wait={100 * train_stats['data_wait_fraction']:.0f}% epoch_min={(time.perf_counter() - tick) / 60:.1f}", flush=True)
        if stop:
            break
        if ctx.main:
            learning_curves(history, out / 'learning_curves.png')
        epoch_wall.append(time.perf_counter() - tick)
        reason = dist_utils.broadcast_object(ctx, pause_reason(epoch, start_epoch, t['epochs'], epoch_wall, time.perf_counter() - invocation_start) if ctx.main else None)
        if reason:
            if ctx.main:
                save_pause(out, ckpt_dir, epoch, reason, time.perf_counter() - invocation_start)
            dist_utils.barrier(ctx)
            return out
    if ctx.main:
        if not history:
            raise ValueError('No training history; check resume epoch.')
        learning_curves(history, out / 'learning_curves.png')
        best = torch.load(ckpt_dir / 'best_model.pt', map_location='cpu', weights_only=True)
        best['metadata']['training_time_seconds'] = elapsed
        torch.save(best, ckpt_dir / 'best_model.tmp')
        (ckpt_dir / 'best_model.tmp').replace(ckpt_dir / 'best_model.pt')
        (out / 'training_complete.json').write_text(json.dumps({'status': 'smoke_only' if cfg.get('smoke') else 'trained', 'epochs': len(history), 'seconds': elapsed}, indent=2))
        (out / 'training_paused.json').unlink(missing_ok=True)
    dist_utils.barrier(ctx)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='configs/vit.yaml')
    parser.add_argument('--data-root')
    parser.add_argument('--manifest')
    parser.add_argument('--device')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, value in (('root', args.data_root), ('manifest', args.manifest)):
        if value:
            cfg['data'][key] = value
    if args.device:
        cfg['device'] = args.device
    if args.smoke:
        cfg.update(smoke=True, run_name=cfg['run_name'] + '_smoke')
        cfg['train'].update(epochs=1, warmup_epochs=0)
        cfg['data'].update(batch_size=2, num_workers=0)
    print(f'Training artifacts: {train(cfg, args.resume)}. Test sets have not been evaluated.')


def chunk_budget():
    """Per-invocation limits that let a Kaggle saved run finish and publish its outputs."""
    epochs = int(os.environ.get("WISH_EPOCHS_PER_INVOCATION", "3"))
    if epochs < 1:
        raise ValueError("WISH_EPOCHS_PER_INVOCATION must be positive")
    seconds = float(os.environ.get("WISH_TIME_BUDGET_SECONDS", "0"))
    if seconds < 0:
        raise ValueError("WISH_TIME_BUDGET_SECONDS must not be negative")
    return (epochs, seconds)

def epoch_estimate(spans):
    """Conservative next-epoch cost from measured epochs only; never an invented constant."""
    recent = [float(span) for span in spans[-3:] if span is not None and float(span) > 0]
    if not recent:
        return None
    return max(recent) * 1.15 + 120.0

def history_epoch_spans(history):
    times = [float(row["elapsed_seconds"]) for row in history if row.get("elapsed_seconds") is not None]
    return [b - a for a, b in zip(times, times[1:])] if len(times) > 1 else times

def pause_reason(epoch, start_epoch, total_epochs, spans, spent):
    epochs, seconds = chunk_budget()
    if epoch >= total_epochs:
        return None
    if epoch - start_epoch + 1 >= epochs:
        return "saved_run_epoch_budget"
    estimate = epoch_estimate(spans)
    if seconds and estimate is not None and spent + estimate > seconds:
        return "saved_run_time_budget"
    return None

def preflight_reason(history, start_epoch, total_epochs, spent):
    epochs, seconds = chunk_budget()
    if start_epoch > total_epochs or not seconds:
        return None
    estimate = epoch_estimate(history_epoch_spans(history))
    if estimate is not None and spent + estimate > seconds:
        return "insufficient_time_budget"
    return None

def save_pause(out, ckpt_dir, epoch, reason, spent):
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_paused.json").write_text(json.dumps({"status": "paused_not_complete", "last_epoch": epoch,
        "reason": reason, "seconds_used_this_invocation": round(spent, 1), "resume": str(ckpt_dir / "last.pt")}, indent=2))
    print(f"Training chunk saved ({reason}). Resume the same experiment from these saved outputs.", flush=True)


if __name__ == '__main__':
    main()
