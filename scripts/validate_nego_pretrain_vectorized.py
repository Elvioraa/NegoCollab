"""Run one real vectorized pretraining update; never save model checkpoints."""
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

from opencood.tools.nego_protocol import ROOT, file_sha, read_yaml, seed_all, validate_config, validate_runtime_batch, verify_lock
from opencood.tools.nego_state import model_for, set_training
from opencood.tools.run_nego_doma import criterion_for, loader, loss_for, parsed, require_vectorized_supervision, to_device


STAGES = ('pretrain/m1', 'pretrain/m2', 'pretrain/m3', 'pretrain/m4')
EXPECTED_UNUSED_M2 = frozenset({
    'encoder_m2.camencode.trunk._conv_head.weight',
    'encoder_m2.camencode.trunk._bn1.weight',
    'encoder_m2.camencode.trunk._bn1.bias',
    'encoder_m2.camencode.trunk._fc.weight',
    'encoder_m2.camencode.trunk._fc.bias',
})


def check_config(cfg, stage, batch_size):
    if stage not in STAGES or cfg['nego_protocol']['stage'] != stage:
        raise ValueError('Requested stage does not match pretraining configuration')
    if cfg['nego_protocol']['dataset'] != 'v2xset' or cfg['nego_protocol']['phase'] != 'pretrain':
        raise ValueError('Expected V2XSet pretraining configuration')
    if isinstance(batch_size, bool) or batch_size < 1 or cfg['train_params']['batch_size'] != batch_size:
        raise ValueError('Requested batch size does not match configuration')
    if cfg['seed'] != 303:
        raise ValueError('Validation requires seed 303 in the supplied YAML')
    options = cfg.get('nego', {}).get('multi_batch', {})
    if options.get('enabled') is not True or options.get('mode') != 'vectorized':
        raise ValueError('Vectorized multi-batch must be enabled in the supplied YAML')
    validate_config(cfg)
    validate_runtime_batch(cfg)


def sensor_statistics(ego, stage, batch_size):
    member = stage.split('/')[1]
    counts = ego['record_len'].tolist()
    if len(counts) != batch_size or any(n < 1 for n in counts):
        raise ValueError('Actual batch size or CAV counts do not match')
    total = sum(counts)
    if ego['agent_modality_list'] != [member] * total:
        raise ValueError('Batch contains unexpected CAV modalities')
    if ego['record_len_modality'][member].tolist() != counts:
        raise ValueError('Per-scene modality counts do not match')
    inputs = ego['inputs_' + member]
    result = {'member': member, 'actual_batch_size': len(counts), 'record_len': counts,
              'cav_count': total}
    if member in ('m1', 'm3'):
        coords = inputs['voxel_coords']
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
            raise ValueError('Invalid or empty voxel coordinates')
        if int(coords[:, 0].min()) < 0 or int(coords[:, 0].max()) >= total:
            raise ValueError('Voxel CAV index outside batch')
        result.update({'voxel_coords_shape': list(coords.shape), 'voxel_count': len(coords)})
    else:
        shapes = {}
        for key in ('imgs', 'rots', 'trans', 'intrins', 'post_rots', 'post_trans'):
            value = inputs[key]
            if not isinstance(value, torch.Tensor) or value.shape[0] != total:
                raise ValueError('Camera CAV count mismatch: ' + key)
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError('Nonfinite camera input: ' + key)
            shapes[key] = list(value.shape)
        imgs = inputs['imgs']
        if imgs.ndim != 5 or imgs.shape[1] < 1:
            raise ValueError('Invalid camera image dimensions')
        camera_count = imgs.shape[1]
        for key in ('rots', 'intrins', 'post_rots'):
            if tuple(inputs[key].shape) != (total, camera_count, 3, 3):
                raise ValueError('Invalid camera matrix dimensions: ' + key)
        for key in ('trans', 'post_trans'):
            if tuple(inputs[key].shape) != (total, camera_count, 3):
                raise ValueError('Invalid camera vector dimensions: ' + key)
        result['camera_tensor_shapes'] = shapes
    return result


def gpu_uuid():
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    matches = [row.split(',')[0].strip() for row in rows
               if len(row.split(',')) >= 2 and row.split(',')[1].strip() == str(os.getpid())]
    if len(matches) != 1:
        raise RuntimeError('Cannot uniquely identify the CUDA process GPU UUID')
    return matches[0]


def run_update(model, criterion, cfg, batch, stage):
    """Use the training loss entrypoint exactly once and inspect its real output."""
    outputs, sparse_sizes = [], []
    def capture_output(_module, _args, output):
        outputs.append(output)
    handles = [model.register_forward_hook(capture_output)]
    if stage == 'pretrain/m3':
        def capture_sparse(_module, args):
            sparse_sizes.append(int(args[0]['batch_size']))
        handles.append(model.encoder_m3.spconv_block.register_forward_pre_hook(capture_sparse))
    try:
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not params:
            raise ValueError('No trainable parameters')
        expected_unused = EXPECTED_UNUSED_M2 if stage == 'pretrain/m2' else frozenset()
        absent = expected_unused - {name for name, _ in params}
        if absent:
            raise ValueError('Expected unused parameters missing from trainable model: ' + ', '.join(sorted(absent)))
        options = cfg['optimizer']
        optimizer = getattr(torch.optim, options['core_method'])(
            [p for _, p in params], lr=options['lr'], **options.get('args', {}))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = loss_for(model, criterion, batch, cfg, 0)
        if len(outputs) != 1:
            raise ValueError('Expected exactly one model forward')
        shapes = {}
        for key in ('cls_preds', 'reg_preds', 'dir_preds'):
            value = outputs[0][key]
            if value.ndim != 4 or value.shape[0] != len(batch['ego']['record_len']) or not torch.isfinite(value).all():
                raise ValueError('Invalid detection output: ' + key)
            shapes[key] = list(value.shape)
        if any(shapes[key][-2:] != shapes['cls_preds'][-2:] for key in ('reg_preds', 'dir_preds')):
            raise ValueError('Detection output spatial dimensions differ')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        if stage == 'pretrain/m3' and sparse_sizes != [int(batch['ego']['record_len'].sum())]:
            raise ValueError('SECOND sparse batch size does not match actual CAV count')
        loss.backward()
        unused = sorted(name for name, p in params if p.grad is None and name in expected_unused)
        missing = [name for name, p in params if p.grad is None and name not in expected_unused]
        if missing:
            raise ValueError('Unexpected missing trainable gradients: ' + ', '.join(missing))
        gradients = [(name, p) for name, p in params if p.grad is not None]
        if not gradients:
            raise ValueError('No trainable gradients')
        if any(not torch.isfinite(p.grad).all() for _, p in gradients):
            raise FloatingPointError('Nonfinite trainable gradient')
        gradient_norm = 0.0
        for _, p in gradients:
            gradient_norm = math.hypot(gradient_norm, float(torch.linalg.vector_norm(p.grad.detach())))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError('Nonfinite gradient norm')
        before = {name: p.detach().cpu().clone() for name, p in params}
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        updated = 0
        updated_elements = 0
        updated_with_gradient = 0
        for name, parameter in params:
            current = parameter.detach().cpu()
            if not torch.isfinite(current).all():
                raise FloatingPointError('Nonfinite parameter after optimizer step: ' + name)
            changed = int(torch.count_nonzero(current != before[name]))
            updated += int(changed > 0)
            updated_elements += changed
            if parameter.grad is not None and changed:
                updated_with_gradient += 1
        if not updated_with_gradient:
            raise ValueError('Optimizer did not update any gradient-bearing parameter')
        report = {'loss': float(loss.detach()), 'output_shapes': shapes,
                  'trainable_parameter_tensors': len(params),
                  'trainable_parameter_count': sum(p.numel() for _, p in params),
                  'gradient_tensor_count': len(gradients), 'gradient_norm': gradient_norm,
                  'unused_trainable_parameters': unused,
                  'updated_parameter_tensors': updated, 'updated_parameter_count': updated_elements,
                  'optimizer': options['core_method'],
                  'learning_rates': [group['lr'] for group in optimizer.param_groups],
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'batch_seconds': elapsed, 'samples_per_second': len(batch['ego']['record_len']) / elapsed}
        if stage == 'pretrain/m3':
            report['second_sparse_batch_size'] = sparse_sizes[0]
        return report
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    cfg = read_yaml(Path(args.config))
    check_config(cfg, args.stage, args.batch_size)
    verify_lock('v2xset')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device must be visible')
    torch.zeros(1, device='cuda:0')
    torch.cuda.synchronize()
    uuid = gpu_uuid()
    seed_all(cfg['seed'])
    runtime = parsed(cfg)
    from opencood.data_utils.datasets.nego_protocol_dataset import build_protocol_dataset
    dataset = build_protocol_dataset(runtime, train=True)
    dataset.reinitialize()
    generator = torch.Generator().manual_seed(cfg['seed'])
    batch = next(iter(loader(dataset, cfg, True, generator)))
    if batch is None:
        raise ValueError('Empty batch')
    report = sensor_statistics(batch['ego'], args.stage, args.batch_size)
    require_vectorized_supervision(batch, cfg)
    batch = to_device(batch, torch.device('cuda:0'))
    model = model_for(runtime).to('cuda:0')
    set_training(model, cfg)
    criterion = criterion_for(cfg).to('cuda:0')
    report.update(run_update(model, criterion, cfg, batch, args.stage))
    report.update({'stage': args.stage, 'seed': cfg['seed'], 'batch_size': args.batch_size,
                   'git_sha': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                   'config_sha256': file_sha(Path(args.config)), 'gpu_uuid': uuid})
    destination = Path(args.report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / ('{}_bs{}.json'.format(args.stage.replace('/', '_'), args.batch_size))
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    print('PRETRAIN_VECTOR_VALIDATION_PASS', flush=True)


if __name__ == '__main__':
    main()
