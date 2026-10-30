"""One real m1 Dataset/DataLoader/optimizer validation; never writes checkpoints."""
import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml

from opencood.data_utils.datasets.nego_protocol_dataset import build_protocol_dataset
from opencood.tools.nego_protocol import read_yaml, seed_all, validate_config, validate_runtime_batch
from opencood.tools.nego_state import model_for, set_training
from opencood.tools.run_nego_doma import criterion_for, loader, parsed, to_device


def query_gpu(pid):
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    matches = [row.strip() for row in text.splitlines() if len(row.split(',')) >= 2
               and row.split(',')[1].strip() == str(pid)]
    if len(matches) != 1:
        raise RuntimeError('Training process does not have exactly one visible GPU entry')
    expected = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    if matches[0].split(',')[0].strip() != expected:
        raise RuntimeError('CUDA process is not on physical GPU 2')
    utilization = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True).strip()
    return expected, utilization


def step(model, criterion, cfg, batch):
    batch = to_device(batch, torch.device('cuda:0'))
    ego = batch['ego']
    ego['stage'] = 'pretrain'
    ego['epoch'] = 0
    options = cfg['optimizer']
    optimizer = getattr(torch.optim, options['core_method'])(
        [p for p in model.parameters() if p.requires_grad], lr=options['lr'], **options.get('args', {}))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(ego)
    output['stage'] = 'pretrain'
    output['epoch'] = 0
    if output['cls_preds'].shape[0] != len(ego['record_len']):
        raise ValueError('Detection output does not have one entry per scene')
    loss = criterion(output, ego['label_dict'])
    if cfg['model']['args']['supervise_single']:
        loss = loss + criterion(output, ego['label_dict_single'], suffix='_single') * cfg['train_params'].get('single_weight', 1)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    gradients = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
    if not gradients or any(not torch.isfinite(v).all() for v in gradients.values()):
        raise FloatingPointError('Missing or nonfinite gradients')
    before = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if k in gradients}
    optimizer.step()
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    if not any(not torch.equal(before[k], p.detach().cpu()) for k, p in model.named_parameters() if k in before):
        raise ValueError('No gradient-bearing parameter updated')
    uuid, utilization = query_gpu(os.getpid())
    report = {'actual_batch': len(ego['record_len']), 'cavs': ego['record_len'].tolist(),
              'voxels': len(ego['inputs_m1']['voxel_coords']), 'loss': float(loss.detach()),
              'output_shapes': {k: list(output[k].shape) for k in ('cls_preds', 'reg_preds', 'dir_preds')},
              'gpu_uuid': uuid, 'gpu_util_percent_snapshot': utilization,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'batch_seconds': duration, 'samples_per_second': len(ego['record_len']) / duration}
    saved = {'output': {k: output[k].detach().cpu().clone() for k in ('cls_preds', 'reg_preds', 'dir_preds')},
             'loss': loss.detach().cpu().clone(), 'gradients': gradients,
             'updated': {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}}
    return report, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID' or os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise ValueError('Physical GPU mapping environment is required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device is required')
    probe = torch.zeros(1, device='cuda:0')
    torch.cuda.synchronize()
    uuid, utilization = query_gpu(os.getpid())
    print('GPU_VERIFIED uuid={} utilization_percent={}'.format(uuid, utilization), flush=True)
    del probe
    cfg = read_yaml(Path(args.config))
    if cfg['nego_protocol']['stage'] != 'pretrain/m1':
        raise ValueError('Only m1 pretrain is supported')
    cfg['train_params']['batch_size'] = args.batch_size
    cfg['nego'] = {'multi_batch': {'enabled': True, 'mode': 'vectorized'}}
    validate_config(cfg)
    validate_runtime_batch(cfg)
    with (Path(args.report_dir) / ('config_batch_{}.yaml'.format(args.batch_size))).open('x', encoding='utf-8') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    seed_all(cfg['seed'])
    runtime = parsed(cfg)
    dataset = build_protocol_dataset(runtime, train=True)
    dataset.reinitialize()
    generator = torch.Generator().manual_seed(cfg['seed'])
    batch = next(iter(loader(dataset, cfg, True, generator)))
    if len(batch['ego']['record_len']) != args.batch_size:
        raise ValueError('Unexpected partial validation batch')
    model = model_for(runtime).to('cuda:0')
    set_training(model, cfg)
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    criterion = criterion_for(cfg).to('cuda:0')
    print('START batch_size={} config={} pid={}'.format(args.batch_size, args.config, os.getpid()), flush=True)
    if args.batch_size == 1:
        native = copy.deepcopy(batch)
        del native['ego']['nego_m1_vectorized']
        del native['ego']['object_ids_by_sample']
        seed_all(cfg['seed'])
        native_report, native_result = step(model, criterion, cfg, native)
        model.load_state_dict(original, strict=True)
        set_training(model, cfg)
    seed_all(cfg['seed'])
    report, result = step(model, criterion, cfg, batch)
    if args.batch_size == 1:
        for key in ('loss',):
            torch.testing.assert_close(native_result[key], result[key], rtol=1e-5, atol=1e-6)
        for key in ('output', 'gradients', 'updated'):
            if native_result[key].keys() != result[key].keys():
                raise ValueError('Native and vectorized result key mismatch: ' + key)
            for name in native_result[key]:
                torch.testing.assert_close(native_result[key][name], result[key][name], rtol=1e-5, atol=1e-6)
        report['native_batch_one_regression'] = 'PASS'
    print(json.dumps(report, sort_keys=True), flush=True)
    print('M1_VECTOR_VALIDATION_PASS', flush=True)


if __name__ == '__main__':
    main()
