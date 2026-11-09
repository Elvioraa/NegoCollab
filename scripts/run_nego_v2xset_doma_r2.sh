#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
EXP="$ROOT/opencood/logs/NEGO_V2XSET_DOMA_R2_1GPU"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--resume" ) ]]; then
    printf 'Usage: %s [--resume]\n' "$0" >&2; exit 2
fi
[[ -d "$EXP" && -f "$EXP/runtime.env" ]] || { echo 'Missing experiment directory or runtime.env' >&2; exit 1; }
trap 'rc=$?; trap - ERR; printf "%s orchestrator shell failed exit=%s\n" "$(date -u +%FT%TZ)" "$rc" >> "$EXP/orchestrator.log"; exit "$rc"' ERR
set -a
source "$EXP/runtime.env"
set +a
export NEGO_TRAIN_NUM_WORKERS="${NEGO_TRAIN_NUM_WORKERS:-8}"
export NEGO_VAL_NUM_WORKERS="${NEGO_TRAIN_NUM_WORKERS:-8}"
: "${NEGO_EXPECTED_GIT_SHA:?Missing NEGO_EXPECTED_GIT_SHA in runtime.env}"
: "${CUDA_VISIBLE_DEVICES:?Missing CUDA_VISIBLE_DEVICES in runtime.env}"
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 1; }
exec 9>"$EXP/orchestrator.lock"
flock -n 9 || { echo 'Experiment lock already held' >&2; exit 1; }
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/scripts/nego_runtime_batch:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python - "$ROOT" "$EXP" "${1:-}" <<'PY'
import csv
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from opencood.tools.nego_protocol import (digest, file_sha, parents, read_yaml,
    validate_config, validate_runtime_batch, verify_lock)
from opencood.tools.nego_state import configuration_identity
from opencood.tools.run_nego_doma import inference_paths

EXPECTED = {
    'pretrain/m1': (4, 31, .002, '091ac3754070'),
    'pretrain/m2': (4, 31, .002, '17a40f363fe2'),
    'pretrain/m3': (8, 31, .002, 'c4c79a6ba417'),
    'pretrain/m4': (4, 31, .002, '92cdb99341df'),
    'alliance/nego': (1, 12, .001, '61f58eef2619'),
    'alliance/ft': (1, 9, .001, '62c1ca1d09a8'),
    'm3/align': (1, 10, .001, 'd40ee46cc882'),
    'm3/ft': (1, 12, .001, '213043d98d32'),
    'm4/align': (1, 12, .001, '8e18652a32ff'),
    'm4/ft': (1, 12, .001, 'db793fc5d6b8'),
    'final_infer': (1, 0, .002, 'c7a7acee603e'),
}
GPU_UUID = 'GPU-8560f39f-f7a9-af3d-6f76-815b1aea0105'
MEMBERS = ('m1', 'm2', 'm3', 'm4')
AP_KEYS = ('AP@0.3', 'AP@0.5', 'AP@0.7')
TRAIN = list(EXPECTED)[:-1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def log(exp, message):
    line = '{} {}'.format(now(), message)
    print(line, flush=True)
    with (exp / 'orchestrator.log').open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
        stream.flush()


def validate_configs(exp):
    configs = {}
    verify_lock('v2xset')
    for stage, (batch, epochs, lr, prefix) in EXPECTED.items():
        cfg = read_yaml(exp / stage / 'config.yaml')
        validate_config(cfg)
        validate_runtime_batch(cfg)
        protocol, train = cfg['nego_protocol'], cfg['train_params']
        if (protocol['stage'] != stage or protocol['dataset'] != 'v2xset' or
            protocol['parents'] != parents(stage) or cfg['seed'] != 303):
            raise ValueError('Audited configuration mismatch: ' + stage)
        if stage.startswith('pretrain/'):
            options = cfg.get('nego', {}).get('multi_batch', {})
            if options.get('enabled') is not True or options.get('mode') != 'vectorized':
                raise ValueError('Pretraining requires vectorized mode: ' + stage)
        configs[stage] = cfg
    return configs


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'], text=True)
    return [(x[0].strip(), x[1].strip()) for x in (row.split(',') for row in output.splitlines()) if len(x) >= 2]


def preflight(root, exp):
    expected = os.environ['NEGO_EXPECTED_GIT_SHA']
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True).strip()
    if not re.fullmatch('[0-9a-f]{40}', expected) or head != expected:
        raise ValueError('Git HEAD does not match NEGO_EXPECTED_GIT_SHA')
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=str(root), text=True)
    if status.strip():
        raise ValueError('Git working tree is not clean')
    import torch
    if sys.version_info[:2] != (3, 8) or torch.__version__.split('+')[0] != '2.0.1':
        raise ValueError('DOMA requires Python 3.8 and PyTorch 2.0.1')
    prefix = os.environ.get('CONDA_PREFIX', '')
    try:
        if not prefix:
            raise ValueError('No conda prefix')
        Path(sys.executable).resolve().relative_to(Path(prefix).resolve())
    except ValueError:
        raise ValueError('DOMA conda environment is not active')
    if os.environ.get('CONDA_DEFAULT_ENV') != 'DOMA':
        raise ValueError('DOMA conda environment is not active')
    for split in ('train', 'validate', 'test'):
        if not (root / 'dataset' / 'V2XSET' / split).is_dir():
            raise FileNotFoundError('Missing dataset split: ' + split)
    physical = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    if physical != GPU_UUID:
        raise RuntimeError('Physical GPU 2 UUID differs from required UUID')
    if any(uuid == GPU_UUID for uuid, _ in gpu_processes()):
        raise RuntimeError('Target GPU already has compute processes')
    probe = r'''import os, subprocess, torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('Expected exactly one visible CUDA device')
x = torch.zeros(1, device='cuda:0'); torch.cuda.synchronize()
rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True).splitlines()
hits = [p[0].strip() for p in (r.split(',') for r in rows) if len(p) >= 2 and p[1].strip() == str(os.getpid())]
if hits != [sys.argv[1]]: raise RuntimeError('Physical GPU UUID mismatch: ' + repr(hits))
'''
    subprocess.run([sys.executable, '-c', 'import sys\n' + probe, GPU_UUID], check=True)
    if any(uuid == GPU_UUID for uuid, _ in gpu_processes()):
        raise RuntimeError('Target GPU occupied after CUDA probe')
    return head


def parent_shas(exp, stage):
    return {name: file_sha(exp / name / 'best.pth') for name in parents(stage)}


def checkpoint_check(root, exp, stage, configs, filename, resume=False):
    # Construct the expected model and use the runner's own strict checkpoint loader
    # in a short-lived child, so camera model initialization cannot retain GPU memory.
    code = r'''import json, sys
from pathlib import Path
from opencood.tools.nego_protocol import read_yaml, file_sha, digest, parents
import torch
from opencood.tools.nego_state import load_stage, model_for, strict_load, set_training, restore_rng
from opencood.tools.run_nego_doma import parsed, restore_training_schedule
root, exp, stage, name, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
cfg = read_yaml(exp / stage / 'config.yaml')
model = model_for(parsed(cfg))
kwargs = {'cfg':cfg, 'protocol_sha':file_sha(root / cfg['nego_protocol']['protocol_lock']), 'model':model}
if mode == 'resume': kwargs['mode'] = 'resume'
else: kwargs.update(mode='pretrained', target_cfg=cfg)
ckpt = load_stage(exp / stage / name, **kwargs)
strict_load(model, ckpt['model'])
if mode == 'resume':
    set_training(model, cfg)
    options = cfg['optimizer']
    optimizer = getattr(torch.optim, options['core_method'])(
        [p for p in model.parameters() if p.requires_grad], lr=options['lr'], **options.get('args', {}))
    scheduler = restore_training_schedule(optimizer, cfg, ckpt['resume'], 1)
    restore_rng(ckpt['resume']['rng'], torch.Generator())
provenance = ckpt['meta']['parents']
if set(provenance) != set(parents(stage)): raise ValueError('Checkpoint parent identities differ')
for parent in parents(stage):
    parent_cfg = read_yaml(exp / parent / 'config.yaml')
    item = provenance[parent]
    actual_parent = load_stage(exp / parent / 'best.pth', cfg=parent_cfg,
                              protocol_sha=file_sha(root / parent_cfg['nego_protocol']['protocol_lock']))
    if item['sha256'] != file_sha(exp / parent / 'best.pth') or item['config_sha'] != actual_parent['meta']['config_sha']:
        raise ValueError('Checkpoint parent source mismatch: ' + parent)
print(json.dumps({'epoch':ckpt['meta']['completed_epoch'], 'parents':provenance, 'owners':ckpt['meta']['owners']}))
'''
    output = subprocess.check_output([sys.executable, '-c', code, str(root), str(exp), stage,
                                      filename, 'resume' if resume else 'pretrained'], cwd=str(root), text=True)
    return json.loads(output.strip().splitlines()[-1])


def train_decision(exp, stage, epochs, resume, check):
    directory = exp / stage
    artifacts = [p for p in directory.iterdir() if p.name != 'config.yaml']
    last, best = directory / 'last.pth', directory / 'best.pth'
    if not resume:
        if artifacts:
            raise FileExistsError('Existing training artifacts: ' + stage)
        return 'start'
    if not artifacts:
        return 'start'
    if not last.is_file():
        raise ValueError('Existing artifacts without last checkpoint: ' + stage)
    last_info = check(stage, 'last.pth', True)
    epoch = last_info['epoch']
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError('Invalid completed epoch: ' + stage)
    if best.exists():
        best_epoch = check(stage, 'best.pth', False)['epoch']
        if not isinstance(best_epoch, int) or not 0 < best_epoch <= epoch:
            raise ValueError('Invalid best checkpoint epoch: ' + stage)
    elif epoch >= epochs:
        raise ValueError('Completed stage lacks best checkpoint: ' + stage)
    # Any dangling sidecar, temporary write, or unexpected checkpoint is unsafe.
    for name in ('last.pth.sha256', 'best.pth.sha256'):
        path = directory / name
        if path.exists() and not (directory / name[:-7]).exists():
            raise ValueError('Dangling checkpoint sidecar: ' + stage)
    if list(directory.glob('*.pth.tmp')):
        raise ValueError('Incomplete checkpoint write: ' + stage)
    for path in directory.glob('*.pth'):
        if path.name in ('last.pth', 'best.pth'):
            continue
        match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
        if not match:
            raise ValueError('Unexpected checkpoint: ' + str(path))
        saved_epoch = check(stage, path.name, True)['epoch']
        if saved_epoch != int(match.group(1)) or saved_epoch > epoch:
            raise ValueError('Inconsistent periodic checkpoint: ' + str(path))
    for path in directory.glob('*.pth.sha256'):
        if not Path(str(path)[:-7]).is_file():
            raise ValueError('Dangling checkpoint sidecar: ' + str(path))
    return 'skip' if epoch >= epochs else 'resume'


def checkpoint_snapshot(exp, stage, epochs, check):
    train_decision(exp, stage, epochs, True, check)
    last = check(stage, 'last.pth', True)['epoch']
    best = check(stage, 'best.pth', False)['epoch']
    if type(last) is not int or type(best) is not int or not 0 < best <= last:
        raise ValueError('Invalid OOM checkpoint epochs: ' + stage)
    return {'completed_epoch': last, 'best_epoch': best,
            'output_sha256': {(exp / stage / name).relative_to(exp).as_posix(): file_sha(exp / stage / name)
                for name in ('last.pth', 'last.pth.sha256', 'best.pth', 'best.pth.sha256')}}


def cuda_oom(text):
    return re.search(r'CUDA out of memory|CUDA error:\s*out of memory', text, re.IGNORECASE) is not None


def archive_oom(exp, stage, batch, reason, snapshot):
    # Copy, never move/delete: resume may update live checkpoints, so retain
    # every failed attempt's weights, logs, TensorBoard and validation outputs.
    label = stage.split('/')[-1] if stage.startswith('pretrain/') else stage.replace('/', '_')
    archive = exp / 'validation' / ('{}_oom_bs{}_{}'.format(
        label, batch, datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    source = exp / stage
    if any(path.is_symlink() for path in [source] + list(source.rglob('*'))):
        raise ValueError('Refusing to archive symlinked stage artifacts: ' + stage)
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copytree(str(source), str(archive / 'artifacts'))
    files = {p.relative_to(archive).as_posix(): file_sha(p) for p in (archive / 'artifacts').rglob('*') if p.is_file()}
    for path, sha in snapshot['output_sha256'].items():
        if file_sha(exp / path) != sha or files['artifacts/' + Path(path).name] != sha:
            raise ValueError('Checkpoint changed during OOM archive: ' + path)
    atomic_json(archive / 'audit.json', {'stage': stage, 'batch_size': batch, 'reason': reason,
                'archived_at': now(), 'checkpoints': snapshot, 'files': files})
    return archive.relative_to(exp).as_posix()


def validate_partial(exp, stage, entry, cfg, snapshot):
    if (entry.get('status') != 'partial_accepted' or
        not 20 < snapshot['completed_epoch'] or
        any(entry.get(k) != snapshot[k] for k in snapshot) or
        (entry.get('configuration_identity') is not None and
         entry['configuration_identity'] != configuration_identity(cfg)) or
        entry.get('parents') != parent_shas(exp, stage)):
        raise ValueError('Partial acceptance no longer matches verified checkpoints: ' + stage)
    if stage == 'pretrain/m4' and (snapshot['completed_epoch'], snapshot['best_epoch']) != (23, 22):
        raise ValueError('Historical M4 acceptance requires last=23, best=22')
    relative = Path(entry.get('oom_archive', ''))
    if relative.is_absolute() or '..' in relative.parts or relative.parts[:1] != ('validation',):
        raise ValueError('Invalid partial archive path: ' + stage)
    audit = exp / relative / 'audit.json'
    if not audit.is_file() or file_sha(audit) != entry.get('oom_audit_sha256'):
        raise ValueError('Missing or changed partial acceptance audit: ' + stage)


def check_result(exp, cav, cfg, assembled_sha, protocol_sha):
    paths = inference_paths(exp,cav)
    result = json.loads(paths['result'].read_text(encoding='utf-8'))
    enc = json.loads(paths['encoders'].read_text(encoding='utf-8'))
    if (result['dataset'] != 'v2xset' or result['stage'] != 'final_infer' or result['requested'] != cav or
        result['protocol_sha256'] != protocol_sha or
        result['checkpoint_sha256'] != assembled_sha or enc['assembled_sha256'] != assembled_sha or
        result['protocol_status'] != 'SERVER_VALIDATION_REQUIRED'):
        raise ValueError('Inference provenance mismatch: CAV' + str(cav))
    frames = result['frames']
    if not frames or not paths['ap'].is_file():
        raise ValueError('Incomplete inference result')
    for index, frame in enumerate(frames):
        members = frame['members']
        if (not members or len(members) > cav or members != list(MEMBERS[:len(members)]) or
            len(frame['forward_encoders']) != len(members) or set(frame['forward_encoders']) != set(members)):
            raise ValueError('Invalid frame encoder prefix')
        if (frame['sample_idx'] != index or frame['requested'] != cav or frame['realized'] != len(members) or
            len(frame['input_cav_ids']) != len(members) or
            frame['input_cav_ids'] != frame['in_range_cav_ids'][:len(members)] or
            not frame['scene'] or not frame['timestamp'] or
            isinstance(frame['gt_count'], bool) or not isinstance(frame['gt_count'], int) or frame['gt_count'] < 0 or
            not re.fullmatch('[0-9a-f]{64}', frame['gt_sha256'])):
            raise ValueError('Invalid frame or GT identity')
    calls = [member for frame in frames for member in frame['forward_encoders']]
    if enc['calls'] != calls or sorted(set(calls)) != enc['unique_members']:
        raise ValueError('Encoder participation does not match frame records')
    for key in AP_KEYS:
        value = result['AP'][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid AP: ' + key)
    total_gt = sum(frame['gt_count'] for frame in frames)
    if total_gt < 1 or any(result['statistics'][key]['gt'] != total_gt for key in ('0.3', '0.5', '0.7')):
        raise ValueError('Invalid inference GT statistics')
    with paths['ap'].open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 2 or rows[0] != ['CAV'] + list(AP_KEYS) or rows[1][0] != str(cav):
        raise ValueError('Invalid inference AP CSV')
    if any(float(rows[1][index]) != result['AP'][key] for index, key in enumerate(AP_KEYS, 1)):
        raise ValueError('Inference AP CSV disagrees with JSON')
    return result


def check_summary(exp, results):
    final = exp / 'final_infer'
    data = json.loads((final / 'AP_CAV1_4.json').read_text(encoding='utf-8'))
    with (final / 'AP_CAV1_4.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if len(data) != 4 or len(rows) != 5 or rows[0] != ['CAV'] + list(AP_KEYS):
        raise ValueError('Invalid summary structure')
    def identity(report):
        return [{k: v for k, v in frame.items() if k not in ('members','requested','realized','input_cav_ids','forward_encoders')}
                for frame in report['frames']]
    baseline = identity(results[0])
    for index, (report, entry, row) in enumerate(zip(results, data, rows[1:]), 1):
        if (not baseline or identity(report) != baseline or entry['CAV'] != index or
            entry['frames'] != len(report['frames']) or entry['protocol_status'] != report['protocol_status'] or
            row[0] != str(index)):
            raise ValueError('Summary frame, CAV or status mismatch')
        if any(report[k] != results[0][k] for k in ('dataset', 'protocol_sha256', 'checkpoint_sha256')):
            raise ValueError('Mixed inference provenance')
        for pos, key in enumerate(AP_KEYS, 1):
            if entry['AP'][key] != report['AP'][key] or float(row[pos]) != report['AP'][key]:
                raise ValueError('Summary AP mismatch: ' + key)


def main(root, exp, resume):
    state_path = exp / 'orchestrator_state.json'
    if state_path.exists():
        if not resume:
            raise FileExistsError('Existing orchestrator state; explicit --resume required')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('git_sha') != os.environ['NEGO_EXPECTED_GIT_SHA']:
            raise ValueError('Existing state belongs to another Git SHA')
    else:
        state = {'git_sha': os.environ['NEGO_EXPECTED_GIT_SHA'], 'experiment_started': now(), 'stages': {}}
    state['last_started'] = now()
    atomic_json(state_path, state)
    def update(name, **values):
        state['stages'].setdefault(name, {}).update(values)
        atomic_json(state_path, state)
    def check(stage, name, full=False):
        return checkpoint_check(root, exp, stage, configs, name, full)
    def check_assembly(assembled):
        info = check('final_infer', 'assembled.pth', False)
        manifest = json.loads(assembled.with_suffix('.sources.json').read_text(encoding='utf-8'))
        sha = file_sha(assembled)
        if (manifest['checkpoint_sha256'] != sha or manifest['parents'] != info['parents'] or
            manifest['owners'] != info['owners']):
            raise ValueError('Assembled sources manifest mismatch')
        return sha
    def run(name, stage, args, config_stage):
        training = args[0] == 'train'
        cfg = configs[config_stage]
        original_batch = cfg['train_params']['batch_size'] if training else None
        if training and 'NEGO_RUNTIME_BATCH_SIZE' in os.environ:
            requested = os.environ['NEGO_RUNTIME_BATCH_SIZE']
            if re.fullmatch(r'[0-9]+',requested) is None or int(requested)<1:
                raise ValueError('NEGO_RUNTIME_BATCH_SIZE must be a positive integer')
            original_batch = int(requested)
        previous = state['stages'].get(name, {})
        batch = (previous.get('batch_size', original_batch) if previous.get('configured_batch_size') == original_batch
                 and previous.get('oom_attempts') else original_batch) if training else None
        retries = previous.get('retry_count', 0)
        if training and (type(batch) is not int or batch < 1):
            raise ValueError('Invalid saved runtime batch: ' + name)
        if training and previous.get('status') == 'retrying':
            history = previous.get('oom_attempts', [])
            if (not history or not 0 < history[-1].get('completed_epoch', 0) <= 20 or
                batch != max(1, history[-1].get('batch_size', 0) // 2)):
                raise ValueError('Reduced batch has no matching OOM audit history: ' + name)
        while True:
            for parent in parents(config_stage):
                check(parent, 'best.pth', False)
            update(name, status='running', started=now(), ended=None, exit_code=None,
                   config_sha=digest(cfg), parents=parent_shas(exp, config_stage), failure=None,
                   batch_size=batch, configured_batch_size=original_batch, retry_count=retries)
            log(exp, 'START {} batch_size={} retry_count={}'.format(name, batch, retries))
            output = exp / stage / ('orchestrator_' + name.replace('/', '_') + '.log')
            offset = output.stat().st_size if output.exists() else 0
            env = os.environ.copy()
            if training:
                env['NEGO_RUNTIME_BATCH_SIZE'] = str(batch)
            with output.open('a', encoding='utf-8') as stream:
                proc = subprocess.run([sys.executable, '-u', '-m', 'opencood.tools.run_nego_doma'] + args,
                                      cwd=str(root), stdout=stream, stderr=subprocess.STDOUT, env=env)
            if proc.returncode == 0:
                return 'complete'
            update(name, status='failed', ended=now(), exit_code=proc.returncode,
                   failure='Runner failed; see ' + str(output))
            with output.open('rb') as stream:
                stream.seek(offset)
                text = stream.read().decode('utf-8', errors='replace')
            if not training or not cuda_oom(text):
                raise RuntimeError('Stage failed: {} exit {}'.format(name, proc.returncode))
            snapshot = checkpoint_snapshot(exp, stage, cfg['train_params']['epoches'], check)
            archive = archive_oom(exp, stage, batch, 'CUDA OOM', snapshot)
            attempts = state['stages'][name].get('oom_attempts', []) + [
                dict(snapshot, batch_size=batch, exit_code=proc.returncode, archive=archive, reason='CUDA OOM')]
            update(name, oom_attempts=attempts, oom_archive=archive, **snapshot)
            if stage == 'pretrain/m4':
                if (snapshot['completed_epoch'], snapshot['best_epoch'], batch) != (23, 22, 4):
                    raise ValueError('M4 OOM requires explicit review; M4 must not be retrained')
                accept_partial(stage, snapshot, batch, archive, 'historical_m4_oom_epoch23')
                return 'partial_accepted'
            if snapshot['completed_epoch'] > 20:
                accept_partial(stage, snapshot, batch, archive, 'cuda_oom_after_epoch20')
                return 'partial_accepted'
            if batch == 1 or snapshot['completed_epoch'] >= cfg['train_params']['epoches']:
                raise RuntimeError('CUDA OOM cannot be retried safely: ' + name)
            batch = max(1, batch // 2)
            retries += 1
            update(name, status='retrying', batch_size=batch, configured_batch_size=original_batch, retry_count=retries)
            if '--resume' not in args:
                args = args + ['--resume']
    def accept_partial(stage, snapshot, batch, archive, reason):
        update(stage, status='partial_accepted', ended=now(), failure='CUDA OOM', acceptance=reason,
               config_sha=digest(configs[stage]), configuration_identity=configuration_identity(configs[stage]),
               parents=parent_shas(exp, stage),
               batch_size=batch, retry_count=state['stages'].get(stage, {}).get('retry_count', 0),
               configured_epochs=configs[stage]['train_params']['epoches'], oom_archive=archive,
               oom_audit_sha256=file_sha(exp / archive / 'audit.json'), **snapshot)
        log(exp, 'PARTIAL_ACCEPTED {} completed_epoch={} best_epoch={}'.format(
            stage, snapshot['completed_epoch'], snapshot['best_epoch']))
    def done(name, output_paths=(), config_stage=None):
        output_hashes = {str(path.relative_to(exp)): file_sha(path) for path in output_paths}
        extra = {}
        if config_stage is not None:
            extra = {'config_sha': digest(configs[config_stage]), 'parents': parent_shas(exp, config_stage),
                     'started': state['stages'].get(name, {}).get('started', now())}
            if config_stage in TRAIN:
                snapshot = checkpoint_snapshot(exp, config_stage, configs[config_stage]['train_params']['epoches'], check)
                extra.update(completed_epoch=snapshot['completed_epoch'], best_epoch=snapshot['best_epoch'],
                             configured_epochs=configs[config_stage]['train_params']['epoches'])
        update(name, status='complete', ended=now(), exit_code=0, failure=None, output_sha256=output_hashes, **extra)
        log(exp, 'COMPLETE ' + name)
    current = 'preflight'
    try:
        head = preflight(root, exp)
        if head != state['git_sha']:
            raise ValueError('Existing state belongs to another Git SHA')
        configs = validate_configs(exp)
        # Reject stale or incomplete later-stage artifacts before any new training.
        for stage in TRAIN:
            train_decision(exp, stage, configs[stage]['train_params']['epoches'], resume, check)
            if state['stages'].get(stage, {}).get('status') == 'partial_accepted':
                validate_partial(exp, stage, state['stages'][stage], configs[stage],
                                 checkpoint_snapshot(exp, stage, configs[stage]['train_params']['epoches'], check))
        assembled_path = exp / 'final_infer' / 'assembled.pth'
        assembly_files = (assembled_path, Path(str(assembled_path) + '.sha256'),
                          assembled_path.with_suffix('.sources.json'))
        if any(p.exists() for p in assembly_files):
            if not resume or not all(p.is_file() for p in assembly_files):
                raise ValueError('Existing or incomplete assembled checkpoint')
            check_assembly(assembled_path)
        for cav in range(1, 5):
            paths = list(inference_paths(exp,cav).values())
            name = 'final_infer/infer_CAV' + str(cav)
            if any(path.exists() for path in paths):
                if (not resume or state['stages'].get(name, {}).get('status') != 'complete' or
                    not all(path.is_file() for path in paths)):
                    raise ValueError('Existing partial or unverified inference: CAV' + str(cav))
                actual = {str(path.relative_to(exp)): file_sha(path) for path in paths}
                if state['stages'][name].get('output_sha256') != actual:
                    raise ValueError('Existing inference output hashes differ: CAV' + str(cav))
        summaries = [exp / 'final_infer' / 'AP_CAV1_4.json', exp / 'final_infer' / 'AP_CAV1_4.csv']
        if any(p.exists() for p in summaries):
            if (not resume or not all(p.is_file() for p in summaries) or
                state['stages'].get('final_infer/summarize', {}).get('status') != 'complete'):
                raise ValueError('Existing partial or unverified summary')
            actual = {str(path.relative_to(exp)): file_sha(path) for path in summaries}
            if state['stages']['final_infer/summarize'].get('output_sha256') != actual:
                raise ValueError('Existing summary output hashes differ')
        for stage in TRAIN:
            current = stage
            decision = train_decision(exp, stage, configs[stage]['train_params']['epoches'], resume, check)
            if state['stages'].get(stage, {}).get('status') == 'partial_accepted':
                validate_partial(exp, stage, state['stages'][stage], configs[stage],
                                 checkpoint_snapshot(exp, stage, configs[stage]['train_params']['epoches'], check))
                log(exp, 'SKIP verified partial_accepted ' + stage)
                continue
            if stage == 'pretrain/m4' and resume and decision != 'start':
                snapshot = checkpoint_snapshot(exp, stage, configs[stage]['train_params']['epoches'], check)
                if (snapshot['completed_epoch'], snapshot['best_epoch']) == (23, 22):
                    archive = archive_oom(exp, stage, 4, 'User-confirmed historical M4 CUDA OOM after epoch 23', snapshot)
                    accept_partial(stage, snapshot, 4, archive, 'historical_m4_oom_epoch23')
                    continue
                if decision == 'resume':
                    raise ValueError('Historical M4 requires last=23, best=22; refusing retraining')
            if decision == 'skip':
                paths = [exp / stage / name for name in ('last.pth', 'last.pth.sha256', 'best.pth', 'best.pth.sha256')]
                done(stage, paths, stage)
                update(stage, disposition='verified_existing')
                log(exp, 'SKIP verified ' + stage)
                continue
            args = ['train', '--experiment', str(exp), '--stage', stage]
            if decision == 'resume': args.append('--resume')
            if run(stage, stage, args, stage) == 'partial_accepted':
                continue
            if train_decision(exp, stage, configs[stage]['train_params']['epoches'], True, check) != 'skip':
                raise ValueError('Training returned without complete checkpoints: ' + stage)
            done(stage, [exp / stage / name for name in ('last.pth', 'last.pth.sha256', 'best.pth', 'best.pth.sha256')], stage)
        current = 'final_infer/assemble'
        assembled = exp / 'final_infer' / 'assembled.pth'
        sources = assembled.with_suffix('.sources.json')
        if assembled.exists() or sources.exists() or Path(str(assembled) + '.sha256').exists():
            if not resume or not assembled.is_file() or not sources.is_file():
                raise ValueError('Existing/incomplete assembled checkpoint')
        else:
            run(current, 'final_infer', ['assemble', '--experiment', str(exp)], 'final_infer')
        cksha = check_assembly(assembled)
        done(current, [assembled, Path(str(assembled) + '.sha256'), sources], 'final_infer')
        protocol_sha = file_sha(root / configs['final_infer']['nego_protocol']['protocol_lock'])
        results = []
        for cav in range(1, 5):
            current = 'final_infer/infer_CAV' + str(cav)
            paths = list(inference_paths(exp,cav).values())
            if any(path.exists() for path in paths):
                if not resume or state['stages'].get(current, {}).get('status') != 'complete':
                    raise ValueError('Existing partial or unverified inference: CAV' + str(cav))
                actual = {str(path.relative_to(exp)): file_sha(path) for path in paths}
                if state['stages'][current].get('output_sha256') != actual:
                    raise ValueError('Existing inference output hashes differ: CAV' + str(cav))
            else:
                run(current, 'final_infer', ['infer', '--experiment', str(exp), '--use-cav', str(cav)], 'final_infer')
            results.append(check_result(exp, cav, configs['final_infer'], cksha, protocol_sha))
            done(current, paths, 'final_infer')
        current = 'final_infer/summarize'
        outputs = [exp / 'final_infer' / 'AP_CAV1_4.json', exp / 'final_infer' / 'AP_CAV1_4.csv']
        if any(path.exists() for path in outputs):
            if not resume or not all(path.is_file() for path in outputs) or state['stages'].get(current, {}).get('status') != 'complete':
                raise ValueError('Existing partial or unverified summary')
            actual = {str(path.relative_to(exp)): file_sha(path) for path in outputs}
            if state['stages'][current].get('output_sha256') != actual:
                raise ValueError('Existing summary output hashes differ')
        else:
            run(current, 'final_infer', ['summarize', '--experiment', str(exp)], 'final_infer')
        check_summary(exp, results)
        done(current, outputs, 'final_infer')
        state['experiment_ended'] = now()
        state['status'] = ('complete_with_partial' if any(
            entry.get('status') == 'partial_accepted' for entry in state['stages'].values()) else 'complete')
        atomic_json(state_path, state)
    except Exception as exc:
        existing = state['stages'].get(current, {})
        update(current, status='failed', ended=now(), exit_code=existing.get('exit_code') if existing.get('exit_code') is not None else 1,
               failure=str(exc))
        state['status'] = 'failed'
        state['experiment_ended'] = now()
        atomic_json(state_path, state)
        log(exp, 'FAILED {}: {}'.format(current, exc))
        raise


if __name__ == '__main__':
    try:
        main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == '--resume')
    except Exception:
        traceback.print_exc()
        sys.exit(1)
PY
