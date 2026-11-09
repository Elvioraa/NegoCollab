"""Interpreter-startup tests with synthetic samples and the unchanged runner loader."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / 'scripts' / 'nego_runtime_batch'
CODE = r'''
import json, sys, torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader as InternalDataLoader
from opencood.tools.run_nego_doma import loader, DataLoader as RunnerDataLoader

class Samples:
    def __len__(self): return 5
    def __getitem__(self, index): return index
    def collate_batch_train(self, items): return torch.tensor(items)
    def collate_batch_test(self, items): return torch.tensor(items)

# These constructor settings represent a YAML batch size of 4 without editing
# or importing the real Dataset/YAML processing chain.
cfg = {'train_params': {'batch_size': 4, 'num_workers': 0}}
generator = torch.Generator().manual_seed(303)
train = loader(Samples(), cfg, True, generator)
val = loader(Samples(), cfg, False, generator)
positional = DataLoader(list(range(5)), 4, True, num_workers=0)
print(json.dumps({'train': train.batch_size, 'val': val.batch_size,
    'train_batches': [len(batch) for batch in train],
    'val_batches': [len(batch) for batch in val],
    'positional': positional.batch_size,
    'same_class': RunnerDataLoader is DataLoader is InternalDataLoader,
    'hooked': getattr(RunnerDataLoader.__init__, '__nego_runtime_batch_hook__', False),
    'sitecustomize': sys.modules['sitecustomize'].__file__}))
'''


def child(value, code=CODE, overrides=None):
    env = os.environ.copy()
    env.pop('NEGO_RUNTIME_BATCH_SIZE', None)
    for name in ('NEGO_TRAIN_NUM_WORKERS', 'NEGO_VAL_NUM_WORKERS',
                 'NEGO_PREFETCH_FACTOR', 'NEGO_PIN_MEMORY',
                 'NEGO_PERSISTENT_WORKERS', 'NEGO_NON_BLOCKING_TRANSFER'):
        env.pop(name, None)
    if value is not None:
        env['NEGO_RUNTIME_BATCH_SIZE'] = value
    env['PYTHONPATH'] = os.pathsep.join([str(HOOK), str(ROOT), env.get('PYTHONPATH', '')])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.update(overrides or {})
    return subprocess.run([sys.executable, '-c', code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=90)


@pytest.mark.parametrize('value,actual,sizes', [
    (None, 4, [4, 1]), ('2', 2, [2, 2, 1]), ('1', 1, [1, 1, 1, 1, 1])])
def test_startup_hook_and_real_runner_loader(value, actual, sizes):
    result = child(value)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(report['sitecustomize']).resolve() == (HOOK / 'sitecustomize.py').resolve()
    assert report['train'] == report['positional'] == actual
    assert report['train_batches'] == sizes
    assert report['val'] == 1 and report['val_batches'] == [1] * 5
    assert report['same_class'] and report['hooked'] == (value is not None)
    if value is None:
        assert '[nego_runtime_batch]' not in result.stderr
    else:
        assert 'actual_batch_size={} original_yaml_batch_size=4'.format(actual) in result.stderr


@pytest.mark.parametrize('value', ['', '0', '-1', 'abc', '2.5', ' 2', 'True'])
def test_invalid_environment_is_fatal_before_user_code(value):
    result = child(value, "print('BODY_EXECUTED')")
    assert result.returncode != 0
    assert 'NEGO_RUNTIME_BATCH_SIZE must be a positive integer' in result.stderr
    assert 'BODY_EXECUTED' not in result.stdout


def test_runtime_batch_can_exceed_original_configuration():
    result = child('8')
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['train'] == 8 and report['val'] == 1


def test_hook_preserves_torch_sampler_validation():
    result = child('2', '''
from torch.utils.data import DataLoader, SequentialSampler
DataLoader(list(range(5)), batch_size=4, shuffle=True, sampler=SequentialSampler(range(5)))
''')
    assert result.returncode != 0
    assert 'sampler option is mutually exclusive with shuffle' in result.stderr


@pytest.mark.parametrize('runtime_env', [
    'NEGO_TRAIN_NUM_WORKERS=8\nNEGO_VAL_NUM_WORKERS=2\n',
    'NEGO_VAL_NUM_WORKERS=2\n'])
def test_orchestrator_validation_workers_and_batch_size(tmp_path, runtime_env):
    script = (ROOT / 'scripts/run_nego_v2xset_doma_r2.sh').read_text(encoding='utf-8')
    # Run only runtime.env loading and exports, stopping before any preflight,
    # experiment logging, locking, or training commands.
    prefix = script.split('set -a\n', 1)[1].split(': "${NEGO_EXPECTED_GIT_SHA', 1)[0]
    (tmp_path / 'runtime.env').write_text(runtime_env, encoding='utf-8')
    env = os.environ.copy()
    env['EXP'] = tmp_path.as_posix()
    env.pop('NEGO_TRAIN_NUM_WORKERS', None)
    env.pop('NEGO_VAL_NUM_WORKERS', None)
    bash = env.get('NEGO_TEST_BASH', 'bash')
    result = subprocess.run([bash, '-c', 'set -eu\nset -a\n' + prefix +
                             '\nprintf "%s,%s" "${NEGO_TRAIN_NUM_WORKERS:-}" "$NEGO_VAL_NUM_WORKERS"'],
                            env=env, capture_output=True, text=True, check=True)
    train_workers, val_workers = result.stdout.strip().split(',')
    assert val_workers == '8'
    overrides = {'NEGO_VAL_NUM_WORKERS': val_workers}
    if train_workers:
        overrides['NEGO_TRAIN_NUM_WORKERS'] = train_workers
    result = child('2', r'''
import json, torch
from opencood.tools.run_nego_doma import loader
class Samples(list):
    collate_batch_train = staticmethod(list)
    collate_batch_test = staticmethod(list)
# Represent the experiment's unchanged training workers=8 in memory.
cfg = {'train_params': {'batch_size': 4, 'num_workers': 8}}
g = torch.Generator().manual_seed(303)
train = loader(Samples(range(5)), cfg, True, g)
val = loader(Samples(range(5)), cfg, False, g)
print(json.dumps({'train_workers': train.num_workers, 'val_workers': val.num_workers,
                  'train_batch': train.batch_size, 'val_batch': val.batch_size}))
''', overrides=overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {'train_workers': 8, 'val_workers': 8, 'train_batch': 2, 'val_batch': 1}
