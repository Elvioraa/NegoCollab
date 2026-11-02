import torch
from torch import nn
from torch.utils.data import Dataset
import types
import sys

import opencood.tools.run_nego_doma as runner
from opencood.tools.nego_protocol import config_root, read_yaml


class TinyDataset(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index

    @staticmethod
    def collate_batch_train(items):
        return items

    @staticmethod
    def collate_batch_test(items):
        return items


def cfg():
    return read_yaml(config_root('v2xset') / 'pretrain/m1' / 'config.yaml')


def clear_loader_env(monkeypatch):
    for name in ('NEGO_TRAIN_NUM_WORKERS', 'NEGO_VAL_NUM_WORKERS',
                 'NEGO_PREFETCH_FACTOR', 'NEGO_PIN_MEMORY',
                 'NEGO_PERSISTENT_WORKERS', 'NEGO_NON_BLOCKING_TRANSFER'):
        monkeypatch.delenv(name, raising=False)


def test_missing_environment_preserves_legacy_loader_settings(monkeypatch):
    clear_loader_env(monkeypatch)
    settings = runner.dataloader_settings(cfg())
    assert settings == {'train_num_workers': cfg()['train_params'].get('num_workers', 0),
                        'val_num_workers': 0, 'prefetch_factor': None,
                        'pin_memory': False, 'persistent_workers': False,
                        'non_blocking_transfer': False}


def test_loader_omits_prefetch_for_zero_workers(monkeypatch):
    clear_loader_env(monkeypatch)
    captured = {}

    def capture(dataset, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(runner, 'DataLoader', capture)
    settings = runner.dataloader_settings(cfg())
    runner.loader(TinyDataset(), cfg(), False, torch.Generator(), settings)
    assert captured['num_workers'] == 0
    assert 'prefetch_factor' not in captured
    assert captured['persistent_workers'] is False


def test_loader_runtime_environment_values(monkeypatch):
    monkeypatch.setenv('NEGO_TRAIN_NUM_WORKERS', '8')
    monkeypatch.setenv('NEGO_VAL_NUM_WORKERS', '4')
    monkeypatch.setenv('NEGO_PREFETCH_FACTOR', '2')
    monkeypatch.setenv('NEGO_PIN_MEMORY', '1')
    monkeypatch.setenv('NEGO_PERSISTENT_WORKERS', '0')
    monkeypatch.setenv('NEGO_NON_BLOCKING_TRANSFER', '1')
    settings = runner.dataloader_settings(cfg())
    assert settings == {'train_num_workers': 8, 'val_num_workers': 4,
                        'prefetch_factor': 2, 'pin_memory': True,
                        'persistent_workers': False,
                        'non_blocking_transfer': True}


def test_invalid_environment_values_fail_clearly(monkeypatch):
    clear_loader_env(monkeypatch)
    monkeypatch.setenv('NEGO_TRAIN_NUM_WORKERS', 'eight')
    try:
        runner.dataloader_settings(cfg())
    except ValueError as exc:
        assert 'NEGO_TRAIN_NUM_WORKERS' in str(exc)
    else:
        raise AssertionError('invalid worker count was accepted')
    monkeypatch.setenv('NEGO_TRAIN_NUM_WORKERS', '0')
    monkeypatch.setenv('NEGO_NON_BLOCKING_TRANSFER', '1')
    try:
        runner.dataloader_settings(cfg())
    except ValueError as exc:
        assert 'requires NEGO_PIN_MEMORY=1' in str(exc)
    else:
        raise AssertionError('incompatible non-blocking transfer was accepted')


def test_recursive_transfer_accepts_non_blocking_flag():
    value = {'a': [torch.ones(1), (torch.zeros(2),)], 'b': 'metadata'}
    moved = runner.to_device(value, torch.device('cpu'), non_blocking=True)
    torch.testing.assert_close(moved['a'][0], value['a'][0])
    torch.testing.assert_close(moved['a'][1][0], value['a'][1][0])
    assert moved['b'] == 'metadata'


def test_evaluation_passes_non_blocking_through_tensor_transfer(monkeypatch):
    settings = {'train_num_workers': 8, 'val_num_workers': 4,
                'prefetch_factor': 2, 'pin_memory': True,
                'persistent_workers': False, 'non_blocking_transfer': True}
    batch = {'ego': {'agent_modality_list': ['m1'], 'cav_id_list': ['cav1']}}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_m1 = nn.Identity()

        def forward(self, ego):
            self.encoder_m1(torch.ones(1))
            return {}

    class EvalDataset:
        protocol_identities = [{}]

        def __len__(self):
            return 1

        def post_process(self, batch_data, output):
            return torch.empty((0, 7)), torch.empty((0,)), torch.ones((1, 7))

    fake_eval_utils = types.SimpleNamespace(
        caluclate_tp_fp=lambda boxes, scores, gt, stats, iou: stats[iou].update(gt=1),
        calculate_ap=lambda stats, iou: (0.0,))
    monkeypatch.setitem(sys.modules, 'opencood.utils.eval_utils', fake_eval_utils)
    monkeypatch.setattr(runner, 'loader', lambda *args, **kwargs: [batch])
    seen = []

    def recording_transfer(value, device, non_blocking=False):
        seen.append(non_blocking)
        return value

    monkeypatch.setattr(runner, 'to_device', recording_transfer)
    report = runner.evaluate(Model(), EvalDataset(), cfg(), torch.device('cpu'),
                             torch.Generator().manual_seed(303), settings=settings)
    assert seen == [True]
    assert report['AP']['AP@0.5'] == 0.0


def test_eight_worker_dataloader_smoke(monkeypatch):
    monkeypatch.setenv('NEGO_TRAIN_NUM_WORKERS', '8')
    monkeypatch.setenv('NEGO_PREFETCH_FACTOR', '2')
    monkeypatch.setenv('NEGO_PIN_MEMORY', '1')
    monkeypatch.setenv('NEGO_PERSISTENT_WORKERS', '0')
    settings = runner.dataloader_settings(cfg())
    batches = list(runner.loader(TinyDataset(), cfg(), True,
                                 torch.Generator().manual_seed(303), settings))
    assert sum(len(batch) for batch in batches) == len(TinyDataset())
    assert all(len(batch) == cfg()['train_params']['batch_size']
               for batch in batches[:-1])
