"""CPU-only tool checks; these do not validate the real Dataset or GPU models."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from opencood.tools.nego_protocol import config_root, read_yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pretrain_validation', ROOT / 'scripts/validate_nego_pretrain_vectorized.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def ego_for(member):
    counts = torch.tensor([2, 1])
    if member in ('m1', 'm3'):
        inputs = {'voxel_coords': torch.tensor([[0, 0, 0, 0], [1, 0, 1, 0]])}
    else:
        inputs = {'imgs': torch.zeros(3, 4, 3, 8, 8)}
        inputs.update({name: torch.zeros(3, 4, 3, 3) for name in ('rots', 'intrins', 'post_rots')})
        inputs.update({name: torch.zeros(3, 4, 3) for name in ('trans', 'post_trans')})
    return {'record_len': counts, 'record_len_modality': {member: counts},
            'agent_modality_list': [member] * 3, 'inputs_' + member: inputs}


@pytest.mark.parametrize('member', ['m1', 'm2', 'm3', 'm4'])
def test_sensor_statistics_selects_actual_member(member):
    result = tool.sensor_statistics(ego_for(member), 'pretrain/' + member, 2)
    assert result['record_len'] == [2, 1] and result['cav_count'] == 3
    if member in ('m1', 'm3'):
        assert result['voxel_count'] == 2 and result['voxel_coords_shape'] == [2, 4]
    else:
        assert result['camera_tensor_shapes']['imgs'] == [3, 4, 3, 8, 8]
        assert set(result['camera_tensor_shapes']) == {'imgs', 'rots', 'trans', 'intrins', 'post_rots', 'post_trans'}


def test_sensor_statistics_rejects_bad_boundaries():
    ego = ego_for('m3')
    with pytest.raises(ValueError, match='batch size'):
        tool.sensor_statistics(ego, 'pretrain/m3', 3)
    ego['inputs_m3']['voxel_coords'][1, 0] = 3
    with pytest.raises(ValueError, match='Voxel CAV'):
        tool.sensor_statistics(ego, 'pretrain/m3', 2)


def test_config_must_explicitly_match_batch_and_stage():
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m2/config.yaml')
    cfg['train_params']['batch_size'] = 4
    cfg['nego'] = {'multi_batch': {'enabled': True, 'mode': 'vectorized'}}
    tool.check_config(cfg, 'pretrain/m2', 4)
    with pytest.raises(ValueError, match='batch size'):
        tool.check_config(cfg, 'pretrain/m2', 8)
    with pytest.raises(ValueError, match='stage'):
        tool.check_config(cfg, 'pretrain/m4', 4)
    cfg['seed'] = 10
    with pytest.raises(ValueError, match='seed 303'):
        tool.check_config(cfg, 'pretrain/m2', 4)


def test_main_configuration_error_creates_no_report_or_checkpoint(tmp_path, monkeypatch):
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m2/config.yaml')
    monkeypatch.setattr(tool, 'read_yaml', lambda path: cfg)
    report_dir = tmp_path / 'reports'
    monkeypatch.setattr(sys, 'argv', ['validate', '--config', 'unused.yaml', '--stage', 'pretrain/m2',
                                    '--batch-size', '4', '--report-dir', str(report_dir)])
    with pytest.raises(ValueError, match='batch size'):
        tool.main()
    assert not report_dir.exists()


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, ego):
        prediction = self.weight * ego['x']
        return {key: prediction for key in ('cls_preds', 'reg_preds', 'dir_preds')}


def criterion(output, target, suffix=''):
    return (output['cls_preds'] - target['y']).square().mean()


def update_inputs():
    cfg = {'optimizer': {'core_method': 'SGD', 'lr': 0.1},
           'nego_protocol': {'phase': 'pretrain'}, 'model': {'args': {'supervise_single': False}},
           'nego': {'multi_batch': {'enabled': True}}}
    batch = {'ego': {'record_len': torch.tensor([1, 1]), 'x': torch.ones(2, 1, 1, 1),
                     'label_dict': {'y': torch.zeros(2, 1, 1, 1)}}}
    return cfg, batch


@pytest.fixture
def synthetic_cuda_metrics(monkeypatch):
    for name in ('reset_peak_memory_stats', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)


def test_update_uses_one_real_loss_entrypoint_and_updates(synthetic_cuda_metrics):
    cfg, batch = update_inputs()
    model = SyntheticModel()
    report = tool.run_update(model, criterion, cfg, batch, 'pretrain/m1')
    assert report['gradient_tensor_count'] == 1
    assert report['updated_parameter_tensors'] == 1
    assert report['output_shapes']['cls_preds'][0] == 2


def test_failure_does_not_create_checkpoint(tmp_path, synthetic_cuda_metrics):
    cfg, batch = update_inputs()
    batch['ego']['x'].fill_(float('nan'))
    with pytest.raises(ValueError, match='Invalid detection output'):
        tool.run_update(SyntheticModel(), criterion, cfg, batch, 'pretrain/m1')
    assert not list(tmp_path.rglob('*.pth'))
    assert not list(tmp_path.rglob('*.pt'))
