import importlib.util
import ast
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from opencood.tools.nego_protocol import config_root, read_yaml, validate_runtime_batch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pretrain_collator', ROOT / 'opencood/data_utils/datasets/nego_m1_batch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('modality', ['m1', 'm2', 'm3', 'm4'])
def test_pretrain_runtime_gates(modality):
    cfg = read_yaml(config_root('v2xset') / 'pretrain' / modality / 'config.yaml')
    cfg['train_params']['batch_size'] = 7
    with pytest.raises(ValueError):
        validate_runtime_batch(cfg)
    cfg['nego'] = {'multi_batch': {'enabled': True, 'mode': 'vectorized'}}
    validate_runtime_batch(cfg)
    cfg['nego_protocol']['stage'] = 'alliance/nego'
    with pytest.raises(ValueError, match='pretraining'):
        validate_runtime_batch(cfg)


def sample(modality, count, index):
    return {'ego': {'cav_num': count, 'agent_modality_list': [modality]*count,
        'cav_id_list': [str(index)+'-'+str(i) for i in range(count)],
        'object_ids': [str(index)], 'sample_idx': index}}


def native_collate(modality, samples):
    counts = [x['ego']['cav_num'] for x in samples]
    total = sum(counts)
    inputs = {'voxel_coords': torch.tensor([[0,0,0,0]])}
    if modality in ('m2', 'm4'):
        inputs = {key: torch.arange(total).view(total,1) for key in
                  ('imgs','rots','trans','intrins','post_rots','post_trans')}
    return {'ego': {'record_len': torch.tensor(counts),
        'record_len_modality': {modality: torch.tensor(counts)},
        'agent_modality_list': [modality]*total,
        'pairwise_t_matrix': torch.zeros(len(counts),4,4,4,4),
        'object_bbx_center': torch.zeros(len(counts),2,7),
        'object_bbx_mask': torch.zeros(len(counts),2),
        'label_dict': {'pos_equal_one': torch.zeros(len(counts),1)},
        'label_dict_single': {'pos_equal_one': torch.zeros(total,1)},
        'inputs_'+modality: inputs}}


@pytest.mark.parametrize('modality', ['m1', 'm2', 'm3', 'm4'])
def test_boundaries_and_calibrations(modality):
    samples = [sample(modality,2,0), sample(modality,1,1), sample(modality,3,2)]
    out = module.collate_pretrain_vectorized(lambda b: native_collate(modality,b), modality, samples)['ego']
    meta = out['nego_batch_meta']
    assert meta['record_len'] == [2,1,3] and meta['cav_offsets'] == [0,2,3,6]
    assert meta['cav_scene_index'] == [0,0,1,2,2,2]
    assert meta['cav_ids'][1] == ['1-0'] and meta['sample_indices'] == [0,1,2]
    if modality in ('m2','m4'):
        assert torch.equal(out['inputs_'+modality]['imgs'], out['inputs_'+modality]['rots'])


def test_camera_calibration_mismatch_raises():
    samples = [sample('m2',2,0)]
    def malformed(items):
        data = native_collate('m2',items)
        data['ego']['inputs_m2']['rots'] = torch.zeros(1,1)
        return data
    with pytest.raises(ValueError, match='Camera'):
        module.collate_pretrain_vectorized(malformed, 'm2', samples)


def test_second_uses_explicit_count_for_empty_trailing_cavs():
    tree = ast.parse((ROOT / 'opencood/models/heter_encoders.py').read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'SECOND')
    forward = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'forward')
    scope = {}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), '<actual-second-forward>', 'exec'), scope)
    observed = []
    def sparse(data):
        observed.append(data['batch_size'])
        return data
    fake = SimpleNamespace(vfe=lambda x: x, spconv_block=sparse,
                           map_to_bev=lambda x: dict(x, spatial_features=torch.zeros(x['batch_size'], 1)))
    data = {'record_len': torch.tensor([2, 3]), 'nego_vectorized_pretrain': True,
            'inputs_m3': {'voxel_features': torch.ones(2, 1),
                          'voxel_coords': torch.tensor([[0, 0, 0, 0], [2, 0, 0, 0]]),
                          'voxel_num_points': torch.ones(2)}}
    result = scope['forward'](fake, data, 'm3')
    assert observed == [5] and result.shape[0] == 5
    data['inputs_m3']['voxel_coords'][1, 0] = 5
    with pytest.raises(ValueError, match='out of bounds'):
        scope['forward'](fake, data, 'm3')


@pytest.mark.parametrize('modality,size', [('m1', 4), ('m2', 4), ('m3', 8), ('m4', 4)])
def test_one_batched_forward_backward_and_step_with_synthetic_model(modality, size):
    from opencood.tools.run_nego_doma import loss_for
    class SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.calls = 0
        def forward(self, ego):
            self.calls += 1
            assert len(ego['record_len']) == size
            return {'prediction': self.weight * ego['values']}
    def criterion(output, target, suffix=''):
        return (output['prediction'] - target['y']).square().mean()
    model = SyntheticModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {'ego': {'record_len': torch.ones(size, dtype=torch.long),
                     'values': torch.arange(size, dtype=torch.float32),
                     'label_dict': {'y': torch.ones(size)}}}
    cfg = {'nego_protocol': {'phase': 'pretrain', 'stage': 'pretrain/' + modality},
           'model': {'args': {'supervise_single': False}},
           'nego': {'multi_batch': {'enabled': True}}}
    loss = loss_for(model, criterion, batch, cfg, 0)
    loss.backward()
    assert model.calls == 1 and torch.isfinite(loss) and torch.isfinite(model.weight.grad)
    before = model.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.weight.detach())


def test_unknown_checkpoint_code_change_is_not_assumed_compatible(tmp_path, monkeypatch):
    from opencood.tools import nego_state
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m1/config.yaml')
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(nego_state, 'code_hash', lambda: 'old-native-code')
    payload = nego_state.pack(model, cfg, {}, {}, {}, 1)
    path = tmp_path / 'legacy.pth'
    nego_state.atomic_save(path, payload)
    monkeypatch.setattr(nego_state, 'code_hash', lambda: 'new-code-with-default-native-path')
    protocol_sha = nego_state.file_sha(nego_state.ROOT / cfg['nego_protocol']['protocol_lock'])
    with pytest.raises(ValueError, match='implementation hash differs'):
        nego_state.load_stage(path, cfg=cfg, protocol_sha=protocol_sha,
                              mode='pretrained', model=model, target_cfg=cfg)
