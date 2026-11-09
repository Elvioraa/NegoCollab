import copy

import pytest
import torch

from opencood.tools import nego_state as state
from opencood.tools.nego_protocol import config_root, read_yaml, file_sha, ROOT, digest


class HomoCollab(torch.nn.Linear):
    pass


@pytest.fixture
def example(tmp_path, monkeypatch):
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m1/config.yaml')
    model = HomoCollab(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    generator = torch.Generator().manual_seed(3)
    monkeypatch.setattr(state, 'code_hash', lambda: 'reviewed-source')
    monkeypatch.setattr(state, 'model_implementation_signature', lambda: 'reviewed-model')
    payload = state.pack(model, cfg, {}, {}, {}, 2, optimizer, scheduler, generator, 0.5, 6)
    path = tmp_path / 'checkpoint.pth'
    state.atomic_save(path, payload)
    protocol_sha = file_sha(ROOT / cfg['nego_protocol']['protocol_lock'])
    return cfg, model, payload, path, protocol_sha


def test_resume_same_version_and_complete_state(example):
    cfg, model, _, path, protocol = example
    ckpt = state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='resume', model=model)
    assert ckpt['resume']['global_step'] == 6 and ckpt['meta']['completed_epoch'] == 2
    assert ckpt['meta']['model_config_sha'] == digest(cfg['model'])


def test_resume_different_code_is_rejected(example, monkeypatch):
    cfg, model, _, path, protocol = example
    monkeypatch.setattr(state, 'code_hash', lambda: 'different-code')
    with pytest.raises(ValueError, match='implementation hash'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='resume', model=model)


def test_reviewed_pretrained_loads_with_strict_state(example, monkeypatch):
    cfg, model, _, path, protocol = example
    monkeypatch.setattr(state, 'code_hash', lambda: 'new-code')
    monkeypatch.setattr(state, 'REVIEWED_PRETRAINED_COMPATIBILITY',
                        {'reviewed-source': {'source_signature': 'reviewed-model', 'target_signatures': {'reviewed-model'}, 'allowed_stages': {'pretrain/m1'}}})
    ckpt = state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)
    state.strict_load(model, ckpt['model'])


def test_pretrained_model_config_mismatch_is_rejected(example):
    cfg, model, _, path, protocol = example
    target = copy.deepcopy(cfg)
    target['model']['args']['supervise_single'] = not target['model']['args']['supervise_single']
    with pytest.raises(ValueError, match='target model configuration'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=target)


def test_pretrained_allows_target_training_hyperparameter_change(example):
    cfg, model, _, path, protocol = example
    target = copy.deepcopy(cfg)
    target['optimizer']['lr'] *= 0.5
    target['train_params']['batch_size'] = 4
    ckpt = state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=target)
    assert ckpt['meta']['stage'] == 'pretrain/m1'


@pytest.mark.parametrize('mutation', ['missing', 'shape', 'dtype'])
def test_pretrained_rejects_schema_even_if_self_consistent(example, mutation):
    cfg, model, payload, path, protocol = example
    if mutation == 'missing':
        del payload['model']['bias']
    elif mutation == 'shape':
        payload['model']['weight'] = torch.zeros(3, 2)
    else:
        payload['model']['weight'] = payload['model']['weight'].double()
    payload['meta']['state_schema'] = digest({k: [list(v.shape), str(v.dtype)] for k,v in payload['model'].items()})
    state.atomic_save(path, payload)
    with pytest.raises(ValueError, match='State key mismatch|State shape/dtype mismatch'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)


def test_unknown_legacy_code_rejected(example, monkeypatch):
    cfg, model, payload, path, protocol = example
    for key in ('source_revision','model_identity','model_config_sha','experiment_config_sha',
                'model_implementation_signature','global_step'):
        payload['meta'].pop(key)
    state.atomic_save(path, payload)
    monkeypatch.setattr(state, 'code_hash', lambda: 'new-code')
    with pytest.raises(ValueError, match='implementation hash'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)


def test_reviewed_legacy_checkpoint_requires_known_source_and_current_model(example, monkeypatch):
    cfg, model, payload, path, protocol = example
    for key in ('source_revision','model_identity','model_config_sha','experiment_config_sha',
                'model_implementation_signature','global_step'):
        payload['meta'].pop(key)
    payload['meta']['code_sha'] = '33ca2a8b10d5bd55847786734c0fbcf5d302e6679aec26513f5b102a6f49e7c3'
    state.atomic_save(path, payload)
    monkeypatch.setattr(state, 'code_hash', lambda: 'new-code')
    monkeypatch.setattr(state, 'model_implementation_signature', lambda: '4d5457fa6bce5c7d2aff099131a739ce7a4e22e77e5037b843ec613d5c967509')
    assert state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)['meta']['stage'] == 'pretrain/m1'
    monkeypatch.setattr(state, 'model_implementation_signature', lambda: 'not-reviewed')
    with pytest.raises(ValueError, match='implementation hash'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)


def test_provenance_and_checksum_remain_mandatory(example):
    cfg, model, _, path, protocol = example
    with pytest.raises(ValueError, match='protocol_sha mismatch'):
        state.load_stage(path, cfg=cfg, protocol_sha='wrong', mode='pretrained', model=model, target_cfg=cfg)
    path.write_bytes(path.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)


def test_resume_rejects_missing_step(example):
    cfg, model, payload, path, protocol = example
    del payload['resume']['global_step']
    state.atomic_save(path, payload)
    with pytest.raises(ValueError, match='complete resume'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='resume', model=model)


def test_reviewed_source_declaration_matches_current_model_files():
    source = '33ca2a8b10d5bd55847786734c0fbcf5d302e6679aec26513f5b102a6f49e7c3'
    assert state.model_implementation_signature() in state.REVIEWED_PRETRAINED_COMPATIBILITY[source]['target_signatures']


def test_legacy_e0_does_not_approve_unreviewed_m2(tmp_path, monkeypatch):
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m2/config.yaml')
    model = HomoCollab(2, 2)
    monkeypatch.setattr(state, 'code_hash', lambda: '33ca2a8b10d5bd55847786734c0fbcf5d302e6679aec26513f5b102a6f49e7c3')
    payload = state.pack(model, cfg, {}, {}, {}, 1)
    for key in ('source_revision','model_identity','model_config_sha','experiment_config_sha',
                'model_implementation_signature','global_step'):
        payload['meta'].pop(key)
    path = tmp_path / 'legacy_m2.pth'
    state.atomic_save(path, payload)
    monkeypatch.setattr(state, 'code_hash', lambda: 'new-code')
    protocol = file_sha(ROOT / cfg['nego_protocol']['protocol_lock'])
    with pytest.raises(ValueError, match='implementation hash'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)


def test_resume_rejects_epoch_disagreement(example):
    cfg, model, payload, path, protocol = example
    payload['resume']['completed_epoch'] += 1
    state.atomic_save(path, payload)
    with pytest.raises(ValueError, match='epoch/global_step'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='resume', model=model)


def test_resume_rejects_incomplete_optimizer(example):
    cfg, model, payload, path, protocol = example
    payload['resume']['optimizer'] = {}
    state.atomic_save(path, payload)
    with pytest.raises(ValueError, match='optimizer/scheduler/RNG'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='resume', model=model)


def test_pretrained_rejects_forged_source_signature(example, monkeypatch):
    cfg, model, payload, path, protocol = example
    payload['meta']['model_implementation_signature'] = 'wrong-source-signature'
    state.atomic_save(path, payload)
    monkeypatch.setattr(state, 'code_hash', lambda: 'new-code')
    monkeypatch.setattr(state, 'REVIEWED_PRETRAINED_COMPATIBILITY',
                        {'reviewed-source': {'source_signature': 'reviewed-model', 'target_signatures': {'reviewed-model'}, 'allowed_stages': {'pretrain/m1'}}})
    with pytest.raises(ValueError, match='implementation hash'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=model, target_cfg=cfg)


def test_pretrained_rejects_wrong_target_model_type(example):
    cfg, _, _, path, protocol = example
    with pytest.raises(ValueError, match='model type'):
        state.load_stage(path, cfg=cfg, protocol_sha=protocol, mode='pretrained', model=torch.nn.Linear(2,2), target_cfg=cfg)


def test_initialize_loads_reviewed_parent_weights_and_preserves_provenance(tmp_path, monkeypatch):
    class HomoCollab(torch.nn.Module):
        def __init__(self, member):
            super().__init__()
            self.add_module('encoder_' + member, torch.nn.Linear(2, 2))
    class Alliance(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_m1 = torch.nn.Linear(2, 2)
            self.encoder_m2 = torch.nn.Linear(2, 2)
    monkeypatch.setattr(state, 'code_hash', lambda: 'reviewed-source')
    monkeypatch.setattr(state, 'model_implementation_signature', lambda: 'reviewed-model')
    parents = {}
    for member in ('m1', 'm2'):
        source = config_root('v2xset') / 'pretrain' / member / 'config.yaml'
        directory = tmp_path / 'pretrain' / member
        directory.mkdir(parents=True)
        (directory / 'config.yaml').write_bytes(source.read_bytes())
        cfg = read_yaml(source)
        parents['pretrain/' + member] = HomoCollab(member)
        state.atomic_save(directory / 'best.pth', state.pack(parents['pretrain/' + member], cfg, {}, {}, {}, 1))
    monkeypatch.setattr(state, 'code_hash', lambda: 'new-code')
    monkeypatch.setattr(state, 'REVIEWED_PRETRAINED_COMPATIBILITY',
                        {'reviewed-source': {'source_signature': 'reviewed-model', 'target_signatures': {'reviewed-model'}, 'allowed_stages': {'pretrain/m1','pretrain/m2'}}})
    cfg = read_yaml(config_root('v2xset') / 'alliance/nego/config.yaml')
    target = Alliance()
    _, provenance, _ = state.initialize(target, cfg, tmp_path, parent_models=parents)
    assert set(provenance) == set(parents)
    for member in ('m1', 'm2'):
        torch.testing.assert_close(getattr(target, 'encoder_' + member).weight,
                                   getattr(parents['pretrain/' + member], 'encoder_' + member).weight)
