"""CPU-only boundary checks; real Dataset/CUDA validation remains separate."""
import importlib.util
import ast
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from opencood.tools.nego_protocol import config_root, read_yaml, validate_runtime_batch


ROOT = Path(__file__).resolve().parents[1]


def load_source(relative):
    spec = importlib.util.spec_from_file_location('m1_test_module', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collate_m1_vectorized = load_source('opencood/data_utils/datasets/nego_m1_batch.py').collate_m1_vectorized
PointPillarScatter = load_source('opencood/models/sub_modules/point_pillar_scatter.py').PointPillarScatter


@pytest.mark.parametrize('size', [1, 2, 4, 8, 9])
def test_vectorized_opt_in_only_for_m1(size):
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m1/config.yaml')
    cfg['train_params']['batch_size'] = size
    cfg['nego'] = {'multi_batch': {'enabled': True, 'mode': 'vectorized'}}
    validate_runtime_batch(cfg)
    cfg['nego_protocol']['stage'] = 'm3/ft'
    with pytest.raises(ValueError, match='restricted'):
        validate_runtime_batch(cfg)


def test_scatter_preserves_empty_last_cav_and_legacy_output():
    scatter = PointPillarScatter({'num_features': 1, 'grid_size': [2, 2, 1]})
    source = {'pillar_features': torch.tensor([[3.], [5.]]),
              'voxel_coords': torch.tensor([[0, 0, 0, 0], [1, 0, 1, 1]])}
    legacy = scatter(dict(source))['spatial_features']
    batched = scatter(dict(source, nego_expected_cavs=3))['spatial_features']
    assert legacy.shape == (2, 1, 2, 2)
    assert batched.shape == (3, 1, 2, 2)
    torch.testing.assert_close(batched[:2], legacy)
    assert torch.count_nonzero(batched[2]) == 0
    with pytest.raises(ValueError, match='voxel index'):
        scatter(dict(source, nego_expected_cavs=1))


def test_scatter_batch_one_output_gradient_and_update_match():
    scatter = PointPillarScatter({'num_features': 1, 'grid_size': [2, 2, 1]})
    results = []
    for vectorized in (False, True):
        weight = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([weight], lr=0.05)
        batch = {'pillar_features': weight * torch.tensor([[1.], [2.]]),
                 'voxel_coords': torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]])}
        if vectorized:
            batch['nego_expected_cavs'] = 1
        out = scatter(batch)['spatial_features']
        loss = out.square().sum()
        loss.backward()
        results.append((out.detach().clone(), loss.detach().clone(), weight.grad.clone()))
        optimizer.step()
        results[-1] += (weight.detach().clone(),)
    for native, vectorized in zip(*results):
        torch.testing.assert_close(native, vectorized)


def fake_native_collator(samples):
    counts = [s['ego']['cav_num'] for s in samples]
    n = sum(counts)
    return {'ego': {'record_len': torch.tensor(counts),
        'record_len_modality': {'m1': torch.tensor(counts)},
        'agent_modality_list': [m for s in samples for m in s['ego']['agent_modality_list']],
        'pairwise_t_matrix': torch.zeros(len(samples), 4, 4, 4, 4),
        'object_bbx_center': torch.zeros(len(samples), 3, 7),
        'object_bbx_mask': torch.zeros(len(samples), 3),
        'label_dict': {'pos_equal_one': torch.zeros(len(samples), 2, 2, 2)},
        'label_dict_single': {'pos_equal_one': torch.zeros(n, 2, 2, 2)},
        'inputs_m1': {'voxel_coords': torch.tensor([[0, 0, 0, 0], [n - 1, 0, 0, 0]])}}}


def sample(n, ident):
    return {'ego': {'cav_num': n, 'agent_modality_list': ['m1'] * n, 'object_ids': [ident]}}


@pytest.mark.parametrize('counts', [[1], [2, 3], [1, 2, 3, 1], [1, 2, 1, 3, 1, 1, 2, 1]])
def test_collate_preserves_scene_boundaries_and_empty_targets(counts):
    samples = [sample(n, str(i)) for i, n in enumerate(counts)]
    out = collate_m1_vectorized(fake_native_collator, samples)['ego']
    assert out['record_len'].tolist() == counts
    assert out['object_ids_by_sample'] == [[str(i)] for i in range(len(counts))]
    assert out['nego_m1_vectorized'] is True
    assert out['object_bbx_mask'].sum() == 0


def test_rejects_foreign_modality_and_invalid_cav():
    bad = sample(2, 'x')
    bad['ego']['agent_modality_list'][1] = 'm2'
    with pytest.raises(ValueError, match='homogeneous'):
        collate_m1_vectorized(fake_native_collator, [bad])
    with pytest.raises(ValueError, match='CAV count'):
        collate_m1_vectorized(fake_native_collator, [sample(0, 'x')])


@pytest.mark.parametrize('counts', [[1], [2, 1], [1, 2, 3, 1], [1, 2, 1, 3, 1, 1, 2, 1]])
def test_native_pyramid_fusion_keeps_scenes_separate(counts):
    from opencood.models.fuse_modules.pyramid_fuse import weighted_fuse
    n = sum(counts)
    features = torch.arange(1, n + 1, dtype=torch.float32).view(n, 1, 1, 1).requires_grad_()
    scores = torch.ones_like(features)
    matrices = torch.zeros(len(counts), max(counts), max(counts), 2, 3)
    output = weighted_fuse(features, scores, torch.tensor(counts), matrices, False, warp=False)
    expected, start = [], 0
    for count in counts:
        expected.append(features[start:start + count].mean())
        start += count
    torch.testing.assert_close(output.flatten(), torch.stack(expected))
    output.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert (features.grad > 0).all()


def test_native_fusion_reorders_without_cross_scene_pollution():
    from opencood.models.fuse_modules.pyramid_fuse import weighted_fuse
    matrices = torch.zeros(2, 2, 2, 2, 3)
    counts = torch.tensor([1, 2])
    first = torch.tensor([1., 4., 6.]).view(3, 1, 1, 1)
    scores = torch.ones_like(first)
    baseline = weighted_fuse(first, scores.clone(), counts, matrices, False, warp=False)
    changed = first.clone()
    changed[1:] += 1000
    output = weighted_fuse(changed, scores.clone(), counts, matrices, False, warp=False)
    torch.testing.assert_close(output[0], baseline[0])
    reordered = first[[1, 2, 0]]
    permuted = weighted_fuse(reordered, scores.clone(), torch.tensor([2, 1]), matrices, False, warp=False)
    torch.testing.assert_close(permuted, baseline[[1, 0]])


def test_training_entry_calls_model_once_for_a_batch():
    from opencood.tools.run_nego_doma import loss_for
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.calls = 0
        def forward(self, ego):
            self.calls += 1
            return {'prediction': self.weight * ego['input']}
    def criterion(out, target, suffix=''):
        return ((out['prediction'] - target['y']) ** 2).mean()
    model = Model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {'ego': {'input': torch.arange(4, dtype=torch.float32),
                     'label_dict': {'y': torch.ones(4)}}}
    cfg = {'nego_protocol': {'phase': 'pretrain'}, 'model': {'args': {'supervise_single': False}}}
    opt.zero_grad()
    loss = loss_for(model, criterion, batch, cfg, 0)
    loss.backward()
    assert model.calls == 1 and torch.isfinite(model.weight.grad)
    before = model.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, model.weight.detach())


def test_native_detection_loss_reduces_over_scenes():
    # Execute the unchanged native class and its two pure torch helpers directly
    # from source; the full module import needs server-only Dataset dependencies.
    source = (ROOT / 'opencood/loss/point_pillar_loss.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = {'PointPillarLoss', 'sigmoid_focal_loss', 'weighted_smooth_l1_loss'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    scope = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<native-pillar-loss>', 'exec'), scope)
    criterion = scope['PointPillarLoss']({'pos_cls_weight': 1.0,
        'cls': {'weight': 1.0, 'alpha': 0.25, 'gamma': 2.0},
        'reg': {'weight': 1.0, 'sigma': 3.0}})
    targets = {'pos_equal_one': torch.tensor([[[[1., 0.]]], [[[0., 0.]]]]),
               'neg_equal_one': torch.tensor([[[[0., 1.]]], [[[1., 1.]]]]),
               'targets': torch.zeros(2, 1, 1, 14)}
    preds = {'cls_preds': torch.tensor([[[[0.2]], [[-0.1]]], [[[0.3]], [[0.4]]]], requires_grad=True),
             'reg_preds': torch.zeros(2, 14, 1, 1, requires_grad=True)}
    combined = criterion(preds, targets)
    individual = []
    for i in range(2):
        individual.append(criterion({k: v[i:i+1] for k, v in preds.items()},
                                    {k: v[i:i+1] for k, v in targets.items()}))
    torch.testing.assert_close(combined, torch.stack(individual).mean())
    combined.backward()
    assert torch.isfinite(preds['cls_preds'].grad).all()
    assert torch.isfinite(preds['reg_preds'].grad).all()


def test_native_pyramid_loss_reduces_over_cavs():
    pillar = ast.parse((ROOT / 'opencood/loss/point_pillar_loss.py').read_text(encoding='utf-8'))
    pyramid = ast.parse((ROOT / 'opencood/loss/point_pillar_pyramid_loss.py').read_text(encoding='utf-8'))
    focal = next(n for n in pillar.body if isinstance(n, ast.FunctionDef) and n.name == 'sigmoid_focal_loss')
    cls = next(n for n in pyramid.body if isinstance(n, ast.ClassDef) and n.name == 'PointPillarPyramidLoss')
    occ = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'calc_occ_loss')
    scope = {'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[focal, occ], type_ignores=[]), '<native-pyramid-loss>', 'exec'), scope)
    options = SimpleNamespace(relative_downsample=[1], pyramid_weight=[1.0],
                              pos_cls_weight=1.0, cls={'alpha': 0.25, 'gamma': 2.0})
    logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True).view(3, 1, 1, 1)
    pos = torch.tensor([[[[1., 0.]]], [[[0., 0.]]], [[[1., 0.]]]])
    neg = 1 - pos
    fn = scope['calc_occ_loss']
    combined = fn(options, [logits], pos, neg, 3)
    first = fn(options, [logits[:1]], pos[:1], neg[:1], 1)
    second = fn(options, [logits[1:]], pos[1:], neg[1:], 2)
    torch.testing.assert_close(combined, (first + 2 * second) / 3)
    grad = torch.autograd.grad(combined, logits)[0]
    assert torch.isfinite(grad).all()
