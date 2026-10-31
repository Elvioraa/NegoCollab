import random

import numpy as np
import pytest
import torch

from opencood.tools.nego_protocol import config_root, read_yaml, seed_all, seed_worker, validate_config
from opencood.tools.nego_state import rng_state, restore_rng
from opencood.tools.run_nego_doma import require_vectorized_supervision


def draw(generator):
    return (random.random(), float(np.random.random()), torch.rand(4),
            torch.randperm(9, generator=generator))


def test_rng_checkpoint_restores_python_numpy_torch_and_sampler():
    seed_all(47)
    generator = torch.Generator().manual_seed(47)
    state = rng_state(generator)
    expected = draw(generator)
    draw(generator)
    restore_rng(state, generator)
    actual = draw(generator)
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])


def test_worker_seed_uses_torch_initial_seed(monkeypatch):
    monkeypatch.setattr(torch, 'initial_seed', lambda: 123456789012)
    seed_worker(0)
    expected = (random.random(), np.random.random())
    seed_worker(7)
    assert (random.random(), np.random.random()) == expected


def test_sampler_order_and_epoch_rebuild_are_reproducible():
    def epoch(seed, number):
        seed_all(seed + number)
        generator = torch.Generator().manual_seed(seed + number)
        loader = torch.utils.data.DataLoader(list(range(11)), batch_size=4,
                                             shuffle=True, drop_last=False,
                                             worker_init_fn=seed_worker, generator=generator)
        return [batch.tolist() for batch in loader], draw(generator)
    first, rng_a = epoch(47, 2)
    resumed, rng_b = epoch(47, 2)
    assert first == resumed and list(map(len, first)) == [4, 4, 3]
    assert rng_a[:2] == rng_b[:2]
    assert torch.equal(rng_a[2], rng_b[2]) and torch.equal(rng_a[3], rng_b[3])


def test_top_level_seed_is_the_runtime_source():
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m1/config.yaml')
    cfg['seed'] = 47
    validate_config(cfg)
    cfg['seed'] = -1
    with pytest.raises(ValueError, match='seed'):
        validate_config(cfg)


def labels(positive=0, negative=0):
    return {'pos_equal_one': torch.full((2, 4, 4, 2), positive),
            'neg_equal_one': torch.full((2, 4, 4, 2), negative)}


def supervision_config():
    cfg = read_yaml(config_root('v2xset') / 'pretrain/m1/config.yaml')
    cfg['loss']['args']['pyramid']['relative_downsample'] = [1]
    cfg['loss']['args']['pyramid']['weight'] = [1]
    return cfg


def test_empty_gt_background_retains_supervision():
    cfg = supervision_config()
    batch = {'ego': {'label_dict': labels(negative=1), 'label_dict_single': labels(negative=1)}}
    require_vectorized_supervision(batch, cfg)


def test_disabled_loss_and_ignored_labels_are_not_supervision():
    cfg = supervision_config()
    cfg['loss']['args']['cls']['weight'] = 0
    cfg['train_params']['single_weight'] = 0
    batch = {'ego': {'label_dict': labels(negative=1), 'label_dict_single': labels(negative=1)}}
    with pytest.raises(ValueError, match='no enabled supervision'):
        require_vectorized_supervision(batch, cfg)
    cfg = supervision_config()
    batch['ego']['label_dict'] = labels()
    batch['ego']['label_dict_single'] = labels()
    with pytest.raises(ValueError, match='no enabled supervision'):
        require_vectorized_supervision(batch, cfg)


def test_single_occupancy_supervision_is_checked_independently():
    cfg = supervision_config()
    batch = {'ego': {'label_dict': labels(), 'label_dict_single': labels(negative=1)}}
    require_vectorized_supervision(batch, cfg)
    cfg['loss']['args']['pyramid']['weight'] = [0]
    with pytest.raises(ValueError, match='no enabled supervision'):
        require_vectorized_supervision(batch, cfg)
