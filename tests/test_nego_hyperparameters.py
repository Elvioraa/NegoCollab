import copy

import pytest

from opencood.tools.nego_protocol import config_root, read_yaml, validate_config, validate_runtime_batch


def sample():
    return read_yaml(config_root('v2xset') / 'pretrain/m1/config.yaml')


@pytest.mark.parametrize('size', [1, 2, 4, 8])
def test_valid_batch_sizes_are_configuration_legal(size):
    cfg = sample()
    cfg['train_params']['batch_size'] = size
    validate_config(cfg)
    if size > 1:
        with pytest.raises(ValueError, match='Runtime multi-batch'):
            validate_runtime_batch(cfg)
    else:
        validate_runtime_batch(cfg)


@pytest.mark.parametrize('size', [0, -1, 1.5, True, False])
def test_invalid_batch_sizes(size):
    cfg = sample()
    cfg['train_params']['batch_size'] = size
    with pytest.raises(ValueError, match='batch_size'):
        validate_config(cfg)


def test_training_values_are_not_locked():
    cfg = sample()
    cfg['train_params'].update(epoches=3, num_workers=2, eval_batch_size=2)
    cfg['optimizer'] = {'core_method': 'SGD', 'lr': 0.025, 'args': {'weight_decay': 0.005}}
    cfg['lr_scheduler'] = {'core_method': 'step', 'step_size': 2, 'gamma': 0.5}
    validate_config(cfg)
    with pytest.raises(ValueError, match='Runtime multi-batch'):
        validate_runtime_batch(cfg)


@pytest.mark.parametrize('field,value', [('lr', 0), ('lr', float('nan')), ('lr', True), ('weight_decay', -1)])
def test_invalid_numeric_values(field, value):
    cfg = sample()
    if field == 'lr':
        cfg['optimizer']['lr'] = value
    else:
        cfg['optimizer']['args'][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_protocol_checks_still_apply():
    cfg = sample()
    cfg['train_params']['batch_size'] = 4
    for change in ('parents', 'scene_exclusions'):
        broken = copy.deepcopy(cfg)
        broken['nego_protocol'][change] = ['not-approved']
        with pytest.raises(ValueError):
            validate_config(broken)


def test_split_and_evaluation_policy_are_locked():
    cfg = sample()
    cfg['root_dir'] = 'dataset/other/train'
    with pytest.raises(ValueError, match='split'):
        validate_config(cfg)
    cfg = read_yaml(config_root('v2xset') / 'final_infer/config.yaml')
    cfg['postprocess']['nms_thresh'] = 0.99
    with pytest.raises(ValueError, match='postprocess'):
        validate_config(cfg)


def test_matching_but_wrong_dataset_and_model_mapping_is_rejected():
    cfg = sample()
    cfg['heter']['mapping_dict']['m1'] = 'm2'
    cfg['model']['args']['mapping_dict']['m1'] = 'm2'
    with pytest.raises(ValueError, match='mapping'):
        validate_config(cfg)
