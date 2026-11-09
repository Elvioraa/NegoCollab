"""Opt-in m1 collation checks; the native collator remains authoritative."""
import torch


def collate_m1_vectorized(native_collate, samples):
    if not samples or any(sample is None for sample in samples):
        raise ValueError('Vectorized m1 received an empty sample')
    counts = []
    object_ids = []
    for sample in samples:
        ego = sample['ego']
        count = ego['cav_num']
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError('Invalid CAV count')
        if ego['agent_modality_list'] != ['m1'] * count:
            raise ValueError('Vectorized m1 expects homogeneous m1 members')
        counts.append(int(count))
        object_ids.append(ego['object_ids'])
    batch = native_collate(samples)
    ego = batch['ego']
    record = ego['record_len']
    if record.tolist() != counts or ego['agent_modality_list'] != ['m1'] * sum(counts):
        raise ValueError('Vectorized CAV boundaries differ from native samples')
    if ego['record_len_modality']['m1'].tolist() != counts:
        raise ValueError('Vectorized per-modality CAV counts differ from samples')
    if ego['pairwise_t_matrix'].shape[0] != len(samples):
        raise ValueError('Vectorized transformation batch mismatch')
    if ego['object_bbx_center'].shape[0] != len(samples) or ego['object_bbx_mask'].shape[0] != len(samples):
        raise ValueError('Vectorized labels batch mismatch')
    if ego['label_dict']['pos_equal_one'].shape[0] != len(samples):
        raise ValueError('Vectorized detection labels batch mismatch')
    if ego['label_dict_single']['pos_equal_one'].shape[0] != sum(counts):
        raise ValueError('Vectorized CAV labels batch mismatch')
    coords = ego['inputs_m1']['voxel_coords']
    if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 4 or coords.shape[0] == 0:
        raise ValueError('Vectorized m1 needs at least one voxel in the batch')
    if int(coords[:, 0].min()) < 0 or int(coords[:, 0].max()) >= sum(counts):
        raise ValueError('Vectorized voxel CAV index outside record_len')
    ego['object_ids_by_sample'] = object_ids
    ego['nego_m1_vectorized'] = True
    return batch


def collate_pretrain_vectorized(native_collate, modality, samples):
    """Collate homogeneous pretraining scenes once and retain their identities."""
    if modality not in ('m1', 'm2', 'm3', 'm4') or not samples or any(x is None for x in samples):
        raise ValueError('Invalid pretraining samples or modality')
    counts, ids, objects, scenes = [], [], [], []
    for sample in samples:
        ego = sample['ego']
        count = ego['cav_num']
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError('Invalid CAV count')
        if ego['agent_modality_list'] != [modality] * count or len(ego['cav_id_list']) != count:
            raise ValueError('CAV identity or homogeneous modality mismatch')
        counts.append(int(count))
        ids.append(list(ego['cav_id_list']))
        objects.append(list(ego['object_ids']))
        scenes.append(ego['sample_idx'])
    batch = collate_m1_vectorized(native_collate, samples) if modality == 'm1' else native_collate(samples)
    ego = batch['ego']
    if ego['record_len'].tolist() != counts or ego['agent_modality_list'] != [modality] * sum(counts):
        raise ValueError('CAV boundaries do not match samples')
    if ego['record_len_modality'][modality].tolist() != counts:
        raise ValueError('Per-modality counts do not match samples')
    if ego['pairwise_t_matrix'].shape[0] != len(samples) or ego['object_bbx_mask'].shape[0] != len(samples):
        raise ValueError('Scene transforms or labels do not match samples')
    if ego['label_dict']['pos_equal_one'].shape[0] != len(samples) or ego['label_dict_single']['pos_equal_one'].shape[0] != sum(counts):
        raise ValueError('Scene/CAV supervision does not match samples')
    inputs = ego['inputs_' + modality]
    if modality in ('m1', 'm3'):
        coords = inputs['voxel_coords']
        if coords.ndim != 2 or coords.shape[1] != 4 or not coords.numel():
            raise ValueError('Entire pretraining batch has no valid voxels')
        if coords[:, 0].min() < 0 or coords[:, 0].max() >= sum(counts):
            raise ValueError('Voxel CAV index outside batch')
    else:
        for key in ('imgs', 'rots', 'trans', 'intrins', 'post_rots', 'post_trans'):
            if key not in inputs or inputs[key].shape[0] != sum(counts):
                raise ValueError('Camera input/calibration CAV mismatch: ' + key)
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    ego['nego_batch_meta'] = {'batch_size': len(samples), 'record_len': counts,
        'record_len_modality': {modality: list(counts)}, 'cav_offsets': offsets,
        'cav_scene_index': [i for i, count in enumerate(counts) for _ in range(count)],
        'ego_modalities': [modality] * len(samples), 'cav_modalities': [modality] * sum(counts),
        'cav_ids': ids, 'object_ids': objects, 'sample_indices': scenes}
    ego['nego_vectorized_pretrain'] = True
    return batch
