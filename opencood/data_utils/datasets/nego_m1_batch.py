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
