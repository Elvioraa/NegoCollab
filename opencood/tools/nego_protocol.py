"""DOMA protocol contracts. No perception/CUDA imports, no implicit experiments."""
import copy
import hashlib
import json
import random
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODALITIES = ('m1', 'm2', 'm3', 'm4')
STAGES = ('pretrain/m1', 'pretrain/m2', 'pretrain/m3', 'pretrain/m4',
          'alliance/nego', 'alliance/ft', 'm3/align', 'm3/ft', 'm4/align', 'm4/ft', 'final_infer')
BASE_SHA = '8cd12e68549ab23109a691230990004a38402fef'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def config_root(dataset):
    if dataset not in ('opv2v', 'v2xset'):
        raise ValueError(dataset)
    return ROOT / 'opencood/hypes_yaml' / dataset / 'doma_protocol_negocollab'


def parents(stage):
    if stage.startswith('pretrain/'):
        return []
    return {'alliance/nego': ['pretrain/m1', 'pretrain/m2'],
            'alliance/ft': ['alliance/nego'],
            'm3/align': ['alliance/nego', 'alliance/ft', 'pretrain/m3'],
            'm4/align': ['alliance/nego', 'alliance/ft', 'pretrain/m4'],
            'm3/ft': ['m3/align'], 'm4/ft': ['m4/align'],
            'final_infer': ['alliance/ft', 'm3/ft', 'm4/ft']}[stage]


def active_members(stage):
    if stage.startswith('pretrain/'):
        return [stage.split('/')[1]]
    if stage.startswith('alliance/'):
        return ['m1', 'm2']
    if stage == 'final_infer':
        return list(MODALITIES)
    return ['m1', 'm2', stage.split('/')[0]]


def lidar_path(original, dataset, modality, require=True):
    # Work on both slash conventions. Never rewrite V2XSET to a beam-specific file.
    original = str(original).replace('\\', '/')
    path = PurePosixPath(original)
    if dataset == 'opv2v' and modality == 'm3':
        parts = list(path.parts)
        if 'OPV2V' not in parts:
            raise ValueError('32-beam input must have an OPV2V path component: ' + original)
        parts[parts.index('OPV2V')] = 'OPV2V_Hetero'
        path = PurePosixPath(*parts).with_name(path.stem + '_32.pcd')
    elif dataset not in ('opv2v', 'v2xset'):
        raise ValueError(dataset)
    result = str(path)
    if require and not Path(result).is_file():
        raise FileNotFoundError(result)
    return result


def assert_prefix(modalities, requested):
    if requested not in (1, 2, 3, 4):
        raise ValueError('use_cav must be 1..4')
    realized = list(modalities[:requested])
    if not realized or realized != list(MODALITIES[:len(realized)]):
        raise ValueError('Non-contiguous DOMA prefix after communication filtering: ' + repr(realized))
    return realized


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    import torch
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


class ProtocolAdaptor:
    """Native assignment semantics with stable iteration and explicit beam policy."""
    def __init__(self, ego_modality, model_modality_list, modality_assignment,
                 lidar_channels_dict, mapping_dict, cav_preference, train, dataset, paired=False):
        self.ego_modality = ego_modality
        self.model_modality_list = sorted(model_modality_list)
        self.modality_assignment = modality_assignment
        self.mapping_dict = mapping_dict
        self.train = train
        self.dataset = dataset
        self.paired = paired
        self.preference = cav_preference or dict.fromkeys(self.model_modality_list, 1.0)
        if dataset == 'v2xset' and lidar_channels_dict:
            raise ValueError('V2XSet forbids beam rewriting')

    def reorder_cav_list(self, cav_list, scenario_name):
        if self.train:
            random.shuffle(cav_list)
            return cav_list
        assignment = self.modality_assignment[scenario_name]
        if set(assignment) != set(cav_list):
            raise ValueError('Assignment/data CAV mismatch: ' + scenario_name)
        # Ordered assignment is authoritative; no hidden ID sorting or renaming.
        ordered = list(assignment)
        if self.mapping_dict[assignment[ordered[0]]] not in self.ego_modality.split('&'):
            ego = next((c for c in ordered if self.mapping_dict[assignment[c]] in self.ego_modality.split('&')), None)
            if ego is None:
                raise ValueError('No legal ego in ' + scenario_name)
            ordered = [ego] + [c for c in ordered if c != ego]
        return ordered

    def reassign_cav_modality(self, modality_name, idx_in_cav_list):
        if not self.train:
            return self.mapping_dict[modality_name]
        if idx_in_cav_list == 0:
            return str(np.random.choice(self.ego_modality.split('&')))
        names = sorted(self.preference)
        return random.choices(names, weights=[self.preference[n] for n in names])[0]

    def unmatched_modality(self, modality):
        return modality not in self.model_modality_list

    def switch_lidar_channels(self, modality, path):
        return lidar_path(path, self.dataset, 'm1' if self.paired else modality, require=False)


def validate_config(cfg):
    p = cfg['nego_protocol']
    stage, dataset = p['stage'], p['dataset']
    if stage not in STAGES or dataset not in ('opv2v', 'v2xset'):
        raise ValueError('Unknown stage/dataset')
    members = active_members(stage)
    args = cfg['model']['args']
    phase = p['phase']
    expected_core = {'pretrain':'homo_collab','nego':'heter_nego_train',
                     'align':'heter_nego_train_align','ft':'heter_nego_inf','inf':'heter_nego_inf'}[phase]
    if cfg['model']['core_method']!=expected_core or args['stage']!=phase:
        raise ValueError('Wrong native model/stage')
    if args['mapping_dict']!=cfg['heter']['mapping_dict']:
        raise ValueError('Model/dataset mapping mismatch')
    if args['pub'].get('pc_grad',False):
        raise ValueError('Selected native templates use pc_grad=false; alternative PCGrad training is not supported by this entry')
    if phase in ('nego','align'):
        loss=cfg['loss']['args']
        if min(loss['unify_dis']['args']['ratio'],loss['unify_stru']['ratio'],loss['unify_pragma']['ratio'],loss['cycle_dis']['ratio'])<=0:
            raise ValueError('All three native alignment objectives and cycle reconstruction must remain active')
    if sorted(cfg['heter']['modality_setting']) != members or sorted(args['modality_setting']) != members:
        raise ValueError('Incorrect member set')
    for m in members:
        a = args[m]
        if not stage.startswith('pretrain/') and a['allied']!=(m not in p['new_members']):
            raise ValueError('Incorrect allied/frozen membership: '+m)
        expected = {'m1': ('lidar', 'point_pillar'), 'm2': ('camera', 'lift_splat_shoot'),
                    'm3': ('lidar', 'second'), 'm4': ('camera', 'lift_splat_shoot')}[m]
        if (a['sensor_type'], a['core_method']) != expected:
            raise ValueError('Wrong modality identity: ' + m)
        if m in ('m2', 'm4'):
            if a['encoder_args']['camera_encoder'] != {'m2': 'EfficientNet', 'm4': 'Resnet101'}[m]:
                raise ValueError('Wrong camera backbone: ' + m)
            data = cfg['heter']['modality_setting'][m]
            if data['data_aug_conf']['Ncams'] != 4 or data['data_aug_conf']['cams'] != ['camera0','camera1','camera2','camera3']:
                raise ValueError('Four cameras required')
            if dataset == 'v2xset' and a['encoder_args']['depth_supervision']:
                raise ValueError('V2XSet has no depth supervision')
        else:
            if a['encoder_args']['lidar_range'] != args['lidar_range']:
                raise ValueError('Encoder canvas mismatch')
        if not stage.startswith('pretrain/'):
            c = a['comm_args']
            if c['unify_parameters']['C_uni'] != 64 or a['local_dim'] != 64:
                raise ValueError('Incompatible common channels')
    beam = {'m3': 32} if dataset == 'opv2v' and 'm3' in members else {}
    if cfg['heter'].get('lidar_channels_dict', {}) != beam:
        raise ValueError('Wrong beam policy')
    if p['parents'] != parents(stage) or p['seed'] != cfg['seed']:
        raise ValueError('DAG/seed mismatch')
    if cfg['train_params']['batch_size'] != 1:
        # Native inference/FT uses agent_modality_list[0] for the whole batch.
        raise ValueError('This native FT/dataset contract is validated for batch_size=1 only')
    if cfg['fusion']['args']['proj_first']:
        raise ValueError('Expected unprojected intermediate fusion')
    if stage == 'final_infer':
        if cfg['heter']['mapping_dict'] != dict(zip(MODALITIES, MODALITIES)) or cfg['heter']['ego_modality'] != 'm1':
            raise ValueError('Final inference requires m1 ego and identity mapping')
    return True


def verify_lock(dataset):
    lock = read_yaml(config_root(dataset) / 'protocol_lock.yaml')
    for entry in list(lock['reference_configs'].values()) + list(lock['assignments'].values()):
        path = ROOT / entry.get('copy', entry.get('path', ''))
        if file_sha(path) != entry['sha256']:
            raise ValueError('Protocol source changed: ' + str(path))
    return lock


def prepare(dataset, experiment):
    """Copy reviewed complete YAMLs into a deterministic root; never overwrite edits."""
    import shutil
    verify_lock(dataset)
    for stage in STAGES:
        src = config_root(dataset) / stage / 'config.yaml'
        cfg = read_yaml(src)
        validate_config(cfg)
        out = Path(experiment) / stage / 'config.yaml'
        if out.exists() and out.read_bytes() != src.read_bytes():
            raise FileExistsError('Refusing to overwrite existing config: ' + str(out))
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copyfile(src, out)
