"""Strict stage ownership, source checks and complete epoch-boundary resume."""
import copy
import hashlib
import importlib
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from opencood.tools.nego_protocol import ROOT, MODALITIES, active_members, digest, file_sha, parents

FORMAT='negocollab-doma-v1'
# Reviewed e0 Linux checkout: default native model paths are unchanged by the
# current opt-in pretraining patch. Any other source/current implementation must
# be separately audited; hashes are not evidence of checkpoint authenticity.
REVIEWED_PRETRAINED_COMPATIBILITY = {
    '33ca2a8b10d5bd55847786734c0fbcf5d302e6679aec26513f5b102a6f49e7c3':
        {'source_signature':'909f2af1da80ed8a9ba11f0eb597ce022dc91b053a06802360f304d954fb8ced',
         'allowed_stages':{'pretrain/m1'},
         'target_signatures':{'4d5457fa6bce5c7d2aff099131a739ce7a4e22e77e5037b843ec613d5c967509'}}
}
CODE_FILES=(
    'opencood/models/homo_collab.py','opencood/models/heter_nego_train.py',
    'opencood/models/heter_nego_train_align.py','opencood/models/heter_nego_inf.py',
    'opencood/models/comm_modules/comm_in_pub.py','opencood/models/sub_modules/keyfeat_modules.py',
    'opencood/loss/nego_loss.py','opencood/loss/nego_loss_align.py','opencood/loss/nego_loss_ft.py',
    'opencood/tools/nego_protocol.py','opencood/tools/nego_state.py',
    'opencood/tools/run_nego_doma.py','opencood/data_utils/datasets/nego_protocol_dataset.py',
    'opencood/utils/nego_pcd.py', 'opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py',
    'opencood/data_utils/datasets/intermediate_heter_pair_fusion_dataset.py',
    'opencood/data_utils/datasets/heter_infer/intermediate_heter_infer_fusion_dataset.py')


def code_hash():
    files=set(CODE_FILES)
    for directory in ('opencood/models','opencood/loss','opencood/utils','opencood/data_utils'):
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*.py'))
    files.add('opencood/hypes_yaml/yaml_utils.py')
    return digest({p:file_sha(ROOT/p) for p in sorted(files)})


def model_implementation_signature():
    files=set()
    for directory in ('opencood/models','opencood/utils','opencood/data_utils'):
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*.py'))
    files.add('opencood/hypes_yaml/yaml_utils.py')
    def normalized_sha(path):
        raw=path.read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n')
        return hashlib.sha256(raw).hexdigest()
    return digest({p:normalized_sha(ROOT/p) for p in sorted(files)})


def source_revision():
    try:
        return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],
                                       stderr=subprocess.DEVNULL,timeout=3,text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def model_for(cfg, inference=False):
    args=copy.deepcopy(cfg['model']['args'])
    core=cfg['model']['core_method']
    if inference and not cfg['nego_protocol']['stage'].startswith('pretrain/'):
        core='heter_nego_inf'
        args['stage']='inf'
    module=importlib.import_module('opencood.models.'+core)
    cls=next(cls for name,cls in vars(module).items() if name.lower()==core.replace('_','').lower())
    members=[args['ego_modality']] if core=='homo_collab' else list(args['modality_setting'])
    if any(args[m].get('encoder_args',{}).get('camera_encoder')=='EfficientNet' for m in members):
        from efficientnet_pytorch.utils import url_map
        filename=url_map['efficientnet-b0'].rsplit('/',1)[-1]
        cached=Path(torch.hub.get_dir())/'checkpoints'/filename
        if not cached.is_file():
            raise FileNotFoundError('Preinstall official EfficientNet-B0 weight in torch cache; automatic weight download disabled: '+str(cached))
    model=cls(args)
    if core=='heter_nego_inf':
        # These have no reference in native HeterNegoInf.forward. Preserve them in
        # common_state provenance, not as randomly initialized inference parameters.
        for name in list(model._modules):
            if name=='negotiator' or name=='occ_head_nego' or name.startswith(('nego_reziser_','negotiator_bak_')):
                delattr(model,name)
    return model


def member_of(key):
    root=key.split('.')[0]
    return next((m for m in MODALITIES if root.endswith('_'+m)),None)


def is_receiver(key):
    return key.startswith('comm_') and any(s in key for s in ('receiver','local_prompt_generator','unify2local'))


def trainable_key(key,cfg):
    p=cfg['nego_protocol']
    phase=p['phase']
    if phase=='pretrain':
        return True
    m=member_of(key)
    if phase=='nego':
        return key.startswith('comm_') or 'nego' in key
    if phase=='align':
        return m in p['new_members'] and key.startswith('comm_')
    if phase=='ft':
        return m in p['new_members'] and is_receiver(key)
    return False


def set_training(model,cfg):
    model.eval()
    for key,p in model.named_parameters():
        p.requires_grad=trainable_key(key,cfg)
    # Keep frozen BatchNorm buffers and dropout fixed. Set only fully trainable
    # leaf/subtrees to train, without recursively enabling frozen perception.
    for name,module in model.named_modules():
        params=list(module.named_parameters())
        if params and all(trainable_key(name+'.'+k if name else k,cfg) for k,_ in params):
            module.train()
    actual={k for k,p in model.named_parameters() if p.requires_grad}
    expected={k for k,p in model.named_parameters() if trainable_key(k,cfg)}
    if actual!=expected or (cfg['nego_protocol']['phase']!='inf' and not actual):
        raise ValueError('Incorrect trainable set')
    return sorted(actual)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def validate_model_state(model,state):
    expected=model.state_dict()
    if set(expected)!=set(state):
        raise ValueError('State key mismatch; missing='+repr(sorted(set(expected)-set(state)))+
                         '; extra='+repr(sorted(set(state)-set(expected))))
    for k,v in expected.items():
        if state[k].shape!=v.shape or state[k].dtype!=v.dtype:
            raise ValueError('State shape/dtype mismatch: '+k)
        if state[k].is_floating_point() and not torch.isfinite(state[k]).all():
            raise ValueError('Nonfinite checkpoint tensor: '+k)


def strict_load(model,state):
    validate_model_state(model,state)
    model.load_state_dict(state,strict=True)


def load_stage(path,cfg=None,dataset=None,stage=None,protocol_sha=None,mode='exact',model=None,target_cfg=None):
    if mode not in ('exact','resume','pretrained'):
        raise ValueError('Unknown checkpoint loading mode')
    if mode in ('resume','pretrained') and (cfg is None or model is None or protocol_sha is None):
        raise ValueError('Explicit loading mode requires configuration, model, and protocol provenance')
    if mode=='pretrained' and target_cfg is None:
        raise ValueError('Pretrained loading requires explicit target model configuration')
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing required stage checkpoint: '+str(path))
    checksum=Path(str(path)+'.sha256')
    if not checksum.is_file() or checksum.read_text(encoding='utf-8').split()[0]!=file_sha(path):
        raise ValueError('Checkpoint SHA256 mismatch/missing sidecar: '+str(path))
    # These are locally produced, trusted complete-state checkpoints, not internet downloads.
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    meta=ckpt.get('meta',{})
    if meta.get('format')!=FORMAT:
        raise ValueError('Not a protocol checkpoint: '+str(path))
    dataset=dataset or (cfg['nego_protocol']['dataset'] if cfg else None)
    stage=stage or (cfg['nego_protocol']['stage'] if cfg else None)
    for key,expected in (('dataset',dataset),('stage',stage),('protocol_sha',protocol_sha)):
        if expected is not None and meta.get(key)!=expected:
            raise ValueError('Checkpoint '+key+' mismatch: '+str(path))
    if cfg and meta['config_sha']!=digest(cfg):
        raise ValueError('Checkpoint config mismatch: '+str(path))
    if 'experiment_config_sha' in meta and meta['experiment_config_sha']!=meta['config_sha']:
        raise ValueError('Checkpoint experiment configuration metadata mismatch')
    if mode=='pretrained' and (digest(cfg['model'])!=digest(target_cfg['model']) or model_identity(cfg)!=model_identity(target_cfg)):
        raise ValueError('Pretrained target model configuration or identity mismatch')
    if meta['state_schema']!=digest({k:[list(v.shape),str(v.dtype)] for k,v in ckpt['model'].items()}):
        raise ValueError('Corrupt state schema')
    current_code=code_hash()
    if meta['code_sha']!=current_code:
        signature=model_implementation_signature()
        declaration=REVIEWED_PRETRAINED_COMPATIBILITY.get(meta['code_sha'],{})
        if (mode!='pretrained' or meta['stage'] not in declaration.get('allowed_stages',set())
                or signature not in declaration.get('target_signatures',set())
                or ('model_implementation_signature' in meta
                    and meta['model_implementation_signature']!=declaration.get('source_signature'))):
            raise ValueError('Checkpoint implementation hash differs; audit/migrate explicitly: '+str(path))
    if mode in ('resume','pretrained'):
        if model.__class__.__name__.lower()!=cfg['model']['core_method'].replace('_','').lower():
            raise ValueError('Checkpoint target model type mismatch')
        validate_model_state(model,ckpt['model'])
        if 'model_config_sha' in meta and meta['model_config_sha']!=digest(cfg['model']):
            raise ValueError('Checkpoint model configuration mismatch')
        if 'model_identity' in meta and meta['model_identity']!=model_identity(cfg):
            raise ValueError('Checkpoint model identity mismatch')
        if 'model_implementation_signature' in meta and meta['code_sha']==current_code and meta['model_implementation_signature']!=model_implementation_signature():
            raise ValueError('Checkpoint model implementation mismatch')
    if mode=='resume':
        state=ckpt.get('resume',{})
        if not all(k in state for k in ('optimizer','scheduler','rng','completed_epoch','global_step')):
            raise ValueError('Checkpoint lacks complete resume state')
        if (isinstance(state['completed_epoch'],bool) or not isinstance(state['completed_epoch'],int)
                or state['completed_epoch']<0 or state['completed_epoch']!=meta['completed_epoch']
                or isinstance(state['global_step'],bool) or not isinstance(state['global_step'],int) or state['global_step']<0
                or meta.get('global_step')!=state['global_step']
                or not isinstance(state['optimizer'],dict) or not isinstance(state['scheduler'],dict) or not isinstance(state['rng'],dict)):
            raise ValueError('Invalid checkpoint resume epoch/global_step')
        if (not {'state','param_groups'}.issubset(state['optimizer'])
                or not isinstance(state['optimizer']['param_groups'],list) or not state['optimizer']['param_groups']
                or not state['scheduler'] or not {'python','numpy','torch','cuda','generator'}.issubset(state['rng'])):
            raise ValueError('Invalid checkpoint optimizer/scheduler/RNG state')
    return ckpt


def model_identity(cfg):
    stage=cfg['nego_protocol']['stage']
    return {'core_method':cfg['model']['core_method'],'members':active_members(stage),'stage':stage}


def source_for(key,stage):
    """Exactly one owner for each non-random tensor; m3/m4 cannot substitute."""
    m=member_of(key)
    if stage=='alliance/nego':
        return None if key.startswith(('comm_','nego','occ_head_nego')) else 'pretrain/'+m
    if stage.endswith('/ft'):
        return 'alliance/nego' if stage=='alliance/ft' else stage.split('/')[0]+'/align'
    if stage.endswith('/align'):
        new=stage.split('/')[0]
        if m==new:
            return None if key.startswith('comm_') else 'pretrain/'+new
        if key.startswith(('negotiator.','nego_reziser_','occ_head_nego.')):
            return 'alliance/nego'
        return 'alliance/ft'
    if stage=='final_infer':
        if m is None:
            raise ValueError('Unexpected inference shared parameter: '+key)
        return 'alliance/ft' if m in ('m1','m2') else m+'/ft'
    if stage.startswith('pretrain/'):
        return None
    raise ValueError(stage)


def initialize(model,cfg,experiment,parent_models=None):
    p=cfg['nego_protocol']
    stage=p['stage']
    protocol_sha=file_sha(ROOT/p['protocol_lock'])
    loaded={}
    for parent in parents(stage):
        parent_cfg_path=Path(experiment)/parent/'config.yaml'
        from opencood.tools.nego_protocol import read_yaml
        parent_cfg=read_yaml(parent_cfg_path)
        if parent_cfg['nego_protocol']['stage']!=parent or parent_cfg['nego_protocol']['dataset']!=p['dataset']:
            raise ValueError('Wrong parent config identity: '+parent)
        if parent_models is None:
            loaded[parent]=load_stage(Path(experiment)/parent/'best.pth',cfg=parent_cfg,protocol_sha=protocol_sha)
        else:
            if callable(parent_models):
                expected_parent=parent_models(parent,parent_cfg)
            else:
                if parent not in parent_models:
                    raise ValueError('Missing expected parent model: '+parent)
                expected_parent=parent_models[parent]
            loaded[parent]=load_stage(Path(experiment)/parent/'best.pth',cfg=parent_cfg,protocol_sha=protocol_sha,
                                      mode='pretrained',model=expected_parent,target_cfg=parent_cfg)
            del expected_parent
    state=cpu_state(model)
    owners={}
    for key,tensor in state.items():
        owner=source_for(key,stage)
        owners[key]=owner or 'fresh:'+stage
        if owner is None:
            continue
        source=loaded[owner]['model']
        if key not in source or source[key].shape!=tensor.shape or source[key].dtype!=tensor.dtype:
            raise ValueError('Missing/incompatible required tensor '+owner+':'+key)
        state[key]=source[key]
    strict_load(model,state)
    common={}
    if stage=='alliance/nego':
        pass  # captured from the trained negotiation checkpoint by pack().
    elif loaded:
        candidates=[c['common_state'] for c in loaded.values() if c.get('common_state')]
        if candidates:
            common=candidates[0]
            for other in candidates[1:]:
                if set(other)!=set(common) or any(not torch.equal(v,other[k]) for k,v in common.items()):
                    raise ValueError('Incompatible negotiated common spaces')
    provenance={parent:{'path':str(Path(experiment)/parent/'best.pth'),
                        'sha256':file_sha(Path(experiment)/parent/'best.pth'),
                        'config_sha':loaded[parent]['meta']['config_sha']} for parent in loaded}
    return owners,provenance,common


def rng_state(generator):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'generator':generator.get_state()}


def restore_rng(state,generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA training resume requires CUDA')
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['generator'])


def pack(model,cfg,owners,provenance,common,epoch,optimizer=None,scheduler=None,generator=None,best=None,global_step=None):
    state=cpu_state(model)
    if cfg['nego_protocol']['stage']=='alliance/nego':
        common={k:v for k,v in state.items() if k.startswith(('negotiator.','nego_reziser_','occ_head_nego.'))}
    meta={'format':FORMAT,'dataset':cfg['nego_protocol']['dataset'],'stage':cfg['nego_protocol']['stage'],
          'config_sha':digest(cfg),'code_sha':code_hash(),'protocol_sha':file_sha(ROOT/cfg['nego_protocol']['protocol_lock']),
          'seed':cfg['seed'],'state_schema':digest({k:[list(v.shape),str(v.dtype)] for k,v in state.items()}),
          'owners':owners,'parents':provenance,'completed_epoch':epoch,
          'source_revision':source_revision(), 'model_identity':model_identity(cfg),
          'model_config_sha':digest(cfg['model']), 'experiment_config_sha':digest(cfg),
          'model_implementation_signature':model_implementation_signature(), 'global_step':global_step}
    result={'meta':meta,'model':state,'common_state':common,'best':best}
    if optimizer is not None:
        result['resume']={'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                          'rng':rng_state(generator),'completed_epoch':epoch}
        if global_step is not None:
            result['resume']['global_step'] = global_step
    return result


def atomic_save(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)
    Path(str(path)+'.sha256').write_text(file_sha(path)+'  '+path.name+'\n',encoding='utf-8')
