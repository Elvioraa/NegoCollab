"""Read-only protocol/source audit, optional complete dataset file audit."""
import argparse
import ast
import copy
import importlib.util
import hashlib
import json
import re
from pathlib import Path

from opencood.tools.nego_protocol import (ROOT, STAGES, MODALITIES, BASE_SHA, active_members,
    config_root, digest, file_sha, lidar_path, read_yaml, validate_config, verify_lock, write_json)


_NATIVE_INTEGRITY_FILES = {'opencood/tools/train_stamp_val_inf.py', 'opencood/tools/train_lsd.py', 'opencood/models/sub_modules/keyfeat_modules.py', 'opencood/tools/train_da.py', 'opencood/tools/val_distance.py', 'opencood/tools/train_nego_vanilla.py', 'opencood/tools/train_utils.py', 'opencood/tools/train_nego_vanilla_onlytrain_ddp.py', 'opencood/tools/train_nego_onlytrain.py', 'opencood/tools/train_nego.py', 'opencood/models/comm_modules/comm_in_pub.py', 'opencood/loss/nego_loss.py', 'opencood/tools/train_stamp_onlytrain.py', 'opencood/loss/nego_loss_ft.py', 'opencood/tools/find_best_ap50_and_save_bak.py', 'opencood/tools/train_da_ddp.py', 'opencood/tools/inference_nego.py', 'opencood/tools/train_nego_val_inf.py', 'opencood/tools/train.py', 'opencood/loss/nego_loss_align.py', 'opencood/tools/train_lsd_ddp.py'}
_WANDB_CLEANED_FILES = {'opencood/tools/val_distance.py', 'opencood/tools/train_stamp_val_inf.py', 'opencood/tools/train_stamp_onlytrain.py', 'opencood/tools/find_best_ap50_and_save_bak.py', 'opencood/tools/train_lsd.py', 'opencood/tools/train_nego_vanilla_onlytrain_ddp.py', 'opencood/tools/train_nego_vanilla.py', 'opencood/tools/train_nego_onlytrain.py', 'opencood/tools/train_da_ddp.py', 'opencood/tools/train_lsd_ddp.py', 'opencood/tools/train_da.py', 'opencood/tools/train_nego.py', 'opencood/tools/train_nego_val_inf.py'}


def _wandb_key_slice(node):
    if isinstance(node, ast.Index):
        node = node.value
    return isinstance(node, ast.Constant) and node.value == 'WANDB_API_KEY'


def _normalized_native_bytes(data, allow_wandb_assignment):
    """Normalize line endings and at most one top-level literal W&B assignment."""
    data = data.replace(b'\r\n', b'\n')
    tree = ast.parse(data)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name) and target.value.value.id == 'os'
            and target.value.attr == 'environ'
            and _wandb_key_slice(target.slice)
            for target in node.targets
        ):
            if (not allow_wandb_assignment or node not in tree.body or len(node.targets) != 1
                    or not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str)
                    or node.lineno != node.end_lineno):
                raise ValueError('Unexpected WANDB_API_KEY assignment form')
            matches.append(node)
    if len(matches) > 1:
        raise ValueError('Multiple WANDB_API_KEY assignments')
    if matches:
        node = matches[0]
        lines = data.splitlines(keepends=True)
        data = b''.join(lines[:node.lineno - 1] + lines[node.end_lineno:])
    return data


def audit_sources():
    # Never return key material. Only file, line and whether code looks active.
    credentials=[]
    invalid_shell=[]
    for path in sorted((ROOT/'opencood/tools').glob('*.py')):
        for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            if re.search(r'(?:WANDB_API_KEY|wandb\.login\s*\(|api_key\s*=)',line,re.I):
                if re.search(r'[a-zA-Z0-9]{32,}',line):
                    credentials.append({'file':path.relative_to(ROOT).as_posix(),'line':number,
                                        'active':not line.lstrip().startswith('#'),'action':'credential rotation recommended if ever valid'})
    for path in sorted((ROOT/'bash_files').glob('*.sh')):
        text=path.read_text(encoding='utf-8')
        if '${yaml_file}' in text and not re.search(r'^\s*yaml_file=',text,re.M):
            invalid_shell.append(path.relative_to(ROOT).as_posix())
    original_unchanged=[]
    manifest = json.loads((ROOT / 'opencood/tools/nego_native_sha256.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('baseline_commit') != BASE_SHA:
        raise ValueError('Invalid native integrity manifest')
    expected_paths = _NATIVE_INTEGRITY_FILES
    if set(manifest['files']) != expected_paths:
        raise ValueError('Native integrity manifest file set changed')
    for rel, entry in manifest['files'].items():
        if entry['normalize_wandb_assignment'] != (rel in _WANDB_CLEANED_FILES):
            raise ValueError('Invalid native normalization policy: ' + rel)
        actual = hashlib.sha256(_normalized_native_bytes((ROOT / rel).read_bytes(), entry['normalize_wandb_assignment'])).hexdigest()
        if actual != entry['sha256']:
            raise ValueError('Unexpected native algorithm change: ' + rel)
        original_unchanged.append(rel)
    return {'credential_locations':credentials,'invalid_yaml_shell_references':invalid_shell,
            'unchanged_native_files':original_unchanged,
            'formal_models':['homo_collab','heter_nego_train','heter_nego_train_align','heter_nego_inf'],
            'legacy_variants_used':False,'native_readme_m3_copy_bug':'README inference copies stage0_m3_collab into stage0_m2_collab; new ownership assembler rejects this'}


def static_check(dataset):
    lock=verify_lock(dataset)
    base=config_root(dataset)
    manifest=read_yaml(base/'manifest.yaml')
    if set(manifest['stages'])!=set(STAGES):
        raise ValueError('Incomplete manifest')
    configs=[]
    for stage in STAGES:
        cfg=read_yaml(base/stage/'config.yaml')
        validate_config(cfg)
        from opencood.hypes_yaml.yaml_utils import load_general_params
        load_general_params(copy.deepcopy(cfg))
        if cfg['nego_protocol']['dataset']!=dataset or cfg['nego_protocol']['stage']!=stage:
            raise ValueError('Stage identity mismatch')
        for m in active_members(stage):
            ref=read_yaml(ROOT/lock['reference_configs'][m]['copy'])
            data=cfg['heter']['modality_setting'][m]
            original=ref['heter']['modality_setting'][m]
            model=cfg['model']['args'][m]
            if m in ('m2','m4'):
                for key in ('grid_conf','data_aug_conf'):
                    if data[key]!=original[key] or model['encoder_args'][key]!=original[key]:
                        raise ValueError('Camera preprocessing differs from DOMA: '+stage+':'+m+':'+key)
            else:
                if data['preprocess']['args']!=original['preprocess']['args']:
                    raise ValueError('Voxel policy differs: '+m)
        configs.append({'stage':stage,'sha256':file_sha(base/stage/'config.yaml'),'status':'PASS'})
    parsed=0
    for folder in (base,ROOT/'docs/protocol_sources'/dataset):
        for f in folder.rglob('*.yaml'):
            read_yaml(f)
            parsed+=1
    assignment_counts={k:e['scenarios'] for k,e in lock['assignments'].items()}
    return {'status':'PASS','yaml_files':parsed,'stage_configs':configs,'assignment_scenarios':assignment_counts,
            'end_to_end':'SERVER_REQUIRED','server_protocol':'SERVER_VALIDATION_REQUIRED'}


def audit_data(dataset,frames_per_scene=0):
    """No data mutation, downloads, exclusions, or fallback sensors."""
    lock=verify_lock(dataset)
    assignments={k:json.loads((ROOT/v['path']).read_text(encoding='utf-8')) for k,v in lock['assignments'].items()}
    counts={}
    scene_sets={}
    for key in ('root_dir','validate_dir','test_dir'):
        root=ROOT/lock['splits'][key]
        if not root.is_dir():
            return {'status':'SERVER_REQUIRED','missing':str(root),'sample_counts':None}
        scene_sets[key]=set(p.resolve() for p in root.iterdir() if p.is_dir())
    keys=list(scene_sets)
    for i,k in enumerate(keys):
        for other in keys[i+1:]:
            if scene_sets[k]&scene_sets[other]:
                raise ValueError('Split directories overlap')
    final=read_yaml(config_root(dataset)/'final_infer/config.yaml')
    excluded={tuple(x) for x in lock['excluded_frames']}
    for key in keys:
        root=ROOT/lock['splits'][key]
        assignment=assignments['evaluation' if key=='test_dir' else 'training']
        scenes=sorted(p for p in root.iterdir() if p.is_dir())
        frame_count=0
        audited=0
        cav_counts={}
        for scene in scenes:
            cavs=sorted(p.name for p in scene.iterdir() if p.is_dir())
            if scene.name not in assignment or set(assignment[scene.name])!=set(cavs):
                raise ValueError('Scenario/CAV assignment mismatch: '+str(scene))
            ordered=list(assignment[scene.name])
            stamps=sorted(p.stem for p in (scene/ordered[0]).glob('*.yaml') if 'additional' not in p.name and (scene.name,p.stem) not in excluded)
            frame_count+=len(stamps)
            cav_counts[scene.name]=len(cavs)
            for stamp in (stamps[:frames_per_scene] if frames_per_scene else stamps):
                audited+=1
                for cav in ordered[:lock['training']['max_cav']]:
                    folder=scene/cav
                    metadata=folder/(stamp+'.yaml')
                    if not metadata.exists():
                        raise FileNotFoundError(metadata)
                    meta=read_yaml(metadata)
                    if 'lidar_pose' not in meta or len(meta['lidar_pose'])!=6:
                        raise ValueError('Missing pose: '+str(metadata))
                    # Native pair training observes every modality at each CAV;
                    # test sensors also require the assigned actual member.
                    members=MODALITIES if key!='test_dir' else (assignment[scene.name][cav],)
                    for member in members:
                        original=folder/(stamp+'.pcd')
                        if member in ('m1','m3'):
                            path=lidar_path(original,dataset,member,require=True)
                            # Header audit only; numerical decoder is exercised by server dataset smoke.
                            with Path(path).open('rb') as f:
                                header=f.read(4096)
                            if b'FIELDS' not in header or b'DATA' not in header:
                                raise ValueError('Invalid PCD header: '+path)
                        else:
                            hdf5=folder/(stamp+'_imgs.hdf5')
                            if dataset=='opv2v' and hdf5.exists():
                                import h5py
                                with h5py.File(hdf5,'r') as f:
                                    for n in range(4):
                                        if 'camera'+str(n) not in f or 'depth'+str(n) not in f:
                                            raise ValueError('Missing camera/depth HDF5 key: '+str(hdf5))
                            else:
                                for n in range(4):
                                    path=folder/f'{stamp}_camera{n}.png'
                                    if not path.exists():
                                        raise FileNotFoundError(path)
                                    if dataset=='opv2v':
                                        depth=Path(str(folder/f'{stamp}_depth{n}.png').replace('OPV2V/','OPV2V_Hetero/').replace('OPV2V\\','OPV2V_Hetero\\'))
                                        if not depth.exists():
                                            raise FileNotFoundError(depth)
                            for n in range(4):
                                if 'camera'+str(n) not in meta:
                                    raise ValueError('Missing camera calibration: '+str(metadata))
        counts[key]={'scenes':len(scenes),'frames':frame_count,'audited_frames':audited,'cav_counts':cav_counts}
    return {'status':'PASS' if not frames_per_scene else 'PASS_SAMPLED','splits':counts,
            'complete_sensor_decode_and_forward':'SERVER_REQUIRED','excluded_frames':list(map(list,excluded)),
            'scene_exclusions':[]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'docs/negocollab_checks.json')
    p.add_argument('--check-data',choices=['opv2v','v2xset'])
    p.add_argument('--frames-per-scene',type=int,default=0)
    args=p.parse_args()
    if args.frames_per_scene<0:
        p.error('frames-per-scene must be >=0')
    report={'base_sha':BASE_SHA,'static':{d:static_check(d) for d in ('opv2v','v2xset')},'native_audit':audit_sources(),
            'dependencies':{m:bool(importlib.util.find_spec(m)) for m in ('torch','torchvision','spconv','torch_scatter','einops','timm','tensorboardX')},
            'training_and_AP':'SERVER_REQUIRED','server_final_protocol':'SERVER_VALIDATION_REQUIRED'}
    if args.check_data:
        try:
            report['data']=audit_data(args.check_data,args.frames_per_scene)
        except Exception as error:
            report['data']={'status':'FAIL','error_type':type(error).__name__,'error':str(error)}
    write_json(args.output,report)
    print(json.dumps({d:report['static'][d]['status'] for d in ('opv2v','v2xset')}))
    if 'data' in report:
        print(json.dumps(report['data'],ensure_ascii=True))
        if report['data']['status'] in ('FAIL','SERVER_REQUIRED'):
            raise SystemExit(2)


if __name__=='__main__':
    main()
