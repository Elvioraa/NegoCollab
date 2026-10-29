"""One-shot local reference extraction. Refuses to replace any different output."""
import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml
from opencood.tools.nego_protocol import (ROOT, BASE_SHA, MODALITIES, STAGES, active_members,
                                         config_root, digest, file_sha, parents)


def put(path, content):
    path = Path(path)
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise FileExistsError('Refusing to replace ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def dump(path, value):
    put(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def set_canvas(value, canvas):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('lidar_range', 'cav_lidar_range', 'gt_range'):
                value[key] = list(canvas)
            else:
                set_canvas(item, canvas)
    elif isinstance(value, list):
        for item in value:
            set_canvas(item, canvas)


def generate(reference):
    source_root = ROOT / 'docs/protocol_sources'
    snapshot_path = reference / 'opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2_R2_TABLE4_EXT/history_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    commit = subprocess.check_output(['git','-c','safe.directory='+reference.as_posix(),'rev-parse','HEAD'], cwd=reference, text=True).strip()
    native = ROOT / 'opencood/hypes_yaml/opv2v'
    native_all = yaml.safe_load((native/'NGCB/inference/m1m2m3m4_pyramid.yaml').read_text(encoding='utf-8'))
    negotiation = yaml.safe_load((native/'NGCB/stage1/m1m2_pyramid.yaml').read_text(encoding='utf-8'))
    homo = yaml.safe_load((native/'Homo/m1_pointpillar_pyramid.yaml').read_text(encoding='utf-8'))
    for dataset in ('opv2v', 'v2xset'):
        refs, provenance = {}, {}
        names = {'m1':'stage1/m1_base', 'm2':'stage2/m2_alignto_m1_lr0p0012',
                 'm3':'stage2/m3_alignto_m1_lr0p0030', 'm4':'stage2/m4_alignto_m1_lr0p0017', 'final':'final_infer'}
        for m in (*MODALITIES, 'final'):
            if dataset == 'opv2v':
                key = f'opencood/logs/DOMA_V2_R2_2GPU_LR1P414_SEED303/{names[m]}/config.yaml'
                content = snapshot[key]
                origin = str(snapshot_path) + '#' + key
            else:
                name = 'stage1/m1' if m == 'm1' else ('final_infer/m1m2m3m4' if m == 'final' else 'stage2/'+m)
                path = reference / 'opencood/hypes_yaml/v2xset/MoreModality/DOMA/V2' / (name+'.yaml')
                content = path.read_text(encoding='utf-8')
                origin = str(path)
            out = source_root / dataset / (m+'.yaml')
            put(out, content)
            refs[m] = yaml.safe_load(content)
            provenance[m] = {'origin':origin, 'copy':out.relative_to(ROOT).as_posix(), 'sha256':file_sha(out),
                             'status':'saved_run_snapshot' if dataset=='opv2v' else 'local_reference_only'}
        assignments = {}
        for kind, suffix in (('training',''), ('evaluation','_in_order')):
            rel = Path('opencood/modality_assign') / f'{dataset}_4modality{suffix}.json'
            source = reference / rel
            # Isolated, dataset-specific source copy. Preserve insertion order exactly.
            out = config_root(dataset) / 'assignments' / rel.name
            put(out, source.read_text(encoding='utf-8'))
            value = json.loads(out.read_text(encoding='utf-8'))
            assignments[kind] = {'path':out.relative_to(ROOT).as_posix(), 'sha256':file_sha(out),
                                  'scenarios':len(value), 'origin':str(source),
                                  'ordered_sha256':digest(list((k,list(v.items())) for k,v in value.items()))}
        final = refs['final']
        eval_canvas = [-204.8,-102.4,-3,204.8,102.4,1]
        # Values sourced from DOMA's ordered entry; actual server CLI is unavailable.
        lock = {'schema_version':1, 'dataset':dataset, 'base_sha':BASE_SHA, 'reference_doma_commit':commit,
                'seed':303, 'reference_configs':provenance, 'assignments':assignments,
                'splits':{k:final[k] for k in ('root_dir','validate_dir','test_dir')},
                'sensor_policy':{'m1':'original PCD + PointPillar','m2':'LSS EfficientNet',
                                 'm3':'OPV2V_Hetero *_32.pcd + SECOND' if dataset=='opv2v' else 'original PCD + SECOND',
                                 'm4':'LSS Resnet101'},
                'training':{'comm_range':70,'max_cav':5,'canvas':negotiation['cav_lidar_range'],
                            'semantics':'native homogeneous collaboration; no CodeAlign scene sharding'},
                'evaluation':{'range':eval_canvas,'comm_range':180,'max_cav':5,'use_cav':[1,2,3,4],
                              'ego':'m1','iou':[0.3,0.5,0.7], 'gt':'all communication-range CAVs before input prefix cutoff; lidar labels',
                              'postprocess_source':final['postprocess'], 'runtime_confirmation':'SERVER_VALIDATION_REQUIRED'},
                'scene_exclusions':[], 'excluded_frames':[['2021_08_20_21_10_24','000265']] if dataset=='opv2v' else [],
                'exclusion_note':'Known broken OPV2V frame matches local DOMA. No V2XSet scene excluded; camera-incomplete scene policy requires server DOMA ledger.',
                'split_counts':'SERVER_REQUIRED: real datasets absent locally',
                'server_validation_required':['actual split/frame and scene-exclusion ledger', 'DOMA final CLI/range/GT/postprocess',
                                               'CUDA/spconv full perception forward/backward', 'trained checkpoints and final AP']}
        dump(config_root(dataset)/'protocol_lock.yaml',lock)
        entries = {}
        for stage in STAGES:
            members = active_members(stage)
            pretrain = stage.startswith('pretrain/')
            isfinal = stage == 'final_infer'
            phase = 'pretrain' if pretrain else ('inf' if isfinal else stage.split('/')[1])
            new = members if stage.startswith('alliance/') else ([stage.split('/')[0]] if not pretrain and not isfinal else [])
            canvas = list(refs[members[0]]['cav_lidar_range'] if pretrain else negotiation['cav_lidar_range'])
            if isfinal:
                canvas = eval_canvas
            data, models = {}, {}
            for m in members:
                sensor = copy.deepcopy(refs[m]['heter']['modality_setting'][m])
                smodel = refs[m]['model']['args'][m]
                model = copy.deepcopy(native_all['model']['args'][m])
                # Sensing/backbone identities from DOMA, communication and fusion from NegoCollab.
                for key in ('core_method','sensor_type','encoder_args','backbone_args','camera_mask_args'):
                    if key in smodel:
                        model[key] = copy.deepcopy(smodel[key])
                model['core_method'] = sensor['core_method']
                model['sensor_type'] = sensor['sensor_type']
                model['aligner_args'] = {'core_method':'identity'}
                model['allied'] = m not in new
                model.pop('model_dir',None)
                if model['sensor_type']=='camera' and dataset=='v2xset':
                    model['encoder_args']['depth_supervision'] = False
                    model['encoder_args']['use_depth_gt'] = False
                set_canvas(sensor,canvas)
                set_canvas(model,canvas)
                data[m], models[m] = sensor,model
            if pretrain:
                mapping = dict.fromkeys(MODALITIES,members[0])
                ego = members[0]
            elif stage.startswith('alliance/'):
                mapping = {'m1':'m1','m2':'m1','m3':'m2','m4':'m2'}
                ego = 'm1&m2'
            elif isfinal:
                mapping = dict(zip(MODALITIES,MODALITIES))
                ego = 'm1'
            else:
                m = stage.split('/')[0]
                mapping = {'m1':m,'m2':m,'m3':'m1','m4':'m2'}
                ego = m
            if pretrain:
                budget = copy.deepcopy(homo)
                budget_source = 'opencood/hypes_yaml/opv2v/Homo/m1_pointpillar_pyramid.yaml'
                train = budget['train_params']
                optimizer, scheduler = budget['optimizer'],budget['lr_scheduler']
                loss = copy.deepcopy(homo['loss'])
                core = 'homo_collab'
            elif isfinal:
                train = {'batch_size':1,'epoches':0,'max_cav':5,'eval_freq':1,'save_freq':1}
                optimizer, scheduler = copy.deepcopy(homo['optimizer']),copy.deepcopy(homo['lr_scheduler'])
                loss = copy.deepcopy(homo['loss'])
                core, budget_source = 'heter_nego_inf','not a training stage'
            else:
                path = native/'NGCB'/('stage1/m1m2_pyramid.yaml' if stage.startswith('alliance/') else f'stage2/m1m2_guide_{stage.split("/")[0]}_pyramid.yaml')
                budget = yaml.safe_load(path.read_text(encoding='utf-8'))
                b = copy.deepcopy(budget['train_setting'][phase])
                train, optimizer, scheduler = b['train_params'],b['optimizer'],b['lr_scheduler']
                loss = copy.deepcopy(budget['loss'])
                loss['core_method'] = loss['core_method'][phase]
                core = {'nego':'heter_nego_train','align':'heter_nego_train_align','ft':'heter_nego_inf'}[phase]
                budget_source = path.relative_to(ROOT).as_posix() + '#train_setting.'+phase
            # Native FT selects one ego modality per batch. Budget is explicit and editable.
            train['batch_size']=1
            train['num_workers']=0
            train['max_cav']=5
            for key in ('eps','weight_decay'):
                if key in optimizer.get('args',{}):
                    optimizer['args'][key]=float(optimizer['args'][key])
            post=copy.deepcopy(final['postprocess'])
            set_canvas(post,canvas)
            post['max_num']=final['postprocess']['max_num']
            preprocess=copy.deepcopy(homo['preprocess'])
            set_canvas(preprocess,canvas)
            fusion = 'intermediateheterpair' if phase in ('nego','align') else 'intermediateheter'
            if isfinal:
                fusion='intermediateheterinfer'
            cfg={'name':f'NegoCollab_{dataset}_{stage.replace("/","_")}', 'seed':303,
                 **{k:final[k] for k in ('root_dir','validate_dir','test_dir')},
                 'yaml_parser':'load_general_params','train_params':train,'optimizer':optimizer,'lr_scheduler':scheduler,
                 'comm_range':180 if isfinal else 70, 'label_type':'lidar', 'cav_lidar_range':canvas,
                 'input_source':['lidar']+(['camera'] if any(m in members for m in ('m2','m4')) else [])+
                                (['depth'] if dataset=='opv2v' and any(m in members for m in ('m2','m4')) else []),
                 'add_data_extension':final.get('add_data_extension',[]) if dataset=='opv2v' else [],
                 'heter':{'assignment_path':assignments['evaluation' if isfinal else 'training']['path'],
                          'ego_modality':ego,'mapping_dict':mapping,'modality_setting':data,
                          'lidar_channels_dict':{'m3':32} if dataset=='opv2v' and 'm3' in members else {}},
                 'fusion':{'core_method':fusion,'dataset':dataset,'args':{'proj_first':False,'grid_conf':None,'data_aug_conf':None}},
                 'preprocess':preprocess,'postprocess':post, 'noise_setting':{'add_noise':False},
                 'model':{'core_method':core,'args':{'ego_modality':ego,'modality_setting':copy.deepcopy(data),
                          'mapping_dict':mapping,'doma_protocol':True,'stage':phase,'lidar_range':canvas,
                          'supervise_single':True,'pub':copy.deepcopy(negotiation['model']['args']['pub']), **models}},
                 'loss':loss, 'nego_protocol':{'dataset':dataset,'stage':stage,'phase':phase,'seed':303,'new_members':new,
                         'parents':parents(stage),'protocol_lock':(config_root(dataset)/'protocol_lock.yaml').relative_to(ROOT).as_posix(),
                         'scene_exclusions':[],'excluded_frames':lock['excluded_frames'],'budget_source':budget_source,
                         'checkpoint_selection':'validate AP50' if not pretrain else 'validate loss',
                         'batch_note':'batch=1 required by native mixed-ego FT; pretrain reduced from native batch=2 for CPU-reviewed contract'}}
            if pretrain:
                cfg['model']['args']['ego_modality']=members[0]
            if isfinal:
                cfg['use_cav']=4
            dump(config_root(dataset)/stage/'config.yaml',cfg)
            entries[stage]={'config':stage+'/config.yaml','parents':parents(stage),'checkpoint':'best.pth' if not isfinal else 'assembled.pth',
                            'core_method':core,'phase':phase,'budget_source':budget_source}
        dump(config_root(dataset)/'manifest.yaml',{'dataset':dataset,'stages':entries,
                    'parallelism':'pretrain independent; m3/m4 branches independent after frozen initial alliance; final waits for both'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--doma',type=Path,required=True)
    generate(p.parse_args().doma)
