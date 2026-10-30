"""Explicit single-GPU stages, strict assembly and ordered AP export. No subprocess jobs."""
import argparse
import copy
import csv
import importlib
import json
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.tools.nego_protocol import (ROOT, MODALITIES, STAGES, assert_prefix, digest,
    file_sha, prepare, read_yaml, seed_all, seed_worker, validate_config, validate_runtime_batch, verify_lock, write_json)
from opencood.tools.nego_state import (atomic_save, cpu_state, initialize, load_stage, model_for,
    pack, restore_rng, rng_state, set_training, strict_load, trainable_key)


def parsed(cfg):
    from opencood.hypes_yaml.yaml_utils import load_general_params
    return load_general_params(copy.deepcopy(cfg))


def to_device(value,device):
    if isinstance(value,torch.Tensor):
        return value.to(device)
    if isinstance(value,dict):
        return {k:to_device(v,device) for k,v in value.items()}
    if isinstance(value,list):
        return [to_device(v,device) for v in value]
    if isinstance(value,tuple):
        return tuple(to_device(v,device) for v in value)
    return value


def loader(dataset,cfg,train,generator):
    collate = dataset.collate_batch_train if train else dataset.collate_batch_test
    if train and cfg.get('nego', {}).get('multi_batch', {}).get('enabled', False):
        from opencood.data_utils.datasets.nego_m1_batch import collate_m1_vectorized
        collate = partial(collate_m1_vectorized, dataset.collate_batch_train)
    return DataLoader(dataset,batch_size=cfg['train_params']['batch_size'] if train else 1,
          shuffle=train,drop_last=False,num_workers=cfg['train_params'].get('num_workers',0) if train else 0,
          collate_fn=collate,
          worker_init_fn=seed_worker,generator=generator,persistent_workers=False,pin_memory=False)


def check_process():
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('DDP is not supported by this reviewed native mixed-ego/FT path; use independent single-GPU stages')


def require_cuda():
    check_process()
    if not torch.cuda.is_available():
        raise RuntimeError('SERVER_REQUIRED: CUDA/spconv perception runtime unavailable; CPU core tests are separate')
    return torch.device('cuda:0')


def criterion_for(cfg):
    core=cfg['loss']['core_method']
    module=importlib.import_module('opencood.loss.'+core)
    cls=next(c for n,c in vars(module).items() if n.lower()==core.replace('_','').lower())
    return cls(cfg['loss']['args'])


def loss_for(model,criterion,batch,cfg,epoch):
    ego=batch['ego']
    phase=cfg['nego_protocol']['phase']
    ego['stage']=phase
    ego['epoch']=epoch
    output=model(ego)
    output['stage']=phase
    output['epoch']=epoch
    if phase=='pretrain':
        loss=criterion(output,ego['label_dict'])
        if cfg['model']['args']['supervise_single']:
            loss=loss+criterion(output,ego['label_dict_single'],suffix='_single')*cfg['train_params'].get('single_weight',1)
        return loss
    return criterion(output,ego['label_dict'],ego['label_dict_single'])


def stats_template():
    return {iou:{'tp':[],'fp':[],'score':[],'gt':0} for iou in (0.3,0.5,0.7)}


def evaluate(model,dataset,cfg,device,generator,output_dir=None,requested=None,checkpoint_sha=None):
    from opencood.utils import eval_utils
    model.eval()
    stats=stats_template()
    ledger=[]
    actual_calls=[]
    hooks=[]
    for member in MODALITIES:
        encoder=getattr(model,'encoder_'+member,None)
        if encoder is not None:
            hooks.append(encoder.register_forward_pre_hook(lambda mod,args,m=member:actual_calls.append(m)))
    for index,batch in enumerate(loader(dataset,cfg,False,generator)):
        if batch is None:
            raise ValueError('Dropped evaluation frame: '+str(index))
        members=list(batch['ego']['agent_modality_list'])
        if requested is not None:
            assert_prefix(members,requested)
            if len(members)>requested or members[0]!='m1':
                raise ValueError('Illegal actual forward members')
        batch=to_device(batch,device)
        actual_calls.clear()
        with torch.no_grad():
            output=model(batch['ego'])
            boxes,scores,gt=dataset.post_process(batch,{'ego':output})
        if set(actual_calls)!=set(members) or len(actual_calls)!=len(set(members)):
            raise ValueError('Encoder calls do not match actual Dataset members: '+repr(actual_calls))
        for iou in stats:
            eval_utils.caluclate_tp_fp(boxes,scores,gt,stats,iou)
        gt_array=gt.detach().cpu().numpy()
        identity=copy.deepcopy(dataset.protocol_identities[index])
        # IDs after communication filtering, before input truncation are native output.
        all_in_range=list(batch['ego']['cav_id_list'])
        ledger.append({**identity,'sample_idx':index,'in_range_cav_ids':all_in_range,
                       'input_cav_ids':all_in_range[:len(members)],'members':members,
                       'requested':requested,'realized':len(members),'gt_count':len(gt_array),
                       'gt_sha256':digest(gt_array.tolist()),'forward_encoders':list(actual_calls)})
    for hook in hooks:
        hook.remove()
    if not ledger or len(ledger)!=len(dataset) or stats[0.5]['gt']==0:
        raise ValueError('Empty evaluation, dropped frames or zero total GT')
    aps={f'AP@{iou:.1f}':float(eval_utils.calculate_ap(stats,iou)[0]) for iou in stats}
    result={'dataset':cfg['nego_protocol']['dataset'],'stage':cfg['nego_protocol']['stage'],
            'requested':requested,'AP':aps,'statistics':stats,'frames':ledger,
            'config_sha256':digest(cfg),'protocol_sha256':file_sha(ROOT/cfg['nego_protocol']['protocol_lock']),
            'checkpoint_sha256':checkpoint_sha,
            'protocol_status':'SERVER_VALIDATION_REQUIRED'}
    if output_dir:
        out=Path(output_dir)
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'result.json',result)
        with (out/'ap.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f)
            writer.writerow(['CAV','AP@0.3','AP@0.5','AP@0.7'])
            writer.writerow([requested,*[format(aps[f'AP@{x:.1f}'],'.17g') for x in (0.3,0.5,0.7)]])
    return result


def train(experiment,stage,resume=False):
    from opencood.data_utils.datasets.nego_protocol_dataset import build_protocol_dataset
    from tensorboardX import SummaryWriter
    cfg=read_yaml(Path(experiment)/stage/'config.yaml')
    validate_config(cfg)
    validate_runtime_batch(cfg)
    if cfg['nego_protocol']['stage']!=stage or stage=='final_infer':
        raise ValueError('Wrong stage YAML')
    verify_lock(cfg['nego_protocol']['dataset'])
    device=require_cuda()
    stage_dir=Path(experiment)/stage
    if not resume and any(stage_dir.glob('*.pth')):
        raise FileExistsError('Existing checkpoint; use --resume for complete training state')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(stage_dir/'train.log',encoding='utf-8'),logging.StreamHandler()])
    seed_all(cfg['seed'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    runtime=parsed(cfg)
    model=model_for(runtime).to(device)
    if resume:
        ckpt=load_stage(stage_dir/'last.pth',cfg=cfg,protocol_sha=file_sha(ROOT/cfg['nego_protocol']['protocol_lock']))
        strict_load(model,ckpt['model'])
        owners,provenance,common=ckpt['meta']['owners'],ckpt['meta']['parents'],ckpt['common_state']
    else:
        owners,provenance,common=initialize(model,cfg,experiment)
    trainable=set_training(model,cfg)
    write_json(stage_dir/'trainable_parameters.json',trainable)
    train_data=build_protocol_dataset(runtime,train=True)
    val_cfg=copy.deepcopy(runtime)
    # Alignment/negotiation selection uses the native final Sender/Receiver graph,
    # evaluated on VALIDATE only. No test-set model selection or asynchronous GPU job.
    if cfg['nego_protocol']['phase'] in ('nego','align'):
        val_cfg['fusion']['core_method']='intermediateheter'
    val_data=build_protocol_dataset(val_cfg,train=False)
    criterion=criterion_for(cfg).to(device)
    opt_cfg=cfg['optimizer']
    optimizer=getattr(torch.optim,opt_cfg['core_method'])([p for p in model.parameters() if p.requires_grad],
                          lr=opt_cfg['lr'],**opt_cfg.get('args',{}))
    from opencood.tools.train_utils import setup_lr_schedular
    scheduler=setup_lr_schedular(cfg,optimizer,len(loader(train_data,cfg,True,generator)))
    start=0
    best=float('inf') if cfg['nego_protocol']['phase']=='pretrain' else float('-inf')
    if resume:
        state=ckpt['resume']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'],generator)
        start=state['completed_epoch']
        best=ckpt['best']
    frozen={k:v for k,v in cpu_state(model).items() if not trainable_key(k,cfg)}
    writer=SummaryWriter(str(stage_dir/'tensorboard'))
    write_json(stage_dir/'provenance.json',{'parents':provenance,'config_sha256':digest(cfg),
          'protocol_sha256':file_sha(ROOT/cfg['nego_protocol']['protocol_lock']),
          'code_sha256':__import__('opencood.tools.nego_state',fromlist=['code_hash']).code_hash(),
          'torch':torch.__version__,'cuda':torch.version.cuda,'DDP':False})
    for epoch in range(start,cfg['train_params']['epoches']):
        # Rebuild the random CAV/ego assignment identically on fresh run or resume.
        seed_all(cfg['seed']+epoch)
        generator.manual_seed(cfg['seed']+epoch)
        train_data.reinitialize()
        set_training(model,cfg)
        losses=[]
        for index,batch in enumerate(loader(train_data,cfg,True,generator)):
            if batch is None:
                raise ValueError('Empty training sample')
            if batch['ego']['object_bbx_mask'].sum()==0:
                continue  # Same empty-positive training skip as native trainer.
            batch=to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_for(model,criterion,batch,cfg,epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite native loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            writer.add_scalar('train/loss',losses[-1],epoch*len(train_data)+index)
            if cfg['lr_scheduler']['core_method']=='cosineannealwarm':
                scheduler.step_update(epoch*len(train_data)+index)
        if not losses:
            raise ValueError('No trainable positive samples')
        if cfg['lr_scheduler']['core_method']!='cosineannealwarm':
            scheduler.step()
        current=cpu_state(model)
        if any(not torch.equal(v,current[k]) for k,v in frozen.items()):
            raise ValueError('Frozen parameter/buffer changed')
        if (epoch+1)%cfg['train_params']['eval_freq']==0 or epoch+1==cfg['train_params']['epoches']:
            saved_rng=rng_state(generator)
            seed_all(cfg['seed'])
            if cfg['nego_protocol']['phase']=='pretrain':
                model.eval()
                values=[]
                with torch.no_grad():
                    for batch in loader(val_data,cfg,False,generator):
                        values.append(float(loss_for(model,criterion,to_device(batch,device),cfg,epoch)))
                metric=float(np.mean(values))
                improved=metric<best
            else:
                validation=model_for(parsed(val_cfg),inference=True).to(device)
                strict_load(validation,{k:current[k] for k in validation.state_dict()})
                report=evaluate(validation,val_data,val_cfg,device,generator,stage_dir/f'validation/epoch_{epoch+1}')
                metric=report['AP']['AP@0.5']
                improved=metric>best
                del validation
            restore_rng(saved_rng,generator)
            if improved:
                best=metric
                atomic_save(stage_dir/'best.pth',pack(model,cfg,owners,provenance,common,epoch+1,best=best))
            writer.add_scalar('validate/selection_metric',metric,epoch+1)
            logging.info('epoch=%d train_loss=%.8g validate_metric=%.17g best=%.17g',epoch+1,np.mean(losses),metric,best)
        payload=pack(model,cfg,owners,provenance,common,epoch+1,optimizer,scheduler,generator,best)
        atomic_save(stage_dir/'last.pth',payload)
        if (epoch+1)%cfg['train_params']['save_freq']==0:
            atomic_save(stage_dir/f'epoch_{epoch+1}.pth',payload)
    writer.close()


def assemble(experiment):
    cfg=read_yaml(Path(experiment)/'final_infer/config.yaml')
    validate_config(cfg)
    verify_lock(cfg['nego_protocol']['dataset'])
    out=Path(experiment)/'final_infer/assembled.pth'
    if out.exists():
        raise FileExistsError(out)
    seed_all(cfg['seed'])
    model=model_for(parsed(cfg))
    owners,provenance,common=initialize(model,cfg,experiment)
    if not common or not any(k.startswith('negotiator.') for k in common):
        raise ValueError('Missing negotiated common-space provenance')
    atomic_save(out,pack(model,cfg,owners,provenance,common,0))
    write_json(out.with_suffix('.sources.json'),{'parents':provenance,'owners':owners,'checkpoint_sha256':file_sha(out)})


def smoke(experiment,stage):
    """Explicit server-only one-batch forward/backward; does not update weights."""
    from opencood.data_utils.datasets.nego_protocol_dataset import build_protocol_dataset
    cfg=read_yaml(Path(experiment)/stage/'config.yaml')
    validate_config(cfg)
    validate_runtime_batch(cfg)
    verify_lock(cfg['nego_protocol']['dataset'])
    if stage=='final_infer':
        raise ValueError('Use infer for final model checks')
    device=require_cuda()
    seed_all(cfg['seed'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    runtime=parsed(cfg)
    model=model_for(runtime).to(device)
    owners,provenance,common=initialize(model,cfg,experiment)
    trainable=set_training(model,cfg)
    dataset=build_protocol_dataset(runtime,train=True)
    batch=next(iter(loader(dataset,cfg,True,generator)))
    criterion=criterion_for(cfg).to(device)
    loss=loss_for(model,criterion,to_device(batch,device),cfg,0)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite smoke loss')
    loss.backward()
    gradients=[k for k,p in model.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0]
    if not gradients or any(not p.requires_grad and p.grad is not None for p in model.parameters()):
        raise ValueError('Missing trainable gradients or unexpected frozen gradients')
    report={'status':'PASS','stage':stage,'dataset':cfg['nego_protocol']['dataset'],
            'loss':float(loss.detach()),'trainable_parameters':trainable,'nonzero_gradients':gradients,
            'parents':provenance,'config_sha256':digest(cfg),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
            'optimizer_step':False}
    write_json(Path(experiment)/stage/'smoke.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','stage','loss','peak_allocated_bytes')}))


def infer(experiment,use_cav):
    from opencood.data_utils.datasets.nego_protocol_dataset import build_protocol_dataset
    cfg=read_yaml(Path(experiment)/'final_infer/config.yaml')
    validate_config(cfg)
    validate_runtime_batch(cfg)
    verify_lock(cfg['nego_protocol']['dataset'])
    device=require_cuda()
    ckpt=load_stage(Path(experiment)/'final_infer/assembled.pth',cfg=cfg,
                    protocol_sha=file_sha(ROOT/cfg['nego_protocol']['protocol_lock']))
    if use_cav not in (1,2,3,4):
        raise ValueError('use_cav must be 1..4')
    out=Path(experiment)/f'final_infer/CAV{use_cav}'
    if (out/'result.json').exists():
        raise FileExistsError('Evaluation already exists: '+str(out))
    seed_all(cfg['seed'])
    runtime=parsed(cfg)
    runtime['use_cav']=use_cav
    runtime['validate_dir']=runtime['test_dir']
    dataset=build_protocol_dataset(runtime,train=False)
    if use_cav==1:
        # Exact native homogeneous m1 path, using its frozen Stage-0 perception,
        # fusion and head tensors retained in alliance FT/final assembly.
        single=copy.deepcopy(runtime)
        single['model']['core_method']='homo_collab'
        single['model']['args']['ego_modality']='m1'
        model=model_for(single).to(device)
    else:
        model=model_for(runtime,inference=True).to(device)
    strict_load(model,{k:ckpt['model'][k] for k in model.state_dict()})
    # Record which actual sensors entered encoders for each call, including CAV1.
    calls=[]
    hooks=[]
    for member in MODALITIES:
        encoder=getattr(model,'encoder_'+member,None)
        if encoder is not None:
            hooks.append(encoder.register_forward_pre_hook(lambda mod,args,m=member:calls.append(m)))
    # Dataset/forward checks, plus encoder calls are retained as audit evidence.
    report=evaluate(model,dataset,runtime,device,torch.Generator().manual_seed(cfg['seed']),out,use_cav,
                    file_sha(Path(experiment)/'final_infer/assembled.pth'))
    for hook in hooks:
        hook.remove()
    used=set(calls)
    if not used.issubset(set(MODALITIES[:use_cav])) or 'm1' not in used:
        raise ValueError('Unexpected encoder participation: '+repr(used))
    write_json(out/'forward_encoders.json',{'calls':calls,'unique_members':sorted(used),
               'assembled_sha256':file_sha(Path(experiment)/'final_infer/assembled.pth'),
               'cav1_path':'native HomoCollab; frozen pretrain/m1 tensors retained in alliance/ft' if use_cav==1 else None})
    return report


def summarize(experiment):
    reports=[json.loads((Path(experiment)/f'final_infer/CAV{n}/result.json').read_text(encoding='utf-8')) for n in (1,2,3,4)]
    def identity(report):
        return [{k:v for k,v in f.items() if k not in ('members','requested','realized','input_cav_ids','forward_encoders')} for f in report['frames']]
    baseline=identity(reports[0])
    for n,report in enumerate(reports,1):
        if not baseline or report['requested']!=n or identity(report)!=baseline:
            raise ValueError('CAV1-4 frame/GT identity mismatch')
        if any(report.get(k)!=reports[0].get(k) for k in ('dataset','protocol_sha256','checkpoint_sha256')):
            raise ValueError('Mixed protocol results')
        for frame in report['frames']:
            assert_prefix(frame['members'],n)
    out=Path(experiment)/'final_infer'
    write_json(out/'AP_CAV1_4.json',[{'CAV':n,'AP':r['AP'],'frames':len(r['frames']),
                 'protocol_status':r['protocol_status']} for n,r in enumerate(reports,1)])
    with (out/'AP_CAV1_4.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['CAV','AP@0.3','AP@0.5','AP@0.7'])
        for n,r in enumerate(reports,1):
            w.writerow([n,*[format(r['AP'][f'AP@{iou:.1f}'],'.17g') for iou in (0.3,0.5,0.7)]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    q.add_argument('--dataset',choices=['opv2v','v2xset'],required=True)
    q.add_argument('--experiment',type=Path,required=True)
    for command in ('train','smoke'):
        q=sub.add_parser(command)
        q.add_argument('--experiment',type=Path,required=True)
        q.add_argument('--stage',choices=STAGES[:-1],required=True)
        if command=='train':
            q.add_argument('--resume',action='store_true')
    for cmd in ('assemble','summarize','infer'):
        q=sub.add_parser(cmd)
        q.add_argument('--experiment',type=Path,required=True)
        if cmd=='infer':
            q.add_argument('--use-cav',type=int,choices=[1,2,3,4],required=True)
    args=p.parse_args()
    # Resolve relative protocol paths from the checkout, not a caller's random cwd.
    os.chdir(ROOT)
    if args.command=='prepare':
        prepare(args.dataset,args.experiment)
    elif args.command=='train':
        train(args.experiment,args.stage,args.resume)
    elif args.command=='smoke':
        smoke(args.experiment,args.stage)
    elif args.command=='assemble':
        assemble(args.experiment)
    elif args.command=='infer':
        infer(args.experiment,args.use_cav)
    else:
        summarize(args.experiment)


if __name__=='__main__':
    main()
