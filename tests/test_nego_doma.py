"""CPU contracts plus real native Negotiator/ComminPub. No perception mocks in core tests."""
import ast
import copy
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from opencood.tools.nego_protocol import (ROOT, MODALITIES, STAGES, ProtocolAdaptor, active_members,
    assert_prefix, config_root, file_sha, lidar_path, parents, prepare, read_yaml, seed_all, validate_config, write_json)
from opencood.tools.nego_state import (atomic_save, cpu_state, initialize, load_stage, pack,
    restore_rng, rng_state, set_training, source_for, strict_load, trainable_key)
from opencood.tools.run_nego_doma import stats_template, summarize, check_process
from opencood.utils.nego_pcd import read_pcd

torch.set_num_threads(2)


def config(stage='alliance/nego',dataset='opv2v'):
    return read_yaml(config_root(dataset)/stage/'config.yaml')


def test_all_yaml_and_source_contracts():
    from opencood.tools.check_nego_doma import static_check, audit_sources
    for ds in ('opv2v','v2xset'):
        result=static_check(ds)
        assert result['status']=='PASS'
        assert len(result['stage_configs'])==11
    assert audit_sources()['legacy_variants_used'] is False


def test_modality_identity_rejects_wrong_camera_and_beams():
    cfg=config('final_infer')
    cfg['model']['args']['m4']['encoder_args']['camera_encoder']='EfficientNet'
    with pytest.raises(ValueError,match='camera backbone'):
        validate_config(cfg)
    cfg=config('final_infer','v2xset')
    cfg['heter']['lidar_channels_dict']={'m3':32}
    with pytest.raises(ValueError,match='beam policy'):
        validate_config(cfg)


def test_pcd_rules_are_strict_and_cross_platform(tmp_path):
    assert lidar_path('dataset/OPV2V/test/s/c/1.pcd','opv2v','m3',False)=='dataset/OPV2V_Hetero/test/s/c/1_32.pcd'
    assert lidar_path(r'dataset\V2XSET\test\s\c\1.pcd','v2xset','m3',False)=='dataset/V2XSET/test/s/c/1.pcd'
    with pytest.raises(FileNotFoundError,match='_32.pcd'):
        lidar_path(tmp_path/'OPV2V/test/s/c/missing.pcd','opv2v','m3')
    path=tmp_path/'original.pcd'
    path.write_text('VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n1 2 3 42\n4 5 6 7\n')
    np.testing.assert_array_equal(read_pcd(path),np.array([[1,2,3,42],[4,5,6,7]],dtype=np.float32))
    path.write_text(path.read_text().replace('intensity','unused'))
    with pytest.raises(ValueError,match='synthetic intensity'):
        read_pcd(path)


def test_pcd_binary_rgb_matches_native_red_channel(tmp_path):
    path=tmp_path/'rgb.pcd'
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')])
    a=np.array([(1,2,3,0xFF1234)],dtype=dtype)
    header=b'FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n'
    path.write_bytes(header+a.tobytes())
    np.testing.assert_array_equal(read_pcd(path),np.array([[1,2,3,1]],dtype=np.float32))


def adaptor(train=False):
    return ProtocolAdaptor('m1',list(MODALITIES),{'s':{'7':'m1','2':'m2','3':'m3','-1':'m4'}},
                           {},dict(zip(MODALITIES,MODALITIES)),None,train,'v2xset')


def test_assignment_order_seed_and_cav_prefix():
    a=adaptor()
    assert a.reorder_cav_list(['-1','2','3','7'],'s')==['7','2','3','-1']
    for n in (1,2,3,4):
        assert assert_prefix(list(MODALITIES),n)==list(MODALITIES[:n])
        assert assert_prefix(['m1','m2'],n)==list(MODALITIES[:min(n,2)])
    with pytest.raises(ValueError,match='Non-contiguous'):
        assert_prefix(['m1','m3'],2)
    with pytest.raises(ValueError,match='mismatch'):
        a.reorder_cav_list(['7','2','3'],'s')
    def sample():
        seed_all(303)
        a=adaptor(True)
        cavs=a.reorder_cav_list(['7','2','3','-1'],'s')
        return cavs,[a.reassign_cav_modality('m1',i) for i in range(4)],random.random(),np.random.rand(),torch.rand(1)
    x,y=sample(),sample()
    assert x[:4]==y[:4]
    assert torch.equal(x[4],y[4])


def test_dag_anchors_new_members_to_same_initial_negotiation():
    for st in STAGES:
        assert all(STAGES.index(p)<STAGES.index(st) for p in parents(st))
    assert parents('m3/align')[:2]==parents('m4/align')[:2]==['alliance/nego','alliance/ft']
    assert source_for('negotiator.weight','m4/align')=='alliance/nego'
    assert source_for('comm_m3.receiver.weight','final_infer')=='m3/ft'
    assert source_for('comm_m4.receiver.weight','final_infer')=='m4/ft'


def fixture_model(cfg):
    """Small state-IO fixture, not a substitute for the real core test below."""
    model=nn.Module()
    stage=cfg['nego_protocol']['stage']
    for m in active_members(stage):
        for prefix in ('encoder','backbone','cls_head','reg_head','dir_head','fusion_net'):
            model.add_module(prefix+'_'+m,nn.Linear(2,2))
        if not stage.startswith('pretrain/'):
            comm=nn.Module()
            comm.add_module('sender',nn.Linear(2,2))
            comm.add_module('receiver',nn.Linear(2,2))
            model.add_module('comm_'+m,comm)
    if stage.endswith(('/nego','/align')):
        model.add_module('negotiator',nn.Linear(2,2))
        model.add_module('occ_head_nego',nn.Linear(2,1))
        for m in ('m1','m2'):
            model.add_module('nego_reziser_'+m,nn.Linear(2,2))
    return model


def make_fixture_chain(experiment):
    prepare('opv2v',experiment)
    states={}
    for stage in STAGES:
        cfg=read_yaml(experiment/stage/'config.yaml')
        model=fixture_model(cfg)
        owners,provenance,common=initialize(model,cfg,experiment)
        for k,p in model.named_parameters():
            if trainable_key(k,cfg):
                p.data.add_(0.123)
        ckpt=pack(model,cfg,owners,provenance,common,1)
        atomic_save(experiment/stage/('assembled.pth' if stage=='final_infer' else 'best.pth'),ckpt)
        states[stage]=ckpt
    return states


def test_complete_checkpoint_dag_strict_ownership_and_wrong_member_rejected(tmp_path):
    states=make_fixture_chain(tmp_path)
    final=states['final_infer']
    assert final['common_state']
    for k,v in final['model'].items():
        source=source_for(k,'final_infer')
        assert torch.equal(v,states[source]['model'][k])
        if k.startswith(('encoder_m1','fusion_net_m1','cls_head_m1','backbone_m1')):
            assert torch.equal(v,states['pretrain/m1']['model'][k])
    with pytest.raises(ValueError,match='stage mismatch'):
        load_stage(tmp_path/'m3/ft/best.pth',dataset='opv2v',stage='m4/ft')
    with pytest.raises(ValueError,match='dataset mismatch'):
        load_stage(tmp_path/'m3/ft/best.pth',dataset='v2xset')
    model=fixture_model(config('final_infer'))
    bad=dict(final['model'])
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError,match='key mismatch'):
        strict_load(model,bad)
    bad=dict(final['model']);k=next(iter(bad));bad[k]=torch.zeros(1)
    with pytest.raises(ValueError,match='shape/dtype'):
        strict_load(model,bad)
    path=tmp_path/'m3/ft/best.pth'
    with path.open('ab') as stream:
        stream.write(b'changed after export')
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        load_stage(path,dataset='opv2v',stage='m3/ft')


def test_missing_parent_and_prepare_never_overwrites(tmp_path):
    prepare('opv2v',tmp_path)
    prepare('opv2v',tmp_path)
    with pytest.raises(FileNotFoundError,match='required stage'):
        initialize(fixture_model(config()),config(),tmp_path)
    path=tmp_path/'pretrain/m1/config.yaml'
    path.write_text('user edited config')
    with pytest.raises(FileExistsError,match='overwrite'):
        prepare('opv2v',tmp_path)
    assert path.read_text()=='user edited config'


def test_full_resume_restores_next_optimizer_update_and_rng(tmp_path):
    cfg=config('pretrain/m1')
    seed_all(303)
    generator=torch.Generator().manual_seed(303)
    model=nn.Linear(3,2)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,[1,3],gamma=.5)
    def update():
        x=torch.randn(4,3)+np.random.rand()+random.random()+torch.rand(1,generator=generator)
        opt.zero_grad();loss=model(x).square().mean();loss.backward();opt.step();sched.step()
        return float(loss.detach())
    update()
    checkpoint=pack(model,cfg,{}, {},{},1,opt,sched,generator,0.1)
    atomic_save(tmp_path/'resume.pth',checkpoint)
    expected_loss=update();expected=cpu_state(model)
    checkpoint=load_stage(tmp_path/'resume.pth',cfg=cfg)
    strict_load(model,checkpoint['model'])
    opt.load_state_dict(checkpoint['resume']['optimizer'])
    sched.load_state_dict(checkpoint['resume']['scheduler'])
    restore_rng(checkpoint['resume']['rng'],generator)
    assert update()==expected_loss
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items())


def native_core():
    from opencood.models.comm_modules.comm_in_pub import ComminPub
    from opencood.models.sub_modules.keyfeat_modules import Negotiator
    cfg=config()
    model=nn.Module()
    for m in ('m1','m2','m3','m4'):
        # All four native communication modules are the same configured algorithm;
        # synthetic BEV replaces only missing sensor data, not Sender/Receiver.
        args=copy.deepcopy(config('final_infer')['model']['args'][m]['comm_args'])
        args.update(local_range=[-6.4,-6.4,-3,6.4,6.4,1],modality=m)
        model.add_module('comm_'+m,ComminPub(args))
    model.add_module('negotiator',Negotiator(cfg['model']['args']['pub']['negotiator']))
    return model


def test_real_negotiator_sender_receiver_backward_and_reproducibility():
    def run():
        seed_all(303)
        model=native_core()
        set_training(model,config())
        x=torch.randn(2,64,16,16,requires_grad=True)
        y=torch.randn_like(x,requires_grad=True)
        common=model.negotiator([x,y])
        affine=torch.zeros(1,2,2,2,3);affine[...,0,0]=1;affine[...,1,1]=1
        loss=0
        for m,feature in (('m1',x),('m2',y)):
            comm=getattr(model,'comm_'+m)
            sent=comm.send(feature)
            received=comm.receive(common,torch.tensor([2]),affine)
            assert sent.shape==received.shape==common.shape==(2,64,16,16)
            loss=loss+(sent-common).square().mean()+(received-feature).square().mean()
        loss.backward()
        assert x.grad.abs().sum()>0 and y.grad.abs().sum()>0
        for name in ('comm_m1.sender_aligner','comm_m1.receiver_converter','negotiator'):
            assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(name))
        return float(loss.detach()),common.detach()
    a,b=run(),run()
    assert a[0]==b[0] and torch.equal(a[1],b[1])


def test_real_receiver_inference_grouping_for_cav2_through_4():
    seed_all(303)
    model=native_core().eval()
    for m in MODALITIES:
        getattr(model,'comm_'+m).inference=True
    affine=torch.zeros(1,5,5,2,3);affine[...,0,0]=1;affine[...,1,1]=1
    for n in (2,3,4):
        messages=[]
        for m in MODALITIES[:n]:
            messages.append(getattr(model,'comm_'+m).send(torch.randn(1,64,16,16)))
        received=model.comm_m1.receive(torch.cat(messages),torch.tensor([n]),affine,torch.tensor([1]))
        assert received.shape==(n,64,16,16)
        assert torch.isfinite(received).all()


@pytest.mark.parametrize('stage',['m3/align','m4/align','alliance/ft','m3/ft','m4/ft'])
def test_real_core_trainability_and_frozen_buffers(stage):
    seed_all(303)
    model=native_core()
    cfg=config(stage)
    selected=set_training(model,cfg)
    assert selected
    frozen={k:v for k,v in cpu_state(model).items() if not trainable_key(k,cfg)}
    new=cfg['nego_protocol']['new_members'][0]
    affine=torch.zeros(1,2,2,2,3);affine[...,0,0]=1;affine[...,1,1]=1
    common=model.negotiator([torch.randn(2,64,16,16),torch.randn(2,64,16,16)])
    comm=getattr(model,'comm_'+new)
    sent=comm.send(torch.randn(2,64,16,16))
    out=comm.receive(common*.5+sent*.5,torch.tensor([2]),affine)
    out.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n in selected)
    assert all(p.grad is None for n,p in model.named_parameters() if n not in selected)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items())
    if stage.endswith('/ft'):
        assert all('receiver' in n or 'local_prompt_generator' in n or 'unify2local' in n for n in selected)
    else:
        assert all(n.startswith('comm_'+new+'.') for n in selected)


def test_real_native_tp_fp_and_full_precision_ap():
    from opencood.utils import eval_utils
    # Two matched GTs and a false positive scored between them gives AP 5/6.
    a=torch.tensor([[0.,0.],[0,2],[2,2],[2,0]])
    b=a+4
    false=a+10
    boxes=torch.stack([a,false,b])
    gt=torch.stack([a,b])
    stats=stats_template()
    for iou in stats:
        eval_utils.caluclate_tp_fp(boxes,torch.tensor([.9,.8,.7]),gt,stats,iou)
        assert stats[iou]['tp']==[1,0,1]
        assert stats[iou]['fp']==[0,1,0]
        assert stats[iou]['gt']==2
        assert eval_utils.calculate_ap(stats,iou)[0]==pytest.approx(5/6)


def test_native_algorithm_forward_methods_unchanged_and_roles():
    files=['homo_collab','heter_nego_train','heter_nego_train_align','heter_nego_inf']
    for name in files:
        path='opencood/models/'+name+'.py'
        original=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT).decode('utf-8')
        current=(ROOT/path).read_text(encoding='utf-8')
        def forward(source):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='forward']
        assert forward(original)==forward(current)
    source=(ROOT/'opencood/models/heter_nego_inf.py').read_text(encoding='utf-8')
    method=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr.startswith('negotiator')
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
        for node in ast.walk(method)
    )


def test_summary_rejects_different_gt_or_frames(tmp_path):
    for n in (1,2,3,4):
        write_json(tmp_path/f'final_infer/CAV{n}/result.json',{
            'dataset':'opv2v','protocol_sha256':'fixed','requested':n,
            'protocol_status':'SERVER_VALIDATION_REQUIRED','AP':{'AP@0.3':.1,'AP@0.5':.2,'AP@0.7':.3},
            'frames':[{'scene':'s','timestamp':'1','sample_idx':0,'in_range_cav_ids':['a','b','c','d'],
                       'input_cav_ids':['a','b','c','d'][:n],'members':list(MODALITIES[:n]),
                       'requested':n,'realized':n,'gt_count':1,'gt_sha256':'same'}]})
    summarize(tmp_path)
    path=tmp_path/'final_infer/CAV3/result.json'
    value=json.loads(path.read_text());value['frames'][0]['gt_sha256']='different';write_json(path,value)
    with pytest.raises(ValueError,match='frame/GT'):
        summarize(tmp_path)


def test_ddp_explicitly_rejected(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError,match='DDP'):
        check_process()
