"""Migration acceptance: real CPU optimizer/checkpoint logic; no perception training."""
import copy
import json
import contextlib
import io
import sys

import pytest
import torch

from opencood.tools import nego_state as state
from opencood.tools import run_nego_doma as runner
from opencood.tools.nego_protocol import ROOT, config_root, read_yaml, file_sha


class HomoCollab(torch.nn.Linear):
    pass


def example(tmp_path):
    cfg=read_yaml(config_root('v2xset')/'pretrain/m1/config.yaml')
    model=HomoCollab(2,2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.1)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,2,gamma=.5)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        scheduler.step()
    payload=state.pack(model,cfg,{}, {}, {},2,optimizer,scheduler,torch.Generator().manual_seed(303),.5,2)
    path=tmp_path/'last.pth'
    state.atomic_save(path,payload)
    return cfg,model,payload,path


def test_resume_accepts_current_hyperparameters_preserving_model_contract(tmp_path):
    cfg,model,payload,path=example(tmp_path)
    changed=copy.deepcopy(cfg)
    changed['train_params'].update(batch_size=8,num_workers=8,epoches=50)
    changed['optimizer']['lr']=.02
    changed['optimizer'].setdefault('args',{})['weight_decay']=.003
    changed['lr_scheduler']={'core_method':'step','step_size':3,'gamma':.1}
    restored=state.load_stage(path,cfg=changed,model=model,mode='resume',
                             protocol_sha=file_sha(ROOT/cfg['nego_protocol']['protocol_lock']))
    assert restored['meta']['config_sha']==payload['meta']['config_sha']
    optimizer=torch.optim.Adam(model.parameters(),lr=.02,weight_decay=.003)
    schedule=runner.restore_training_schedule(optimizer,changed,restored['resume'],5)
    assert optimizer.param_groups[0]['lr']==pytest.approx(.02)
    assert optimizer.param_groups[0]['weight_decay']==.003
    assert schedule.last_epoch==2 and schedule.step_size==3 and schedule.gamma==.1
    assert all(value['step'].item()==2 for value in optimizer.state.values())
    optimizer.zero_grad(); model(torch.ones(1,2)).sum().backward(); optimizer.step(); schedule.step()
    assert optimizer.param_groups[0]['lr']==pytest.approx(.002)
    changed['postprocess']['anchor_args']['cav_lidar_range'][0]-=1
    with pytest.raises(ValueError,match='configuration mismatch'):
        state.load_stage(path,cfg=changed,model=model,mode='resume',
                         protocol_sha=file_sha(ROOT/cfg['nego_protocol']['protocol_lock']))


def test_scheduler_type_and_current_batch_progress(tmp_path):
    cfg,model,payload,_=example(tmp_path)
    cfg['optimizer']['lr']=.03
    cfg['lr_scheduler']={'core_method':'multistep','step_size':[1,4],'gamma':.25}
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    schedule=runner.restore_training_schedule(optimizer,cfg,payload['resume'],5)
    assert isinstance(schedule,torch.optim.lr_scheduler.MultiStepLR)
    assert schedule.last_epoch==2 and optimizer.param_groups[0]['lr']==pytest.approx(.0075)
    cfg['lr_scheduler']={'core_method':'cosineannealwarm','epoches':10,'warmup_epoches':0,
                         'warmup_lr':0.,'lr_min':0.}
    optimizer=torch.optim.Adam(model.parameters(),lr=.03)
    schedule=runner.restore_training_schedule(optimizer,cfg,payload['resume'],5)
    assert schedule.t_initial==50
    assert optimizer.param_groups[0]['lr']==pytest.approx(schedule._get_lr(10)[0])


def test_changed_optimizer_uses_current_type_without_foreign_moments(tmp_path):
    cfg,model,payload,_=example(tmp_path)
    cfg['optimizer']={'core_method':'SGD','lr':.04,'args':{'momentum':.8}}
    cfg['lr_scheduler']={'core_method':'step','step_size':10,'gamma':.1}
    optimizer=torch.optim.SGD(model.parameters(),lr=.04,momentum=.8)
    runner.restore_training_schedule(optimizer,cfg,payload['resume'],2)
    assert not optimizer.state and optimizer.param_groups[0]['momentum']==.8
    assert optimizer.param_groups[0]['lr']==.04


def test_reviewed_tooling_migration_keeps_checkpoint_compatibility(tmp_path,monkeypatch):
    cfg,model,payload,path=example(tmp_path)
    payload['meta']['code_sha']='d825dfe341d0440787e97579b873b2d924a2958352fb9274a36a26b3a8ad21b4'
    state.atomic_save(path,payload)
    assert state.load_stage(path,cfg=cfg,model=model,mode='resume',
                            protocol_sha=file_sha(ROOT/cfg['nego_protocol']['protocol_lock']))
    monkeypatch.setattr(state,'model_implementation_signature',lambda:'unreviewed-model-change')
    with pytest.raises(ValueError,match='implementation hash'):
        state.load_stage(path,cfg=cfg,model=model,mode='resume',
                         protocol_sha=file_sha(ROOT/cfg['nego_protocol']['protocol_lock']))


def test_evaluation_and_summary_share_flat_final_directory(tmp_path,monkeypatch):
    from opencood.utils import eval_utils
    cfg=read_yaml(config_root('v2xset')/'final_infer/config.yaml')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder_m1=torch.nn.Identity()
        def forward(self,ego):
            return self.encoder_m1(torch.ones(1))
    class Data:
        protocol_identities=[{'scene':'scene','timestamp':'1'}]
        def __len__(self): return 1
        def post_process(self,batch,output): return torch.empty(0,7),torch.empty(0),torch.ones(1,7)
    batch={'ego':{'agent_modality_list':['m1'],'cav_id_list':['cav1']}}
    monkeypatch.setattr(runner,'loader',lambda *args,**kwargs:[copy.deepcopy(batch)])
    monkeypatch.setattr(eval_utils,'caluclate_tp_fp',lambda boxes,scores,gt,stats,iou:stats[iou].update(gt=1))
    monkeypatch.setattr(eval_utils,'calculate_ap',lambda stats,iou:(.1,))
    final=tmp_path/'final_infer'
    for cav in (1,2,3,4):
        runner.evaluate(Model(),Data(),cfg,torch.device('cpu'),torch.Generator(),final,cav,
                        'same-checkpoint',output_suffix='_CAV'+str(cav))
        paths=runner.inference_paths(tmp_path,cav)
        paths['encoders'].write_text(json.dumps({'calls':['m1']}))
        assert all(path.parent==final and path.is_file() for path in paths.values())
    runner.summarize(tmp_path)
    assert not any(path.is_dir() for path in final.iterdir())
    assert len(json.loads((final/'AP_CAV1_4.json').read_text()))==4


def test_controller_partial_acceptance_allows_hyperparameter_updates(tmp_path,monkeypatch):
    from test_nego_r2_orchestrator import namespace,oom_experiment,saved_pair
    root,exp,records,_=oom_experiment(tmp_path,monkeypatch,'pretrain/m4')
    saved_pair(exp,'pretrain/m4',23,22)
    namespace['main'](root,exp,True)
    cfgs=namespace['validate_configs'](exp)
    cfgs['pretrain/m4']['train_params'].update(batch_size=8,epoches=40,num_workers=8)
    cfgs['pretrain/m4']['optimizer']={'lr':.03}
    records.clear()
    namespace['main'](root,exp,True)
    assert records==[]
    assert json.loads((exp/'orchestrator_state.json').read_text())['stages']['pretrain/m4']['status']=='partial_accepted'


def test_controller_runtime_batch_can_exceed_yaml_then_halve(tmp_path,monkeypatch):
    from test_nego_r2_orchestrator import namespace,oom_experiment,install_oom_attempts
    root,exp,records,_=oom_experiment(tmp_path,monkeypatch,'pretrain/m1')
    monkeypatch.setenv('NEGO_RUNTIME_BATCH_SIZE','8')
    batches=install_oom_attempts(monkeypatch,exp,'pretrain/m1',records,[20,None])
    namespace['main'](root,exp,False)
    assert batches==[(8,False),(4,True)]


def test_controller_checkpoint_probe_accepts_changed_optimizer_and_scheduler(tmp_path,monkeypatch):
    from test_nego_r2_orchestrator import namespace
    from opencood.tools import nego_protocol
    cfg,model,payload,path=example(tmp_path)
    directory=tmp_path/'pretrain/m1'
    directory.mkdir(parents=True)
    state.atomic_save(directory/'last.pth',payload)
    changed=copy.deepcopy(cfg)
    changed['optimizer']={'core_method':'SGD','lr':.07,'args':{'momentum':.8}}
    changed['lr_scheduler']={'core_method':'multistep','step_size':[1,5],'gamma':.2}
    monkeypatch.setattr(nego_protocol,'read_yaml',lambda path:changed)
    monkeypatch.setattr(state,'model_for',lambda cfg:model)
    monkeypatch.setattr(runner,'parsed',lambda cfg:cfg)
    def execute_probe(command,**kwargs):
        monkeypatch.setattr(sys,'argv',['-c']+command[3:])
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(command[2],'<checkpoint-probe>','exec'),{})
        return output.getvalue()
    monkeypatch.setattr(namespace['subprocess'],'check_output',execute_probe)
    result=namespace['checkpoint_check'](ROOT,tmp_path,'pretrain/m1',{},'last.pth',True)
    assert result['epoch']==2


def test_shortened_epoch_budget_does_not_relabel_historical_m4_complete(tmp_path,monkeypatch):
    from test_nego_r2_orchestrator import namespace,oom_experiment,saved_pair
    root,exp,records,_=oom_experiment(tmp_path,monkeypatch,'pretrain/m4')
    saved_pair(exp,'pretrain/m4',23,22)
    namespace['validate_configs'](exp)['pretrain/m4']['train_params']['epoches']=20
    namespace['main'](root,exp,True)
    entry=json.loads((exp/'orchestrator_state.json').read_text())['stages']['pretrain/m4']
    assert entry['status']=='partial_accepted' and entry['completed_epoch']==23
    assert not any(stage=='pretrain/m4' for stage,_ in records)
