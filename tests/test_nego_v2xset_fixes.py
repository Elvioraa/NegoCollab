"""Cross-platform V2XSet protocol and safe metadata regressions."""
import hashlib
from pathlib import Path
import pytest
import yaml
from opencood.tools.nego_protocol import (read_yaml, text_protocol_sha, train_scene_exclusions, lidar_path, config_root, verify_lock)

# NumPy scalar representation from V2XSet train/2021_08_23_16_42_39/10354/000085.yaml.
METADATA = """lidar_pose: [29.728961944580078, 128.037353515625, 1.932119607925415, -0.1858520358800888, -133.30355834960938, 0.10293777287006378]
camera0:
  intrinsic: [[335.639852470912, 0.0, 400.0], [0.0, 335.639852470912, 300.0], [0.0, 0.0, 1.0]]
plan_trajectory:
- - 26.939550399780273
  - 124.68225860595703
  - !!python/object/apply:numpy.core.multiarray.scalar
    - &id001 !!python/object/apply:numpy.dtype
      args: [f8, false, true]
      state: !!python/tuple [3, '<', null, null, null, -1, -1, 0]
    - !!binary |
      Xg1PQMa6TUA=
"""

@pytest.mark.parametrize('name,base', [('config.yaml',b'a: 1\nb: 2\n'),('assignment.json',b'{\"a\":1}\n')])
def test_hash_normalizes_lf_crlf_cr(tmp_path,name,base):
    values=[]
    for i,b in enumerate((base,base.replace(b'\n',b'\r\n'),base.replace(b'\n',b'\r'))):
        p=tmp_path/(str(i)+name);p.write_bytes(b);values.append(text_protocol_sha(p))
    assert len(set(values))==1
    assert values[0]==hashlib.sha256(base).hexdigest()


def test_restricted_numpy_yaml(tmp_path):
    p=tmp_path/'sample.yaml';p.write_text(METADATA,encoding='utf8');v=read_yaml(p)
    assert v['lidar_pose']==[29.728961944580078,128.037353515625,1.932119607925415,-0.1858520358800888,-133.30355834960938,0.10293777287006378]
    assert v['camera0']['intrinsic'][0]==[335.639852470912,0.,400.]
    assert v['plan_trajectory'][0][2]==pytest.approx(59.45917514667484)
    p.write_text('x: !!python/object/apply:os.system [echo unsafe]',encoding='utf8')
    with pytest.raises(yaml.constructor.ConstructorError):read_yaml(p)
    p.write_text('x: [1, 2]\ny: {a: 3}\n',encoding='utf8')
    assert read_yaml(p)=={'x':[1,2],'y':{'a':3}}


@pytest.mark.parametrize('stage,excluded', [('pretrain/m1',False),('pretrain/m2',True),('pretrain/m3',False),('pretrain/m4',True),('alliance/nego',True),('m3/align',True),('m4/ft',True),('final_infer',False)])
def test_stage_filtered_only_for_training(stage,excluded):
    cfg=read_yaml(config_root('v2xset')/stage/'config.yaml')
    assert bool(train_scene_exclusions(cfg,True))==excluded
    assert train_scene_exclusions(cfg,False)==set()
    cfg['nego_protocol']['scene_exclusions']=['unapproved']
    with pytest.raises(ValueError):train_scene_exclusions(cfg,True)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):lidar_path(tmp_path/'missing.pcd','v2xset','m3')


def test_both_locks():
    for ds in ('opv2v','v2xset'):
        assert verify_lock(ds)['protocol_text_hash']=='sha256_lf_normalized'


def test_unregistered_missing_camera_rejected(tmp_path, monkeypatch):
    import json
    from opencood.tools import check_nego_doma as checker
    rel='dataset/V2XSET/'
    assignments={'train_scene':{'1':'m1'},'val_scene':{'1':'m1'}}
    test_assignment={'test_scene':{'1':'m1'}}
    (tmp_path/'assignment.json').write_text(json.dumps(assignments))
    (tmp_path/'test_assignment.json').write_text(json.dumps(test_assignment))
    lock={'assignments':{'training':{'path':'assignment.json'},'evaluation':{'path':'test_assignment.json'}},
          'splits':{'root_dir':rel+'train','validate_dir':rel+'validate','test_dir':rel+'test'},
          'excluded_frames':[],'training':{'max_cav':5}}
    for split,scene in [('train','train_scene'),('validate','val_scene'),('test','test_scene')]:
        folder=tmp_path/rel/split/scene/'1';folder.mkdir(parents=True)
        metadata={'lidar_pose':[0,0,0,0,0,0]}
        metadata.update({'camera'+str(i):{} for i in range(4)})
        (folder/'0001.yaml').write_text(yaml.safe_dump(metadata))
        (folder/'0001.pcd').write_bytes(b'FIELDS x y z intensity\nDATA ascii\n')
        for i in range(3):(folder/('0001_camera'+str(i)+'.png')).write_bytes(b'fixture')
    config=tmp_path/'cfg/final_infer/config.yaml';config.parent.mkdir(parents=True);config.write_text('{}')
    monkeypatch.setattr(checker,'ROOT',tmp_path)
    monkeypatch.setattr(checker,'verify_lock',lambda ds:lock)
    monkeypatch.setattr(checker,'config_root',lambda ds:tmp_path/'cfg')
    with pytest.raises(FileNotFoundError,match='camera3'):
        checker.audit_data('v2xset',1)


def test_gitignore_matches_dataset_symlink(tmp_path):
    import os,subprocess
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'.gitignore').write_text('/dataset\n',encoding='utf8')
    target=tmp_path/'external';target.mkdir()
    try:os.symlink(str(target),str(tmp_path/'dataset'),target_is_directory=True)
    except OSError:pytest.skip('Directory symlink unavailable on this host')
    p=subprocess.run(['git','-C',str(tmp_path),'check-ignore','-v','dataset'],capture_output=True,text=True)
    assert p.returncode==0 and '/dataset' in p.stdout
