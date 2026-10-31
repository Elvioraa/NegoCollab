"""Isolated orchestration tests: no dataset, CUDA, or training subprocesses."""
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_nego_v2xset_doma_r2.sh'
source = SCRIPT.read_text(encoding='utf-8')
python_source = source.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
namespace = {'__name__': 'orchestrator_test'}
exec(compile(python_source, str(SCRIPT), 'exec'), namespace)


def test_python38_and_bash_syntax():
    def parse(text):
        if sys.version_info[:2] > (3, 8):
            return ast.parse(text, feature_version=(3, 8))
        return ast.parse(text)
    tree = parse(python_source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('probe', 'code') for t in node.targets)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            parse(node.value.value)
    bash = os.environ.get('NEGO_TEST_BASH', 'bash')
    subprocess.run([bash, '-n', str(SCRIPT)], check=True)


def test_train_start_and_reject_logs_or_checkpoints(tmp_path):
    stage = 'pretrain/m1'
    directory = tmp_path / stage
    directory.mkdir(parents=True)
    (directory / 'config.yaml').write_text('config')
    decision = namespace['train_decision']
    assert decision(tmp_path, stage, 31, False, None) == 'start'
    (directory / 'train.log').write_text('old')
    with pytest.raises(FileExistsError):
        decision(tmp_path, stage, 31, False, None)
    with pytest.raises(ValueError, match='without last'):
        decision(tmp_path, stage, 31, True, None)


def test_resume_skip_and_incomplete_checkpoints(tmp_path):
    stage = 'pretrain/m1'
    directory = tmp_path / stage
    directory.mkdir(parents=True)
    (directory / 'last.pth').write_bytes(b'last')
    check = lambda stage, name, full: {'epoch': 3 if name == 'last.pth' else 2}
    decision = namespace['train_decision']
    assert decision(tmp_path, stage, 4, True, check) == 'resume'
    with pytest.raises(ValueError, match='lacks best'):
        decision(tmp_path, stage, 3, True, check)
    (directory / 'best.pth').write_bytes(b'best')
    assert decision(tmp_path, stage, 3, True, check) == 'skip'
    with pytest.raises(ValueError, match='Invalid completed epoch'):
        decision(tmp_path, stage, 2, True, check)
    (directory / 'last.pth.tmp').write_bytes(b'partial')
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        decision(tmp_path, stage, 3, True, check)


def test_resume_does_not_ignore_strict_loader_failure(tmp_path):
    directory = tmp_path / 'pretrain/m1'
    directory.mkdir(parents=True)
    (directory / 'last.pth').write_bytes(b'corrupt')
    def reject(stage, name, full):
        raise ValueError('Checkpoint SHA256 mismatch/missing sidecar')
    with pytest.raises(ValueError, match='SHA256'):
        namespace['train_decision'](tmp_path, 'pretrain/m1', 31, True, reject)


def test_independent_audited_configuration_manifest(tmp_path, monkeypatch):
    configs = {}
    for stage, (batch, epochs, lr, prefix) in namespace['EXPECTED'].items():
        configs[stage] = {'seed': 303, 'train_params': {'batch_size': batch, 'epoches': epochs},
                          'optimizer': {'lr': lr}, 'fake_sha': prefix + '0' * 52,
                          'nego_protocol': {'dataset': 'v2xset', 'stage': stage,
                                            'parents': namespace['parents'](stage)},
                          'nego': {'multi_batch': {'enabled': stage.startswith('pretrain/'),
                                                   'mode': 'vectorized' if stage.startswith('pretrain/') else 'native'}}}
    monkeypatch.setitem(namespace, 'read_yaml', lambda path: configs[Path(path).parent.relative_to(tmp_path).as_posix()])
    monkeypatch.setitem(namespace, 'digest', lambda cfg: cfg['fake_sha'])
    monkeypatch.setitem(namespace, 'verify_lock', lambda dataset: None)
    monkeypatch.setitem(namespace, 'validate_config', lambda cfg: None)
    monkeypatch.setitem(namespace, 'validate_runtime_batch', lambda cfg: None)
    assert len(namespace['validate_configs'](tmp_path)) == 11
    configs['pretrain/m3']['train_params']['batch_size'] = 4
    with pytest.raises(ValueError, match='pretrain/m3'):
        namespace['validate_configs'](tmp_path)
    configs['pretrain/m3']['train_params']['batch_size'] = 8
    configs['pretrain/m3']['fake_sha'] = 'wrong'
    with pytest.raises(ValueError, match='pretrain/m3'):
        namespace['validate_configs'](tmp_path)


def fake_experiment(tmp_path, monkeypatch, fail=None):
    root = tmp_path / 'root'
    exp = root / 'exp'
    exp.mkdir(parents=True)
    (root / 'lock.yaml').write_text('lock')
    expected_sha = 'a' * 40
    monkeypatch.setenv('NEGO_EXPECTED_GIT_SHA', expected_sha)
    configs = {}
    for stage in namespace['EXPECTED']:
        (exp / stage).mkdir(parents=True)
        (exp / stage / 'config.yaml').write_text('config')
        configs[stage] = {'stage': stage, 'nego_protocol': {'protocol_lock': 'lock.yaml'}}
    monkeypatch.setitem(namespace, 'preflight', lambda root, exp: expected_sha)
    monkeypatch.setitem(namespace, 'validate_configs', lambda exp: configs)
    records = []
    def fake_check(root, exp, stage, configs, name, full=False):
        assert (exp / stage / name).is_file()
        provenance = {p: {'sha256': namespace['file_sha'](exp / p / 'best.pth'),
                          'config_sha': namespace['digest'](configs[p])}
                      for p in namespace['parents'](stage)}
        return {'epoch': namespace['EXPECTED'][stage][1] if stage != 'final_infer' else 0,
                'parents': provenance, 'owners': {}}
    monkeypatch.setitem(namespace, 'checkpoint_check', fake_check)
    monkeypatch.setitem(namespace, 'check_result', lambda exp, cav, cfg, cksha, protocol_sha: {'cav': cav})
    monkeypatch.setitem(namespace, 'check_summary', lambda exp, results: None)
    def fake_run(command, cwd, stdout, stderr):
        args = command[4:]
        action = args[0]
        stage = args[args.index('--stage') + 1] if '--stage' in args else action
        if action == 'infer': stage = 'infer_CAV' + args[args.index('--use-cav') + 1]
        records.append((stage, '--resume' in args))
        if stage == fail:
            return SimpleNamespace(returncode=17)
        if action == 'train':
            for name in ('last.pth', 'best.pth'):
                (exp / stage / name).write_bytes(b'checkpoint')
                (exp / stage / (name + '.sha256')).write_text('checksum')
        elif action == 'assemble':
            path = exp / 'final_infer/assembled.pth'
            path.write_bytes(b'assembled')
            Path(str(path) + '.sha256').write_text('checksum')
            provenance = fake_check(root, exp, 'final_infer', configs, 'assembled.pth')['parents']
            path.with_suffix('.sources.json').write_text(json.dumps({'parents': provenance, 'owners': {},
                                                                    'checkpoint_sha256': namespace['file_sha'](path)}))
        elif action == 'infer':
            directory = exp / 'final_infer' / ('CAV' + args[args.index('--use-cav') + 1])
            directory.mkdir()
            (directory / 'result.json').write_text('{}')
            (directory / 'forward_encoders.json').write_text('{}')
            (directory / 'ap.csv').write_text('csv')
        elif action == 'summarize':
            (exp / 'final_infer/AP_CAV1_4.json').write_text('[]')
            (exp / 'final_infer/AP_CAV1_4.csv').write_text('csv')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(namespace['subprocess'], 'run', fake_run)
    return root, exp, records


def test_serial_order_and_resume_completed_skip(tmp_path, monkeypatch):
    root, exp, records = fake_experiment(tmp_path, monkeypatch)
    namespace['main'](root, exp, False)
    assert [stage for stage, _ in records] == namespace['TRAIN'] + [
        'assemble', 'infer_CAV1', 'infer_CAV2', 'infer_CAV3', 'infer_CAV4', 'summarize']
    state = json.loads((exp / 'orchestrator_state.json').read_text())
    assert state['status'] == 'complete'
    with pytest.raises(FileExistsError):
        namespace['main'](root, exp, False)
    records.clear()
    namespace['main'](root, exp, True)
    assert records == []


def test_failure_stops_later_stages(tmp_path, monkeypatch):
    root, exp, records = fake_experiment(tmp_path, monkeypatch, fail='pretrain/m2')
    with pytest.raises(RuntimeError, match='exit 17'):
        namespace['main'](root, exp, False)
    assert [stage for stage, _ in records] == ['pretrain/m1', 'pretrain/m2']
    state = json.loads((exp / 'orchestrator_state.json').read_text())
    assert state['stages']['pretrain/m2']['exit_code'] == 17


def test_incomplete_stage_calls_official_resume(tmp_path, monkeypatch):
    root, exp, records = fake_experiment(tmp_path, monkeypatch)
    directory = exp / 'pretrain/m1'
    (directory / 'last.pth').write_bytes(b'checkpoint')
    original = namespace['train_decision']
    calls = []
    def decision(exp, stage, epochs, resume, check):
        if stage == 'pretrain/m1' and len(calls) < 2:
            calls.append(1)
            return 'resume'
        return original(exp, stage, epochs, resume, check)
    monkeypatch.setitem(namespace, 'train_decision', decision)
    namespace['main'](root, exp, True)
    assert records[0] == ('pretrain/m1', True)


def test_verified_completed_stage_is_skipped(tmp_path, monkeypatch):
    root, exp, records = fake_experiment(tmp_path, monkeypatch)
    for name in ('last.pth', 'best.pth'):
        (exp / 'pretrain/m1' / name).write_bytes(b'checkpoint')
        (exp / 'pretrain/m1' / (name + '.sha256')).write_text('checksum')
    namespace['main'](root, exp, True)
    assert records[0] == ('pretrain/m2', False)
    state = json.loads((exp / 'orchestrator_state.json').read_text())
    assert state['stages']['pretrain/m1']['disposition'] == 'verified_existing'


def test_later_incomplete_stage_stops_before_first_train(tmp_path, monkeypatch):
    root, exp, records = fake_experiment(tmp_path, monkeypatch)
    (exp / 'pretrain/m4/train.log').write_text('old incomplete run')
    with pytest.raises(ValueError, match='without last'):
        namespace['main'](root, exp, True)
    assert records == []


def test_lock_conflict_stops_before_python(tmp_path):
    root = tmp_path / 'repo'
    scripts = root / 'scripts'
    scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
    exp = root / 'opencood/logs/NEGO_V2XSET_DOMA_R2_1GPU'
    exp.mkdir(parents=True)
    (exp / 'runtime.env').write_text('NEGO_EXPECTED_GIT_SHA=x\nCUDA_VISIBLE_DEVICES=2\n')
    bin_dir = root / 'bin'
    bin_dir.mkdir()
    flock = bin_dir / 'flock'
    flock.write_text('#!/usr/bin/env bash\nexit 1\n')
    flock.chmod(0o755)
    env = os.environ.copy()
    env['PATH'] = str(bin_dir) + os.pathsep + env['PATH']
    bash = env.get('NEGO_TEST_BASH', 'bash')
    result = subprocess.run([bash, str(scripts / SCRIPT.name)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Experiment lock already held' in result.stderr
    assert not (exp / 'orchestrator_state.json').exists()


def test_partial_inference_is_rejected(tmp_path, monkeypatch):
    root, exp, records = fake_experiment(tmp_path, monkeypatch)
    directory = exp / 'final_infer/CAV1'
    directory.mkdir()
    (directory / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='partial or unverified'):
        namespace['main'](root, exp, False)
    assert records == []


def test_inference_result_checks_encoder_and_provenance(tmp_path):
    directory = tmp_path / 'final_infer/CAV2'
    directory.mkdir(parents=True)
    cfg = {'x': 1}
    ap = {'AP@0.3': .3, 'AP@0.5': .2, 'AP@0.7': .1}
    report = {'dataset': 'v2xset', 'stage': 'final_infer', 'requested': 2,
              'config_sha256': namespace['digest'](cfg), 'protocol_sha256': 'p',
              'checkpoint_sha256': 'c', 'protocol_status': 'SERVER_VALIDATION_REQUIRED',
              'frames': [{'members': ['m1', 'm2'], 'forward_encoders': ['m2', 'm1'],
                          'sample_idx': 0, 'requested': 2, 'realized': 2, 'input_cav_ids': ['a', 'b'],
                          'in_range_cav_ids': ['a', 'b'], 'scene': 's', 'timestamp': 't',
                          'gt_count': 1, 'gt_sha256': 'a' * 64}],
              'statistics': {key: {'gt': 1} for key in ('0.3', '0.5', '0.7')}, 'AP': ap}
    (directory / 'result.json').write_text(json.dumps(report))
    encoder = {'assembled_sha256': 'c', 'calls': ['m2', 'm1'], 'unique_members': ['m1', 'm2']}
    (directory / 'forward_encoders.json').write_text(json.dumps(encoder))
    with (directory / 'ap.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['CAV', 'AP@0.3', 'AP@0.5', 'AP@0.7'])
        writer.writerow([2, .3, .2, .1])
    namespace['check_result'](tmp_path, 2, cfg, 'c', 'p')
    encoder['calls'].append('m4')
    (directory / 'forward_encoders.json').write_text(json.dumps(encoder))
    with pytest.raises(ValueError, match='Encoder participation'):
        namespace['check_result'](tmp_path, 2, cfg, 'c', 'p')
    report['AP']['AP@0.5'] = 1.1
    (directory / 'result.json').write_text(json.dumps(report))
    (directory / 'forward_encoders.json').write_text(json.dumps(dict(encoder, calls=['m2', 'm1'])))
    with pytest.raises(ValueError, match='Invalid AP'):
        namespace['check_result'](tmp_path, 2, cfg, 'c', 'p')


def test_summary_checks_ap_identity_and_ranges(tmp_path):
    final = tmp_path / 'final_infer'
    final.mkdir()
    ap = {'AP@0.3': .3, 'AP@0.5': .2, 'AP@0.7': .1}
    results = [{'dataset': 'v2xset', 'protocol_sha256': 'p', 'checkpoint_sha256': 'c',
                'protocol_status': 'SERVER_VALIDATION_REQUIRED', 'AP': dict(ap),
                'frames': [{'scene': 's', 'gt_sha256': 'g', 'members': ['m1']}]} for _ in range(4)]
    entries = [{'CAV': i, 'frames': 1, 'protocol_status': 'SERVER_VALIDATION_REQUIRED', 'AP': dict(ap)}
               for i in range(1, 5)]
    (final / 'AP_CAV1_4.json').write_text(json.dumps(entries))
    with (final / 'AP_CAV1_4.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['CAV', 'AP@0.3', 'AP@0.5', 'AP@0.7'])
        writer.writerows([[i, .3, .2, .1] for i in range(1, 5)])
    namespace['check_summary'](tmp_path, results)
    results[3]['frames'][0]['gt_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='frame'):
        namespace['check_summary'](tmp_path, results)
