"""Regression tests for public-clone native integrity audit."""
import json
import shutil
from unittest.mock import patch
import pytest
from opencood.tools import check_nego_doma as checker

@pytest.fixture
def tree(tmp_path, monkeypatch):
    source = checker.ROOT
    manifest = 'opencood/tools/nego_native_sha256.json'
    data = json.loads((source / manifest).read_text(encoding='utf-8'))
    for rel in list(data['files']) + [manifest]:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dest)
    (tmp_path / 'bash_files').mkdir()
    monkeypatch.setattr(checker, 'ROOT', tmp_path)
    return tmp_path

def test_without_git_history(tree):
    assert not (tree / '.git').exists()
    with patch('subprocess.check_output', side_effect=AssertionError('unexpected Git lookup')):
        assert len(checker.audit_sources()['unchanged_native_files']) == 21

def test_credential_normalization(tree):
    path = tree / 'opencood/tools/train_nego_onlytrain.py'
    original = path.read_bytes()
    path.write_bytes(b"os.environ['WANDB_API_KEY'] = 'synthetic_test_key'\n" + original)
    assert checker.audit_sources()['unchanged_native_files']
    path.write_bytes(b"os.environ['WANDB_API_KEY'] = 'synthetic_test_key'\nprint('changed')\n" + original)
    with pytest.raises(ValueError, match='Unexpected native algorithm change'):
        checker.audit_sources()
    path.write_bytes(b"os.environ['WANDB_API_KEY'] = str('test')\n" + original)
    with pytest.raises(ValueError, match='assignment form'):
        checker.audit_sources()

def test_algorithm_tampering(tree):
    path = tree / 'opencood/loss/nego_loss.py'
    path.write_bytes(path.read_bytes() + b'\n# unexpected modification\n')
    with pytest.raises(ValueError, match='Unexpected native algorithm change'):
        checker.audit_sources()
