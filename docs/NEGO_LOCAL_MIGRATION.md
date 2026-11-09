# NegoCollab local repository consolidation

Target: `F:\zzuStudy\exp_gpt\Baselines\NegoCollab`
Source: `F:\zzuStudy\exp_gpt\Baselines\NegoCollab-protocol-review`

Target branch/main HEAD preserved: `8cd12e68549ab23109a691230990004a38402fef`. Source HEAD: `620385e6d001eec26c8faa5be8f73bb601e51d07`.
This is a file-level consolidation without commits, pushes, history resets, or a copied .git directory. The 12 source commits remain available in the shared Git database.

## Verification

- 722 source paths covered by SHA-256 records in NEGO_LOCAL_MIGRATION.json.
- 41 new/updated files migrated; identical/line-ending-equivalent target files retained.
- 7 source-relative extensions implement the latest requested behavior; exact diffs saved in NEGO_LOCAL_MIGRATION.diff.
- 193 CPU tests passed. Runtime Hook, workers=8, OOM/partial acceptance, current-configuration resume and flat inference layout covered.
- Python compilation and Python 3.8 AST syntax checks passed; no Python 3.8 runtime or GPU claim.
- Bash syntax and git diff --check passed. Model/data implementation signature remains 4d5457fa6bce5c7d2aff099131a739ce7a4e22e77e5037b843ec613d5c967509.
- Source deletion status is recorded in the JSON manifest. No server, formal training, user dataset or experiment checkpoint was modified.

## Current behavior

Runtime batch sizes are positive integers without a YAML upper bound, and only affect shuffled training loaders. Both worker pools default to 8 in the launcher. Validation/test batch size stays 1.
Checkpoint file integrity, state keys/shapes/dtypes, model/data protocol identity and real parent provenance stay strict. Full config hashes remain audit records. Current optimizer hyperparameters replace saved group settings; compatible parameter moments are restored. A current-configuration scheduler is positioned at the completed epoch with the current loader length. Incompatible optimizer types start fresh optimizer state and log that decision.
History from reviewed d825dfe tooling is accepted only with the unchanged model/data implementation signature. Unknown implementations and ambiguous legacy resume metadata remain rejected.
Historical M4 last=23/best=22 remains partial_accepted even if the requested epoch budget changes. It does not retrain; m4/align consumes its unchanged best checkpoint. Other CUDA OOMs use verified checkpoints and archived artifacts before retries or partial acceptance.
Final inference files are final_infer/result_CAVn.json, forward_encoders_CAVn.json and ap_CAVn.csv; assembly/source records and AP_CAV1_4 summaries share that directory.

## Migrated file list

- `MULTI_BATCH_FULL_AUDIT.md` (copy_new)
- `MULTI_BATCH_PRETRAIN_IMPLEMENTATION.md` (copy_new)
- `scripts/nego_runtime_batch/sitecustomize.py` (copy_new)
- `tests/test_nego_runtime_batch_hook.py` (copy_new)
- `.gitignore` (merge_source_update)
- `docs/NEGO_PRETRAIN_VECTOR_VALIDATION.md` (copy_new)
- `opencood/data_utils/datasets/nego_m1_batch.py` (copy_new)
- `opencood/data_utils/datasets/nego_protocol_dataset.py` (merge_source_update)
- `opencood/hypes_yaml/opv2v/doma_protocol_negocollab/protocol_lock.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/alliance/ft/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/alliance/nego/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/m3/align/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/m3/ft/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/m4/align/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/m4/ft/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/pretrain/m2/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/pretrain/m4/config.yaml` (merge_source_update)
- `opencood/hypes_yaml/v2xset/doma_protocol_negocollab/protocol_lock.yaml` (merge_source_update)
- `opencood/models/heter_encoders.py` (merge_source_update)
- `opencood/models/sub_modules/pillar_vfe.py` (merge_source_update)
- `opencood/models/sub_modules/point_pillar_scatter.py` (merge_source_update)
- `opencood/tools/check_nego_doma.py` (merge_source_update)
- `opencood/tools/generate_nego_doma_configs.py` (merge_source_update)
- `opencood/tools/nego_native_sha256.json` (copy_new)
- `opencood/tools/nego_protocol.py` (merge_source_update)
- `opencood/tools/nego_state.py` (merge_source_update)
- `opencood/tools/run_nego_doma.py` (merge_source_update)
- `scripts/run_nego_v2xset_doma_r2.sh` (copy_new)
- `scripts/validate_nego_m1_vectorized.py` (copy_new)
- `scripts/validate_nego_pretrain_vectorized.py` (copy_new)
- `tests/test_nego_checkpoint_compatibility.py` (copy_new)
- `tests/test_nego_dataloader_runtime.py` (copy_new)
- `tests/test_nego_doma.py` (merge_source_update)
- `tests/test_nego_hyperparameters.py` (copy_new)
- `tests/test_nego_m1_vectorized.py` (copy_new)
- `tests/test_nego_native_integrity.py` (copy_new)
- `tests/test_nego_pretrain_multibatch.py` (copy_new)
- `tests/test_nego_pretrain_validation_tool.py` (copy_new)
- `tests/test_nego_r2_orchestrator.py` (copy_new)
- `tests/test_nego_training_reproducibility.py` (copy_new)
- `tests/test_nego_v2xset_fixes.py` (copy_new)

Additional acceptance test: `tests/test_nego_migration.py`. Original audit reports were preserved as historical records.
