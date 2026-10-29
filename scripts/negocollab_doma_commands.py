"""PRINT ONLY. Review/copy commands individually on Silas; does not launch tasks."""
import argparse
from pathlib import PurePosixPath

from opencood.tools.nego_protocol import STAGES, config_root, read_yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['opv2v','v2xset'],required=True)
    p.add_argument('--gpu',default='CHOOSE_IDLE_GPU',help='Explicit idle GPU selected by the user')
    args=p.parse_args()
    name='OPV2V_H' if args.dataset=='opv2v' else 'V2XSET'
    root=PurePosixPath('opencood/logs')/f'NegoCollab_{name}_SEED303'
    base='python -u -B -m opencood.tools.run_nego_doma'
    print('# Planned deployment path only; first transfer/review the code yourself.')
    print('cd /home/fkk/my_data/disk2/code/NegoCollab')
    print(f'python -B -m opencood.tools.check_nego_doma --check-data {args.dataset} --output docs/server_{args.dataset}_file_audit.json')
    print(f'{base} prepare --dataset {args.dataset} --experiment {root}')
    for stage in STAGES[:-1]:
        cfg=read_yaml(config_root(args.dataset)/stage/'config.yaml')
        budget=cfg['train_params']
        print(f'\n# {stage}: real YAML {root/stage}/config.yaml; epochs={budget["epoches"]}, batch={budget["batch_size"]}, lr={cfg["optimizer"]["lr"]}; parents={cfg["nego_protocol"]["parents"]}')
        print(f'CUDA_VISIBLE_DEVICES={args.gpu} {base} smoke --experiment {root} --stage {stage}')
        print(f'CUDA_VISIBLE_DEVICES={args.gpu} {base} train --experiment {root} --stage {stage}')
        print('# To resume the complete state, append --resume to the train command.')
    print('\n# Assembly requires every selected parent best.pth; key/shape/source checked.')
    print(f'CUDA_VISIBLE_DEVICES={args.gpu} {base} assemble --experiment {root}')
    for n in (1,2,3,4):
        print(f'CUDA_VISIBLE_DEVICES={args.gpu} {base} infer --experiment {root} --use-cav {n}')
    print(f'{base} summarize --experiment {root}')


if __name__=='__main__':
    main()
