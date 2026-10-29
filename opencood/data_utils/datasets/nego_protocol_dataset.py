"""Isolated indexing/sensors; native pair, heter and ordered-GT processing."""
import bisect
import copy
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from opencood.tools.nego_protocol import ProtocolAdaptor, assert_prefix, lidar_path
from opencood.utils.nego_pcd import read_pcd


def build_protocol_dataset(cfg, train=False):
    from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
    from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset
    from opencood.data_utils.datasets.intermediate_heter_pair_fusion_dataset import getIntermediateheterpairFusionDataset
    from opencood.data_utils.datasets.heter_infer.intermediate_heter_infer_fusion_dataset import getIntermediateheterinferFusionDataset
    from opencood.hypes_yaml.yaml_utils import load_yaml

    class ProtocolBase(OPV2VBaseDataset):
        def generate_object_center_camera(self,cav_contents,reference_lidar_pose):
            if self.params['nego_protocol']['dataset']=='v2xset':
                return self.post_processor.generate_object_center_v2xset_camera(cav_contents,reference_lidar_pose)
            return super().generate_object_center_camera(cav_contents,reference_lidar_pose)

        def make_protocol_adaptor(self, *args):
            return ProtocolAdaptor(*args, dataset=self.params['nego_protocol']['dataset'],
                                   paired=self.params['fusion']['core_method']=='intermediateheterpair')

        def reinitialize(self):
            # OPV2VBase calls this once before the fusion constructor installs its adaptor.
            if not getattr(self,'heterogeneous',False):
                return
            self.scenario_database=OrderedDict()
            self.len_record=[]
            self.protocol_identities=[]
            dataset=self.params['nego_protocol']['dataset']
            paired=self.params['fusion']['core_method']=='intermediateheterpair'
            excluded=set(tuple(x) for x in self.params['nego_protocol']['excluded_frames'])
            for folder in self.scenario_folders:
                folder=Path(folder)
                if folder.name in self.params['nego_protocol']['scene_exclusions']:
                    raise ValueError('Scene exclusions require an audited implementation and server ledger')
                scene=folder.name
                cavs=sorted(p.name for p in folder.iterdir() if p.is_dir())
                cavs=self.adaptor.reorder_cav_list(cavs,scene)[:self.max_cav]
                if not cavs:
                    raise ValueError('Empty scenario: '+scene)
                records=OrderedDict()
                timestamps=None
                for j,cav in enumerate(cavs):
                    frames=sorted(p.stem for p in (folder/cav).glob('*.yaml') if 'additional' not in p.name and (scene,p.stem) not in excluded)
                    if timestamps is None:
                        timestamps=frames
                    if not set(timestamps).issubset(frames):
                        raise ValueError('Missing CAV frame metadata: '+str(folder/cav))
                    frames_dict=OrderedDict()
                    # Assignment is per CAV/epoch, not independently resampled per frame.
                    modality=self.adaptor.reassign_cav_modality(self.modality_assignment[scene][cav],j)
                    for stamp in timestamps:
                        original=(folder/cav/(stamp+'.pcd')).as_posix()
                        frame={'yaml':str(folder/cav/(stamp+'.yaml')), 'original_lidar':original,
                               'lidar':lidar_path(original,dataset,'m1' if paired else modality,require=False),
                               'cameras':[str(folder/cav/f'{stamp}_camera{k}.png') for k in range(4)],
                               'depths':[str(folder/cav/f'{stamp}_depth{k}.png').replace('OPV2V/','OPV2V_Hetero/').replace('OPV2V\\','OPV2V_Hetero\\') for k in range(4)],
                               'modality_name':modality,'subdataset_label':self.modality_assignment[scene][cav].replace('m','d')}
                        for ext in self.add_data_extension:
                            frame[ext]=str(folder/cav/f'{stamp}_{ext}')
                        frames_dict[stamp]=frame
                    frames_dict['ego']=j==0
                    records[cav]=frames_dict
                if not timestamps:
                    raise ValueError('Empty scenario timestamps: '+scene)
                self.scenario_database[len(self.scenario_database)]=records
                self.protocol_identities.extend({'scene':scene,'timestamp':t,'all_cav_ids':cavs} for t in timestamps)
                self.len_record.append(len(self.protocol_identities))
            if not self.len_record:
                raise ValueError('Empty split: '+self.root_dir)

        def retrieve_base_data(self,idx):
            scene_idx=bisect.bisect_right(self.len_record,idx)
            identity=self.protocol_identities[idx]
            timestamp=identity['timestamp']
            data=OrderedDict()
            for cav,frames in self.scenario_database[scene_idx].items():
                frame=frames[timestamp]
                metadata=Path(frame['yaml'])
                json_file=metadata.with_suffix('.json')
                params=json.loads(json_file.read_text(encoding='utf-8')) if json_file.exists() else load_yaml(str(metadata))
                item={'ego':frames['ego'],'params':params,'modality_name':frame['modality_name'],
                      'subdataset_label':frame['subdataset_label'],'_nego_original_pcd':frame['original_lidar']}
                if self.load_lidar_file or self.visualize:
                    item['lidar_np']=read_pcd(frame['lidar'])
                if self.load_camera_file or self.load_depth_file:
                    hdf5=Path(frame['cameras'][0].replace('camera0.png','imgs.hdf5'))
                    if self.use_hdf5 and hdf5.is_file():
                        with h5py.File(hdf5,'r') as f:
                            if self.load_camera_file:
                                item['camera_data']=[Image.fromarray(f[f'camera{k}'][()]) for k in range(4)]
                            if self.load_depth_file:
                                item['depth_data']=[Image.fromarray(f[f'depth{k}'][()]) for k in range(4)]
                    else:
                        for key,paths,enabled in (('camera_data',frame['cameras'],self.load_camera_file),('depth_data',frame['depths'],self.load_depth_file)):
                            if enabled:
                                images=[]
                                for path in paths:
                                    with Image.open(path) as im:
                                        images.append(im.copy())
                                item[key]=images
                for ext in self.add_data_extension:
                    import cv2
                    path=Path(frame[ext])
                    if not path.is_file():
                        # Same additional metadata tree as native OPV2V loader.
                        parts=list(path.parts)
                        for split in ('train','validate','test'):
                            if split in parts:
                                i=parts.index(split)
                                parts.insert(i,'additional')
                                break
                        path=Path(*parts)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    item[ext]=load_yaml(str(path)) if path.suffix=='.yaml' else cv2.imread(str(path))
                    if item[ext] is None:
                        raise ValueError('Cannot decode '+str(path))
                data[cav]=item
            return data

    fusion=cfg['fusion']['core_method']
    factory={'intermediateheter':getIntermediateheterFusionDataset,
             'intermediateheterpair':getIntermediateheterpairFusionDataset,
             'intermediateheterinfer':getIntermediateheterinferFusionDataset}[fusion]
    Native=factory(ProtocolBase)

    class ProtocolDataset(Native):
        def __init__(self,params,visualize,train):
            super().__init__(params,visualize,train)
            if params['nego_protocol']['dataset']=='v2xset':
                self.use_hdf5=False
            self.modality_name_list=sorted(params['heter']['modality_setting'])
            if hasattr(self,'modality_type_list'):
                self.modality_type_list=list(self.modality_name_list)

        def get_item_single_car(self, selected, ego, *modality_args):
            if fusion=='intermediateheterpair':
                modality=modality_args[0]
                selected=copy.copy(selected)
                selected['lidar_np']=read_pcd(lidar_path(selected['_nego_original_pcd'],self.params['nego_protocol']['dataset'],modality))
            return super().get_item_single_car(selected,ego,*modality_args)

        def __getitem__(self,idx):
            item=super().__getitem__(idx)
            if item is None:
                raise ValueError('Dataset returned an empty frame: '+str(idx))
            if fusion=='intermediateheterinfer':
                realized=item['ego']['agent_modality_list']
                assert_prefix(realized,self.use_cav)
                if len(realized)!=min(self.use_cav,len(item['ego']['cav_id_list'])):
                    raise ValueError('Incorrect input cutoff')
            return item

    # DOMA ordered inference uses visualize=True to select LiDAR GT for camera members.
    # Training retains native sensor-specific label behavior.
    return ProtocolDataset(cfg,visualize=not train and cfg['nego_protocol']['phase']!='pretrain',train=train)
