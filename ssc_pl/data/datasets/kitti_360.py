import glob
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from ...utils.helper import compute_CP_mega_matrix, compute_local_frustums, vox2pix

SPLITS = {
    'train':
    ('2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
     '2013_05_28_drive_0007_sync'),
    'val': ('2013_05_28_drive_0006_sync', ),
    'test': ('2013_05_28_drive_0009_sync', ),
}

KITTI_360_CLASS_FREQ = torch.tensor([
    2264087502, 20098728, 104972, 96297, 1149426, 4051087, 125103, 105540713, 16292249, 45297267,
    14454132, 110397082, 6766219, 295883213, 50037503, 1561069, 406330, 30516166, 1950115
])


class KITTI360(Dataset):

    META_INFO = {
        'class_weights':
        1 / torch.log(KITTI_360_CLASS_FREQ + 1e-6),
        'class_names':
        ('empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object')
    }

    def __init__(
        self,
        split,
        data_root,
        label_root,
        depth_root=None,
        project_scale=2,
        frustum_size=4,
        context_prior=False,
        flip=True,
    ):
        super().__init__()
        self.data_root = data_root
        self.label_root = label_root
        self.sequences = SPLITS[split]
        self.split = split

        self.depth_root = depth_root
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.context_prior = context_prior
        self.flip = flip
        self.num_classes = 19

        self.voxel_origin = np.array((0, -25.6, -2))
        self.voxel_size = 0.2
        self.scene_size = (51.2, 51.2, 6.4)
        self.img_shape = (1408, 376)

        self.scans = []
        calib = self.read_calib()
        for sequence in self.sequences:
            P = calib['P2']
            T_velo_2_cam = calib['Tr']
            proj_matrix = P @ T_velo_2_cam

            glob_path = osp.join(self.data_root, 'data_2d_raw', sequence, 'voxels', '*.bin')
            for voxel_path in glob.glob(glob_path):
                self.scans.append({
                    'sequence': sequence,
                    'P': P,
                    'T_velo_2_cam': T_velo_2_cam,
                    'proj_matrix': proj_matrix,
                    'voxel_path': voxel_path,
                })

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan = self.scans[idx]
        sequence = scan['sequence']
        P = scan['P']
        T_velo_2_cam = scan['T_velo_2_cam']
        proj_matrix = scan['proj_matrix']

        filename = osp.basename(scan['voxel_path'])
        frame_id = osp.splitext(filename)[0]
        data = {
            'frame_id': frame_id,
            'sequence': sequence,
            'P': P,
            'cam_pose': T_velo_2_cam,
            'proj_matrix': proj_matrix,
            'voxel_origin': self.voxel_origin
        }
        label = {}

        scale_3ds = (self.output_scale, self.project_scale)
        data['scale_3ds'] = scale_3ds
        cam_K = P[:3, :3]
        data['cam_K'] = cam_K
        for scale_3d in scale_3ds:
            # compute the 3D-2D mapping
            projected_pix, fov_mask, pix_z = vox2pix(T_velo_2_cam, cam_K, self.voxel_origin,
                                                     self.voxel_size * scale_3d, self.img_shape,
                                                     self.scene_size)
            data[f'projected_pix_{scale_3d}'] = projected_pix
            data[f'pix_z_{scale_3d}'] = pix_z
            data[f'fov_mask_{scale_3d}'] = fov_mask

        flip = random.random() > 0.5 if self.flip and self.split == 'train' else False
        target_1_path = osp.join(self.label_root, sequence, frame_id + '_1_1.npy')
        target = np.load(target_1_path)
        if flip:
            target = np.flip(target, axis=1).copy()
        label['target'] = target

        if self.context_prior:
            target_8_path = osp.join(self.label_root, sequence, frame_id + '_1_8.npy')
            target_1_8 = np.load(target_8_path)
            CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
            label['CP_mega_matrix'] = CP_mega_matrix

        if self.depth_root is not None:
            depth_path = osp.join(self.depth_root, sequence, frame_id + '.npy')
            depth = np.load(depth_path)[:self.img_shape[1], :self.img_shape[0]]
            if flip:
                depth = np.flip(depth, axis=1).copy()
            data['depth'] = depth

        # Compute the masks, each indicate the voxels of a local frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            data[f'projected_pix_{self.output_scale}'],
            data[f'pix_z_{self.output_scale}'],
            target,
            self.img_shape,
            n_classes=self.num_classes,
            size=self.frustum_size,
        )
        label['frustums_masks'] = frustums_masks
        label['frustums_class_dists'] = frustums_class_dists

        img_path = osp.join(self.data_root, 'data_2d_raw', sequence, 'image_00/data_rect',
                            frame_id + '.png')
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img, dtype=np.float32) / 255.0
        if flip:
            img = np.flip(img, axis=1).copy()
        img = img[:self.img_shape[1], :self.img_shape[0]]  # crop image
        data['img'] = self.transforms(img)  # (3, H, W)

        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)

        ndarray_to_tensor(data)
        ndarray_to_tensor(label)
        return data, label

    @staticmethod
    def read_calib():
        P = np.array([
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]).reshape(3, 4)

        cam2velo = np.array([
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]).reshape(3, 4)
        C2V = np.concatenate([cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        V2C = np.linalg.inv(C2V)
        V2C = V2C[:3, :]

        out = {}
        out['P2'] = P
        out['Tr'] = np.identity(4)
        out['Tr'][:3, :4] = V2C
        return out
