import os.path as osp
import glob
import numpy as np
import torch
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from ...utils.helper import vox2pix, compute_local_frustums, compute_CP_mega_matrix


class NYUv2(Dataset):

    META_INFO = {
        'class_weights':
        torch.tensor((0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
        'class_names': ('empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa',
                        'table', 'tvs', 'furn', 'objs'),
    }

    def __init__(self, split, data_root, label_root, depth_root=None, frustum_size=4):
        self.data_root = osp.join(data_root, 'NYU' + split)
        self.label_root = osp.join(label_root, 'NYU' + split)
        self.depth_root = osp.join(depth_root, 'NYU' + split) if depth_root else None
        self.frustum_size = frustum_size
        self.num_classes = 12

        self.voxel_size = 0.08  # meters
        self.scene_size = (4.8, 4.8, 2.88)  # meters
        self.img_shape = (640, 480)
        self.cam_K = np.array(((518.8579, 0, 320), (0, 518.8579, 240), (0, 0, 1)))

        self.scan_names = glob.glob(osp.join(self.data_root, '*.bin'))
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        filename = osp.basename(self.scan_names[idx])[:-4]
        filepath = osp.join(self.label_root, filename + '.pkl')
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        label = {}
        cam_pose = np.linalg.inv(data['cam_pose'])
        data['cam_pose'] = cam_pose
        voxel_origin = data['voxel_origin']
        data['cam_K'] = self.cam_K
        # Following SSC literature, the output resolution on NYUv2 is set to 1/4
        target = data.pop('target_1_4').transpose(0, 2, 1)
        label['target'] = target
        target_1_4 = data.pop('target_1_16').transpose(0, 2, 1)

        CP_mega_matrix = compute_CP_mega_matrix(target_1_4, is_binary=False)
        label['CP_mega_matrix'] = CP_mega_matrix

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(cam_pose, self.cam_K, voxel_origin,
                                                 self.voxel_size, self.img_shape, self.scene_size)
        data['projected_pix_1'] = projected_pix
        data['fov_mask_1'] = fov_mask

        # compute the masks, each indicates voxels inside a frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            self.img_shape,
            n_classes=self.num_classes,
            size=self.frustum_size,
        )
        label['frustums_masks'] = frustums_masks
        label['frustums_class_dists'] = frustums_class_dists

        img_path = osp.join(self.data_root, filename + '_color.jpg')
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img, dtype=np.float32) / 255.0
        data['img'] = self.transforms(img)  # (3, H, W)

        if self.depth_root is not None:
            depth_path = osp.join(self.data_root, filename + '.png')
            depth = Image.open(depth_path)
            data['depth'] = np.array(depth) / 8000.  # noqa

        def ndarray_to_tensor(data: dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        v = v.astype('float32')
                    data[k] = torch.from_numpy(v)

        ndarray_to_tensor(data)
        ndarray_to_tensor(label)
        return data, label
