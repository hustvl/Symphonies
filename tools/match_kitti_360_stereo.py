import os
import os.path as osp
import shutil

import numpy as np
from PIL import Image
from rich.progress import track

src_dir = '/home/users/haoyi.jiang/data/kitti_360/'
dst_dir = '/home/users/haoyi.jiang/data/kitti_360/data_2d_raw/'

for sequence in os.listdir(dst_dir):
    src_dir_l = osp.join(src_dir, 'image_00', sequence, 'image_00', 'data_rect')
    src_dir_r = osp.join(src_dir, 'image_01', sequence, 'image_01', 'data_rect')
    match_dir = osp.join(dst_dir, sequence, 'image_00', 'data_rect')
    src_files_l = iter(sorted(os.listdir(src_dir_l)))
    f2 = next(src_files_l)

    for f1 in track(os.listdir(match_dir)):
        img1 = Image.open(os.path.join(match_dir, f1))
        img1 = np.array(img1)

        while True:
            img2 = Image.open(os.path.join(src_dir_l, f2))
            img2 = np.array(img2)
            # NOTE: Compare by image hash can not work precisely due to the similarity of images
            if np.all(img1 == img2):
                dst_dir_ = os.path.join(dst_dir, sequence, 'image_01', 'data_rect')
                os.makedirs(dst_dir_, exist_ok=True)
                shutil.copy(os.path.join(src_dir_r, f2), os.path.join(dst_dir_, f1))
                print(sequence, f2, f1)
                break
            else:
                f2 = next(src_files_l)
