import os
import pickle

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.progress import track
from mayavi import mlab

COLORS = np.array([
    [100, 150, 245, 255],
    [100, 230, 245, 255],
    [30, 60, 150, 255],
    [80, 30, 180, 255],
    [100, 80, 250, 255],
    [255, 30, 30, 255],
    [255, 40, 200, 255],
    [150, 30, 90, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [175, 0, 75, 255],
    [255, 200, 0, 255],
    [255, 120, 50, 255],
    [0, 175, 0, 255],
    [135, 60, 0, 255],
    [150, 240, 80, 255],
    [255, 240, 150, 255],
    [255, 0, 0, 255],
]).astype(np.uint8)

KITTI360_COLORS = np.concatenate((
    COLORS[0:6],
    COLORS[8:15],
    COLORS[16:],
    np.array([[250, 150, 0, 255], [50, 255, 255, 255]]).astype(np.uint8),
), 0)


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(float)
    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)
    return coords_grid


def draw(
    voxels,
    cam_pose,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
    colors=None,
    save=False,
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array([
        [0, 0, 0],
        [x, y, d],
        [-x, y, d],
        [-x, -y, d],
        [x, -y, d],
    ])
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(cam_pose) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)
    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]
    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]
    # Draw the camera
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.triangular_mesh(
        x, y, z, triangles, representation='wireframe', color=(0, 0, 0), line_width=10)

    outfov_colors = colors.copy()
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2

    for i, grid_coords in enumerate((fov_grid_coords, outfov_grid_coords)):
        # Remove empty and unknown voxels
        voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]
        plt_plot = mlab.points3d(
            voxels[:, 0],
            voxels[:, 1],
            voxels[:, 2],
            voxels[:, 3],
            colormap='viridis',
            scale_factor=voxel_size - 0.05 * voxel_size,
            mode='cube',
            opacity=1.0,
            vmin=1,
            vmax=19)

        plt_plot.glyph.scale_mode = 'scale_by_vector'
        plt_plot.module_manager.scalar_lut_manager.lut.table = colors if i == 0 else outfov_colors

    plt_plot.scene.camera.zoom(1.3)
    if save:
        mlab.savefig(save, size=(224, 224))
        mlab.close()
    else:
        mlab.show()


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(config: DictConfig):
    files = ([os.path.join(config.path, f)
              for f in os.listdir(config.path)] if os.path.isdir(config.path) else [config.path])

    for file in track(files):
        with open(file, 'rb') as f:
            outputs = pickle.load(f)

        cam_pose = outputs['cam_pose'] if 'cam_pose' in outputs else outputs[
            'T_velo_2_cam']  # compatible with MonoScene
        vox_origin = np.array([0, -25.6, -2])
        fov_mask = outputs['fov_mask_1']
        pred = outputs['pred'] if 'pred' in outputs else outputs['y_pred']
        target = outputs['target']

        if config.data.datasets.type == 'SemanticKITTI':
            params = dict(
                img_size=(1220, 370),
                f=707.0912,
                voxel_size=0.2,
                d=7,
                colors=COLORS,
            )
        elif config.data.datasets.type == 'KITTI360':
            # Otherwise the trained model would output distorted results, due to unreasonably labeling
            # a large number of voxels as "ignored" in the annotations.
            pred[target == 255] = 0
            params = dict(
                img_size=(1408, 376),
                f=552.55426,
                voxel_size=0.2,
                d=7,
                colors=KITTI360_COLORS,
            )
        else:
            raise NotImplementedError

        file_name = file.split(os.sep)[-1].split(".")[0]
        for i, vol in enumerate((pred, target)):
            draw(vol, cam_pose, vox_origin, fov_mask, **params,
                 save=f'outputs/figures/{file_name}_{"gt" if i else "pred"}.png')


if __name__ == '__main__':
    main()
