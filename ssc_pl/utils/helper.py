import numpy as np
from . import fusion


def compute_CP_mega_matrix(target, is_binary=False):
    """
    Args:
        target: (H, W, D)
            contains voxels semantic labels
        is_binary: bool
            if True, return binary voxels relations else return 4-way relations
    """
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i // 2 for i in target.shape]
    if is_binary:
        matrix = np.zeros((2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]),
                          dtype=np.uint8)
    else:
        matrix = np.zeros((4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]),
                          dtype=np.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = (
                    xx * super_voxel_size[1] * super_voxel_size[2] + yy * super_voxel_size[2] + zz)
                label_col_megas = np.array([
                    target[xx * 2, yy * 2, zz * 2],
                    target[xx * 2 + 1, yy * 2, zz * 2],
                    target[xx * 2, yy * 2 + 1, zz * 2],
                    target[xx * 2, yy * 2, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2, zz * 2 + 1],
                    target[xx * 2, yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N) * label_col_mega
                    if not is_binary:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0),
                               col_idx] = 1.0  # non non same
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) &
                               (label_row != 0), col_idx] = 1.0  # non non diff
                        matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0),
                               col_idx] = 1.0  # empty empty
                        matrix[3, (label_row != 255) & (label_row != label_col) &
                               ((label_row == 0) | (label_col == 0)),
                               col_idx] = 1.0  # nonempty empty
                    else:
                        matrix[0, (label_row != 255) & (label_col != label_row),
                               col_idx] = 1.0  # diff
                        matrix[1, (label_row != 255) & (label_col == label_row),
                               col_idx] = 1.0  # same
    return matrix


def vox2pix(cam_E, cam_K, vol_origin, vox_size, img_shape, scene_size):
    """
    Compute the 2D projection of voxels centroids.

    Args:
        cam_E: (4, 4)
            camera pose in case of NYUv2 dataset,
            transformation from camera to lidar coordinate in case of SemKITTI
        cam_k: (3, 3)
            camera intrinsics
        vox_origin: (3,)
            world(NYU) / lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
        img_shape: (image width, image height)
        scene_size: (3,)
            scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

    Returns:
        projected_pix: (N, 2)
            Projected 2D positions of voxels
        fov_mask: (N,)
            Voxels mask indice voxels inside image's FOV
        pix_z: (N,)
            Voxels' distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vol_origin
    vol_bnds[:, 1] = vol_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / vox_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
    vox_coords = np.concatenate([xv.reshape(
        1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0).astype(int).T

    # Project voxels' centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vol_origin, vox_coords, vox_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_K)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    img_w, img_h = img_shape
    fov_mask = np.logical_and(
        pix_x >= 0,
        np.logical_and(pix_x < img_w,
                       np.logical_and(pix_y >= 0, np.logical_and(pix_y < img_h, pix_z > 0))))

    return projected_pix, fov_mask, pix_z


def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    return (pix_x >= min_x) & (pix_x < max_x) & (pix_y >= min_y) & (pix_y < max_y) & (pix_z > 0)


def compute_local_frustums(projected_pix, pix_z, target, img_shape, n_classes, size=4):
    """
    Compute the local frustums mask and their class frequencies

    Args:
        projected_pix: (N, 2)
            2D projected pix of all voxels
        pix_z: (N,)
            Distance of the camera sensor to voxels
        target: (H, W, D)
            Voxelized sematic labels
        img_shape: (image width, image height)
        n_classes: int
            Number of classes (12 for NYU and 20 for SemKITTI)
        size: int
            determine the number of local frustums i.e. size * size

    Returns:
        frustums_masks: (n_frustums, N)
            List of frustums_masks, each indicates the belonging voxels
        frustums_class_dists: (n_frustums, n_classes)
            Contains the class frequencies in each frustum
    """
    H, W, D = target.shape
    ranges = [(i * 1.0 / size, (i * 1.0 + 1) / size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    img_w, img_h = img_shape
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_w
            end_x = x[1] * img_w
            start_y = y[0] * img_h
            end_y = y[1] * img_h
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y,
                                                  pix_z)
            mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks = np.array(local_frustum_masks, dtype=np.float32)
    frustums_class_dists = np.array(local_frustum_class_dists, dtype=np.float32)
    return frustums_masks, frustums_class_dists
