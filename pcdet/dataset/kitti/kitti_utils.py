import numpy as np
from ...utils import box_utils

#该函数用于将注释数据转换为KITTI格式
def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:注释数据，通常是一个字典列表
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)一个字典,用于将类别名称映射到KITTI格式的名称(例如,Car、Pedestrian、Cyclist)
        info_with_fakelidar:一个布尔值,表示注释数据中是否包含伪激光雷达信息
    Returns:函数的返回值是转换后的注释数据，仍然是一个字典列表

    """
    for anno in annos:
        # For lyft and nuscenes, different anno key in info
        if 'name' not in anno:
            anno['name'] = anno['gt_names']
            anno.pop('gt_names')

        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos

#用于将校准对象（calibration object）转换为变换矩阵
def calib_to_matricies(calib):
    #V2R：Lidar到矫正后的相机的变换矩阵，大小为(4, 4)。变换矩阵V2R表示从Lidar坐标系到矫正后的相机坐标系的变换
    #P2：相机投影矩阵，大小为(3, 4)投影矩阵P2用于将相机坐标系中的点投影到图像平面上
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2