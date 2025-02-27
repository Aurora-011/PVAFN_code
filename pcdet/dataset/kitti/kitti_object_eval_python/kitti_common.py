import concurrent.futures as futures
import os
import pathlib
import re
from collections import OrderedDict

import numpy as np
from skimage import io

#用于将图像索引转换为字符串形式
def get_image_index_str(img_idx):
    #函数的参数是 img_idx，表示图像索引
    return "{:06d}".format(img_idx)

#用于获取KITTI数据集中特定类型的信息文件的路径
def get_kitti_info_path(idx,
                        prefix,
                        info_type='image_2',
                        file_tail='.png',
                        training=True,
                        relative_path=True):
    #idx表示图像索引或信息文件的索引;
    #prefix：表示数据集的路径前缀。
    #info_type：表示要获取的信息文件的类型，默认为 'image_2'。
    #file_tail：表示信息文件的文件后缀，默认为 '.png'。
    #training：表示是否为训练数据集，默认为 True。
    #relative_path：表示是否返回相对路径，默认为 True
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_tail
    prefix = pathlib.Path(prefix)
    if training:
        file_path = pathlib.Path('training') / info_type / img_idx_str
    else:
        file_path = pathlib.Path('testing') / info_type / img_idx_str
    if not (prefix / file_path).exists():
        raise ValueError("file not exist: {}".format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)

#获取图片路径
def get_image_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'image_2', '.png', training,
                               relative_path)

#获取标签路径
def get_label_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'label_2', '.txt', training,
                               relative_path)

#获取点云路径
def get_velodyne_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'velodyne', '.bin', training,
                               relative_path)

#获取标定文件路径
def get_calib_path(idx, prefix, training=True, relative_path=True):
    return get_kitti_info_path(idx, prefix, 'calib', '.txt', training,
                               relative_path)

#用于将传入的矩阵进行扩展
def _extend_matrix(mat):
    #该函数的作用是将传入的矩阵在垂直方向上扩展，添加一个表示平移的行，形成一个 4x4 的齐次变换矩阵。这种扩展常用于将旋转矩阵与平移向量合并为一个完整的变换矩阵
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

#用于获取KITTI数据集中图像相关的信息
def get_kitti_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    #path：数据集的路径。
    #training：表示是否为训练数据集，默认为 True。
    #label_info：表示是否获取标签信息，默认为 True。
    #velodyne：表示是否获取Velodyne点云信息，默认为 False。
    #calib：表示是否获取标定信息，默认为 False。
    #image_ids：指定要获取信息的图像索引，默认为 7481，可以是单个索引或索引列表。
    #extend_matrix：表示是否对标定矩阵进行扩展，默认为 True。
    #num_worker：表示并行处理的工作进程数，默认为 8。
    #relative_path：表示是否返回相对路径，默认为 True。
    #with_imageshape：表示是否获取图像尺寸信息，默认为 True
    # image_infos = []
    root_path = pathlib.Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            image_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
        image_info['img_path'] = get_image_path(idx, path, training,
                                                relative_path)
        if with_imageshape:
            img_path = image_info['img_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['img_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array(
                [float(info) for info in lines[0].split(' ')[1:13]]).reshape(
                    [3, 4])
            P1 = np.array(
                [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
                    [3, 4])
            P2 = np.array(
                [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
                    [3, 4])
            P3 = np.array(
                [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
                    [3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            image_info['calib/P0'] = P0
            image_info['calib/P1'] = P1
            image_info['calib/P2'] = P2
            image_info['calib/P3'] = P3
            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect
            image_info['calib/R0_rect'] = rect_4x4
            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
            image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
        if annotations is not None:
            image_info['annos'] = annotations
            add_difficulty_to_annos(image_info)
        return image_info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)
    return list(image_infos)

#用于过滤KITTI数据集中图像的注释信息
def filter_kitti_anno(image_anno,
                      used_classes,
                      used_difficulty=None,
                      dontcare_iou=None):
    #image_anno：图像的注释信息，通常是一个字典，包含各种注释字段。
    #used_classes：要保留的类别列表或单个类别。
    #used_difficulty：要保留的难度等级列表，默认为 None，表示保留所有难度等级的注释信息。
    #dontcare_iou：如果使用了 DontCare 类别，表示与 DontCare 区域的IoU阈值，超过此阈值的边界框将被移除，默认为 None，表示不执行此操作。
    if not isinstance(used_classes, (list, tuple)):
        used_classes = [used_classes]
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(image_anno['name']) if x in used_classes
    ]
    for key in image_anno.keys():
        img_filtered_annotations[key] = (
            image_anno[key][relevant_annotation_indices])
    if used_difficulty is not None:
        relevant_annotation_indices = [
            i for i, x in enumerate(img_filtered_annotations['difficulty'])
            if x in used_difficulty
        ]
        for key in image_anno.keys():
            img_filtered_annotations[key] = (
                img_filtered_annotations[key][relevant_annotation_indices])

    if 'DontCare' in used_classes and dontcare_iou is not None:
        dont_care_indices = [
            i for i, x in enumerate(img_filtered_annotations['name'])
            if x == 'DontCare'
        ]
        # bounding box format [y_min, x_min, y_max, x_max]
        all_boxes = img_filtered_annotations['bbox']
        ious = iou(all_boxes, all_boxes[dont_care_indices])

        # Remove all bounding boxes that overlap with a dontcare region.
        if ious.size > 0:
            boxes_to_remove = np.amax(ious, axis=1) > dontcare_iou
            for key in image_anno.keys():
                img_filtered_annotations[key] = (img_filtered_annotations[key][
                    np.logical_not(boxes_to_remove)])
    return img_filtered_annotations

#用于过滤图像注释中得分低于指定阈值的注释信息
def filter_annos_low_score(image_annos, thresh):
    #thresh阈值
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, s in enumerate(anno['score']) if s >= thresh
        ]
        for key in anno.keys():
            img_filtered_annotations[key] = (
                anno[key][relevant_annotation_indices])
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos

#用于生成适用于KITTI数据集结果文件的注释行
def kitti_result_line(result_dict, precision=4):
    #包含注释信息的字典，其中键是注释字段的名称，值是相应的注释值;precision：可选参数，用于指定浮点数的精度，默认为4
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', None),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)

#用于根据给定的注释信息向注释中添加难度（difficulty）标签
def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for eval_utils其中 0 表示没有遮挡，1 表示部分遮挡，2 表示完全遮挡
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for eval_utils
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool_)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool_)
    hard_mask = np.ones((len(dims), ), dtype=np.bool_)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    return diff

#它读取一个标签文件，并返回一个包含注释信息的字典
def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations

#它批量从指定的标签文件夹中获取注释信息。
def get_label_annos(label_folder, image_ids=None):
    #label_folder：标签文件夹的路径
    #image_ids要获取注释信息的图像的标识符列表或范围
    if image_ids is None:
        filepaths = pathlib.Path(label_folder).glob('*.txt')
        prog = re.compile(r'^\d{6}.txt$')
        filepaths = filter(lambda f: prog.match(f.name), filepaths)
        image_ids = [int(p.stem) for p in filepaths]
        image_ids = sorted(image_ids)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    annos = []
    label_folder = pathlib.Path(label_folder)
    for idx in image_ids:
        image_idx = get_image_index_str(idx)
        label_filename = label_folder / (image_idx + '.txt')
        annos.append(get_label_anno(label_filename))
    return annos

#用于计算一组框（boxes）的面积
def area(boxes, add1=False):
    """Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    if add1:
        return (boxes[:, 2] - boxes[:, 0] + 1.0) * (
            boxes[:, 3] - boxes[:, 1] + 1.0)
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

#用于计算两组框（boxes1 和 boxes2）之间的两两交集区域的面积
def intersection(boxes1, boxes2, add1=False):
    """Compute pairwise intersection areas between boxes.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes
        boxes2: a numpy array with shape [M, 4] holding M boxes

    Returns:
        a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    if add1:
        all_pairs_min_ymax += 1.0
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)

    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    if add1:
        all_pairs_min_xmax += 1.0
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

#用于计算两组框（boxes1 和 boxes2）之间的两两交并比（Intersection over Union，IoU）得分
def iou(boxes1, boxes2, add1=False):
    """Computes pairwise intersection-over-union between box collections.

    Args:
        boxes1: a numpy array with shape [N, 4] holding N boxes.
        boxes2: a numpy array with shape [M, 4] holding N boxes.

    Returns:
        a numpy array with shape [N, M] representing pairwise iou scores.
    """
    intersect = intersection(boxes1, boxes2, add1)
    area1 = area(boxes1, add1)
    area2 = area(boxes2, add1)
    union = np.expand_dims(
        area1, axis=1) + np.expand_dims(
            area2, axis=0) - intersect
    return intersect / union
