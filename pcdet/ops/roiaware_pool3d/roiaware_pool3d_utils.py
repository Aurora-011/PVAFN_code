import torch
import torch.nn as nn
from torch.autograd import Function

from ...utils import common_utils
from . import roiaware_pool3d_cuda

#用于将一组点（points）分配到一组包围框（boxes）中
def points_in_boxes_cpu(points, boxes):
    # 框：xyz坐标，dx,dy,dz表示框的尺寸，heading表示框的旋转角度（朝向）
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)#存储每个点所属的包围框索引
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)
    #它会将每个点分配到对应的包围框中，并将结果存储到 point_indices 中
    return point_indices.numpy() if is_numpy else point_indices

#用于在 GPU 上计算点在包围框中索引的函数
def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)
    return box_idxs_of_pts

#用于执行 RoI（Region of Interest）感知的 3D 池化操作
class RoIAwarePool3d(nn.Module):
    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size#out_size：指定输出的尺寸，通常是一个三元组 (D, H, W)，表示输出的深度、高度和宽度
        self.max_pts_each_voxel = max_pts_each_voxel#指定每个体素（voxel）中最大的点数

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        #rois：感兴趣区域（Region of Interest），通常是一组包围框或感兴趣区域的坐标信息。
        #pts：点云数据，包含点的三维坐标。
        #pts_feature：点云特征数据，例如每个点的颜色、法线等。
        #pool_method：指定池化方法，可以是 'max'（最大池化）或 'avg'（平均池化）
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)


class RoIAwarePool3dFunction(Function):
    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        Args:
            ctx:上下文对象，用于保存中间计算结果，以供反向传播时使用
            rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
            pts: (npoints, 3)
            pts_feature: (npoints, C)
            out_size: int or tuple, like 7 or (7, 7, 7),可以是一个整数或一个三元组 (out_x, out_y, out_z)，表示输出的深度、高度和宽度
            max_pts_each_voxel:
            pool_method: 'max' or 'avg'

        Returns:
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        assert rois.shape[1] == 7 and pts.shape[1] == 3
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size

        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]

        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)

        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)

        ctx.roiaware_pool3d_for_backward = (pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels)
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward

        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method)

        return None, None, grad_in, None, None, None


if __name__ == '__main__':
    pass
