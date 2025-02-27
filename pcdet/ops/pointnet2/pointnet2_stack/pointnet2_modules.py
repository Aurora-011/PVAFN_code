from typing import List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils

#用于构建局部聚合模块
def build_local_aggregation_module(input_channels, config):
    #cur_layer：局部聚合模块的实例
    #num_c_out：整数，表示输出特征的通道数
    local_aggregation_name = config.get('NAME', 'StackSAModuleMSG')

    if local_aggregation_name == 'StackSAModuleMSG':
        mlps = config.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]
        cur_layer = StackSAModuleMSG(
            radii=config.POOL_RADIUS, nsamples=config.NSAMPLE, mlps=mlps, use_xyz=True, pool_method='max_pool',
        )
        num_c_out = sum([x[-1] for x in mlps])#计算输出特征的通道数 num_c_out，通过对 mlps 中每个配置取最后一个元素，并求和得到
    elif local_aggregation_name == 'VectorPoolAggregationModuleMSG':
        cur_layer = VectorPoolAggregationModuleMSG(input_channels=input_channels, config=config)
        num_c_out = config.MSG_POST_MLPS[-1]
    else:
        raise NotImplementedError

    return cur_layer, num_c_out


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]],
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features
            )  # (M1 + M2, C, nsample)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)  # (1, C, M1 + M2 ..., nsample)
            new_features = self.mlps[k](new_features)  # (1, C, M1 + M2 ..., nsample)

            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)  # (M1 + M2 ..., C)

        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):
    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        new_features = self.mlp(new_features)

        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features


class VectorPoolLocalInterpolateModule(nn.Module):
    def __init__(self, mlp, num_voxels, max_neighbour_distance, nsample, neighbor_type, use_xyz=True,
                 neighbour_distance_multiplier=1.0, xyz_encoding_type='concat'):
        """
        Args:
            mlp:
            num_voxels:
            max_neighbour_distance:
            neighbor_type: 1: ball, others: cube
            nsample: find all (-1), find limited number(>0)
            use_xyz:
            neighbour_distance_multiplier:
            xyz_encoding_type:
        """
        super().__init__()
        self.num_voxels = num_voxels  # [num_grid_x, num_grid_y, num_grid_z]: number of grids in each local area centered at new_xyz
        self.num_total_grids = self.num_voxels[0] * self.num_voxels[1] * self.num_voxels[2]
        self.max_neighbour_distance = max_neighbour_distance
        self.neighbor_distance_multiplier = neighbour_distance_multiplier
        self.nsample = nsample
        self.neighbor_type = neighbor_type
        self.use_xyz = use_xyz
        self.xyz_encoding_type = xyz_encoding_type

        if mlp is not None:
            if self.use_xyz:
                mlp[0] += 9 if self.xyz_encoding_type == 'concat' else 0
            shared_mlps = []
            for k in range(len(mlp) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp[k + 1]),
                    nn.ReLU()
                ])
            self.mlp = nn.Sequential(*shared_mlps)
        else:
            self.mlp = None

        self.num_avg_length_of_neighbor_idxs = 1000

    def forward(self, support_xyz, support_features, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt):
        """
        Args:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            support_features: (N1 + N2 ..., C) point-wise features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        with torch.no_grad():
            dist, idx, num_avg_length_of_neighbor_idxs = pointnet2_utils.three_nn_for_vector_pool_by_two_step(
                support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt,
                self.max_neighbour_distance, self.nsample, self.neighbor_type,
                self.num_avg_length_of_neighbor_idxs, self.num_total_grids, self.neighbor_distance_multiplier
            )
        self.num_avg_length_of_neighbor_idxs = max(self.num_avg_length_of_neighbor_idxs, num_avg_length_of_neighbor_idxs.item())

        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / torch.clamp_min(norm, min=1e-8)

        empty_mask = (idx.view(-1, 3)[:, 0] == -1)
        idx.view(-1, 3)[empty_mask] = 0

        interpolated_feats = pointnet2_utils.three_interpolate(support_features, idx.view(-1, 3), weight.view(-1, 3))
        interpolated_feats = interpolated_feats.view(idx.shape[0], idx.shape[1], -1)  # (M1 + M2 ..., num_total_grids, C)
        if self.use_xyz:
            near_known_xyz = support_xyz[idx.view(-1, 3).long()].view(-1, 3, 3)  # ( (M1 + M2 ...)*num_total_grids, 3)
            local_xyz = (new_xyz_grid_centers.view(-1, 1, 3) - near_known_xyz).view(-1, idx.shape[1], 9)
            if self.xyz_encoding_type == 'concat':
                interpolated_feats = torch.cat((interpolated_feats, local_xyz), dim=-1)  # ( M1 + M2 ..., num_total_grids, 9+C)
            else:
                raise NotImplementedError

        new_features = interpolated_feats.view(-1, interpolated_feats.shape[-1])  # ((M1 + M2 ...) * num_total_grids, C)
        new_features[empty_mask, :] = 0
        if self.mlp is not None:
            new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
            new_features = self.mlp(new_features)

            new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return new_features


class VectorPoolAggregationModule(nn.Module):
    def __init__(
            self, input_channels, num_local_voxel=(3, 3, 3), local_aggregation_type='local_interpolation',
            num_reduced_channels=30, num_channels_of_local_aggregation=32, post_mlps=(128,),
            max_neighbor_distance=None, neighbor_nsample=-1, neighbor_type=0, neighbor_distance_multiplier=2.0):
        super().__init__()
        self.num_local_voxel = num_local_voxel
        self.total_voxels = self.num_local_voxel[0] * self.num_local_voxel[1] * self.num_local_voxel[2]
        self.local_aggregation_type = local_aggregation_type
        assert self.local_aggregation_type in ['local_interpolation', 'voxel_avg_pool', 'voxel_random_choice']
        self.input_channels = input_channels
        self.num_reduced_channels = input_channels if num_reduced_channels is None else num_reduced_channels
        self.num_channels_of_local_aggregation = num_channels_of_local_aggregation
        self.max_neighbour_distance = max_neighbor_distance
        self.neighbor_nsample = neighbor_nsample
        self.neighbor_type = neighbor_type  # 1: ball, others: cube

        if self.local_aggregation_type == 'local_interpolation':
            self.local_interpolate_module = VectorPoolLocalInterpolateModule(
                mlp=None, num_voxels=self.num_local_voxel,
                max_neighbour_distance=self.max_neighbour_distance,
                nsample=self.neighbor_nsample,
                neighbor_type=self.neighbor_type,
                neighbour_distance_multiplier=neighbor_distance_multiplier,
            )
            num_c_in = (self.num_reduced_channels + 9) * self.total_voxels
        else:
            self.local_interpolate_module = None
            num_c_in = (self.num_reduced_channels + 3) * self.total_voxels

        num_c_out = self.total_voxels * self.num_channels_of_local_aggregation

        self.separate_local_aggregation_layer = nn.Sequential(
            nn.Conv1d(num_c_in, num_c_out, kernel_size=1, groups=self.total_voxels, bias=False),
            nn.BatchNorm1d(num_c_out),
            nn.ReLU()
        )

        post_mlp_list = []
        c_in = num_c_out
        for cur_num_c in post_mlps:
            post_mlp_list.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.post_mlps = nn.Sequential(*post_mlp_list)

        self.num_mean_points_per_grid = 20
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def extra_repr(self) -> str:
        ret = f'radius={self.max_neighbour_distance}, local_voxels=({self.num_local_voxel}, ' \
              f'local_aggregation_type={self.local_aggregation_type}, ' \
              f'num_c_reduction={self.input_channels}->{self.num_reduced_channels}, ' \
              f'num_c_local_aggregation={self.num_channels_of_local_aggregation}'
        return ret

    def vector_pool_with_voxel_query(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        use_xyz = 1
        pooling_type = 0 if self.local_aggregation_type == 'voxel_avg_pool' else 1

        new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid = pointnet2_utils.vector_pool_with_voxel_query_op(
            xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt,
            self.num_local_voxel[0], self.num_local_voxel[1], self.num_local_voxel[2],
            self.max_neighbour_distance, self.num_reduced_channels, use_xyz,
            self.num_mean_points_per_grid, self.neighbor_nsample, self.neighbor_type,
            pooling_type
        )
        self.num_mean_points_per_grid = max(self.num_mean_points_per_grid, num_mean_points_per_grid.item())

        num_new_pts = new_features.shape[0]
        new_local_xyz = new_local_xyz.view(num_new_pts, -1, 3)  # (N, num_voxel, 3)
        new_features = new_features.view(num_new_pts, -1, self.num_reduced_channels)  # (N, num_voxel, C)
        new_features = torch.cat((new_local_xyz, new_features), dim=-1).view(num_new_pts, -1)

        return new_features, point_cnt_of_grid

    @staticmethod
    def get_dense_voxels_by_center(point_centers, max_neighbour_distance, num_voxels):
        """
        Args:
            point_centers: (N, 3)
            max_neighbour_distance: float
            num_voxels: [num_x, num_y, num_z]

        Returns:
            voxel_centers: (N, total_voxels, 3)
        """
        R = max_neighbour_distance
        device = point_centers.device
        x_grids = torch.arange(-R + R / num_voxels[0], R - R / num_voxels[0] + 1e-5, 2 * R / num_voxels[0], device=device)
        y_grids = torch.arange(-R + R / num_voxels[1], R - R / num_voxels[1] + 1e-5, 2 * R / num_voxels[1], device=device)
        z_grids = torch.arange(-R + R / num_voxels[2], R - R / num_voxels[2] + 1e-5, 2 * R / num_voxels[2], device=device)
        x_offset, y_offset, z_offset = torch.meshgrid(x_grids, y_grids, z_grids)  # shape: [num_x, num_y, num_z]
        xyz_offset = torch.cat((
            x_offset.contiguous().view(-1, 1),
            y_offset.contiguous().view(-1, 1),
            z_offset.contiguous().view(-1, 1)), dim=-1
        )
        voxel_centers = point_centers[:, None, :] + xyz_offset[None, :, :]
        return voxel_centers

    def vector_pool_with_local_interpolate(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        """
        Args:
            xyz: (N, 3)
            xyz_batch_cnt: (batch_size)
            features: (N, C)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size)
        Returns:
            new_features: (M, total_voxels * C)
        """
        voxel_centers = self.get_dense_voxels_by_center(
            point_centers=new_xyz, max_neighbour_distance=self.max_neighbour_distance, num_voxels=self.num_local_voxel
        )  # (M1 + M2 + ..., total_voxels, 3)
        voxel_features = self.local_interpolate_module.forward(
            support_xyz=xyz, support_features=features, xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz, new_xyz_grid_centers=voxel_centers, new_xyz_batch_cnt=new_xyz_batch_cnt
        )  # ((M1 + M2 ...) * total_voxels, C)

        voxel_features = voxel_features.contiguous().view(-1, self.total_voxels * voxel_features.shape[-1])
        return voxel_features

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features, **kwargs):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        N, C = features.shape

        assert C % self.num_reduced_channels == 0, \
            f'the input channels ({C}) should be an integral multiple of num_reduced_channels({self.num_reduced_channels})'

        features = features.view(N, -1, self.num_reduced_channels).sum(dim=1)

        if self.local_aggregation_type in ['voxel_avg_pool', 'voxel_random_choice']:
            vector_features, point_cnt_of_grid = self.vector_pool_with_voxel_query(
                xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features,
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )
        elif self.local_aggregation_type == 'local_interpolation':
            vector_features = self.vector_pool_with_local_interpolate(
                xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features,
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )  # (M1 + M2 + ..., total_voxels * C)
        else:
            raise NotImplementedError

        vector_features = vector_features.permute(1, 0)[None, :, :]  # (1, num_voxels * C, M1 + M2 ...)

        new_features = self.separate_local_aggregation_layer(vector_features)

        new_features = self.post_mlps(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return new_xyz, new_features


class VectorPoolAggregationModuleMSG(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        self.model_cfg = config
        self.num_groups = self.model_cfg.NUM_GROUPS

        self.layers = []
        c_in = 0
        for k in range(self.num_groups):
            cur_config = self.model_cfg[f'GROUP_CFG_{k}']
            cur_vector_pool_module = VectorPoolAggregationModule(
                input_channels=input_channels, num_local_voxel=cur_config.NUM_LOCAL_VOXEL,
                post_mlps=cur_config.POST_MLPS,
                max_neighbor_distance=cur_config.MAX_NEIGHBOR_DISTANCE,
                neighbor_nsample=cur_config.NEIGHBOR_NSAMPLE,
                local_aggregation_type=self.model_cfg.LOCAL_AGGREGATION_TYPE,
                num_reduced_channels=self.model_cfg.get('NUM_REDUCED_CHANNELS', None),
                num_channels_of_local_aggregation=self.model_cfg.NUM_CHANNELS_OF_LOCAL_AGGREGATION,
                neighbor_distance_multiplier=2.0
            )
            self.__setattr__(f'layer_{k}', cur_vector_pool_module)
            c_in += cur_config.POST_MLPS[-1]

        c_in += 3  # use_xyz

        shared_mlps = []
        for cur_num_c in self.model_cfg.MSG_POST_MLPS:
            shared_mlps.extend([
                nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False),
                nn.BatchNorm1d(cur_num_c),
                nn.ReLU()
            ])
            c_in = cur_num_c
        self.msg_post_mlps = nn.Sequential(*shared_mlps)

    def forward(self, **kwargs):
        features_list = []
        for k in range(self.num_groups):
            cur_xyz, cur_features = self.__getattr__(f'layer_{k}')(**kwargs)
            features_list.append(cur_features)

        features = torch.cat(features_list, dim=-1)
        features = torch.cat((cur_xyz, features), dim=-1)
        features = features.permute(1, 0)[None, :, :]  # (1, C, N)
        new_features = self.msg_post_mlps(features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)  # (N, C)

        return cur_xyz, new_features



#新增
class PyramidModule(nn.Module):

    def __init__(self, input_channels, nsamples, grid_sizes, num_heads, head_dims, attention_op, dp_value = 0.1, tr_mode = 'Normal'):
        """
        Args:
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        """
        super().__init__()
        input_channels += 3
        self.num_pyramid_levels = len(nsamples)
        self.pos_dims = 3
        self.num_heads = num_heads
        self.head_dims = head_dims
        self.output_dims = self.num_heads * self.head_dims
        self.grid_sizes = grid_sizes

        self.tr_mode = tr_mode
        assert self.tr_mode in ['NoTr', 'Normal', 'Residual']

        if self.tr_mode != 'NoTr':
            self.transformer_encoders = nn.ModuleList()
        self.cls_weight_module = nn.ModuleList()
        self.reg_weight_module = nn.ModuleList()

        self.pos_embeddings = []

        self.groupers = nn.ModuleList()
        self.pos_proj = nn.ModuleList()
        self.key_proj = nn.ModuleList()
        self.value_proj = nn.ModuleList()
        self.attention_proj = nn.ModuleList()
        self.norm_layer = nn.ModuleList()
        self.k_coef = nn.ModuleList()
        self.qk_coef = nn.ModuleList()
        self.q_coef = nn.ModuleList()
        self.v_coef = nn.ModuleList()
        for i in range(self.num_pyramid_levels):
            nsample = nsamples[i]
            grid_size = grid_sizes[i]
            self.groupers.append(pointnet2_utils.QueryAndGroupPyramidAttention(nsample))
            # vmh 1
            self.pos_proj.append(nn.Sequential(
                nn.Conv1d(self.pos_dims, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU(),
            ))
            self.key_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.ReLU()
            ))
            self.value_proj.append(nn.Sequential(
                nn.Conv1d(input_channels, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
                nn.Conv1d(self.output_dims, self.output_dims, 1, groups=1, bias=False),
                nn.BatchNorm1d(self.output_dims),
                nn.ReLU(),
            ))
            self.attention_proj.append(nn.Sequential(
                nn.Conv1d(self.output_dims, self.num_heads, 1, groups=1, bias=False),
            ))
            self.norm_layer.append(nn.Softmax(dim=-1))
            self.k_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.q_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.qk_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))
            self.v_coef.append(nn.Sequential(
                nn.Linear(self.output_dims, self.output_dims, bias=False),
                nn.Sigmoid()
            ))

            num_out_ch = self.output_dims

            if self.tr_mode != 'NoTr':
                encoder_layer = TransformerEncoderLayer(
                    d_model=num_out_ch,
                    nhead=4,
                    dim_feedforward=num_out_ch,
                    dropout=dp_value,
                    activation="relu",
                    normalize_before=False,
                )
                self.transformer_encoders.append(TransformerEncoder(encoder_layer = encoder_layer, num_layers = 1, norm = None))
                self.pos_embeddings.append(
                    nn.Parameter(
                        torch.zeros((grid_size ** 3, num_out_ch)).cuda()
                    )
                )

            self.cls_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch//2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))
            self.reg_weight_module.append(nn.Sequential(
                nn.Linear(num_out_ch, num_out_ch //2),
                nn.ReLU(),
                nn.Linear(num_out_ch//2, 1),
                nn.Sigmoid()
            ))

        self.attention_op = attention_op
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz_list, new_xyz_r_list, new_xyz_batch_cnt_list,
                features=None, batch_size=None, num_rois=None):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz_list: [(B, N x grid_size^3, 3)]
        :param new_xyz_r_list: [(B, N x grid_size^3, 1)]
        :param new_xyz_batch_cnt_list: (batch_size), N x grid_size^3
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        cls_features_list = []
        reg_features_list = []
        for i in range(self.num_pyramid_levels):
            new_xyz = new_xyz_list[i]
            new_xyz_r = new_xyz_r_list[i]
            new_xyz_batch_cnt = new_xyz_batch_cnt_list[i]
            new_xyz = new_xyz.view(-1, 3).contiguous()
            new_xyz_r = new_xyz_r.view(-1, 1).contiguous()

            grouped_xyz, grouped_features, empty_mask = self.groupers[i](
                xyz, xyz_batch_cnt, new_xyz, new_xyz_r, new_xyz_batch_cnt, features
            )

            pos_embedding = self.pos_proj[i](grouped_xyz)
            key_embedding = self.key_proj[i](grouped_features)
            value_embedding = self.value_proj[i](grouped_features)
            pos_key_embedding = pos_embedding * key_embedding

            v_coef = self.v_coef[i](pos_embedding.mean(2))
            q_coef = self.q_coef[i](pos_embedding.mean(2))
            k_coef = self.k_coef[i](key_embedding.mean(2))
            qk_coef = self.qk_coef[i](pos_key_embedding.mean(2))

            value_embedding = value_embedding + pos_embedding * v_coef.unsqueeze(2)
            attention_embedding = pos_embedding * q_coef.unsqueeze(2) + key_embedding * k_coef.unsqueeze(2) + pos_key_embedding * qk_coef.unsqueeze(2)

            attention_map = self.attention_proj[i](attention_embedding)
            attention_map = self.norm_layer[i](attention_map)
            # (N, num_heads, ns) -> (N, num_heads, head_dims, ns) -> (N, num_heads * head_dims, ns)
            attention_map = attention_map.unsqueeze(2).repeat(1, 1, self.head_dims, 1).reshape(attention_map.shape[0], -1, attention_map.shape[-1])

            attend_features = (attention_map * value_embedding).sum(-1)

            new_features = attend_features

            num_features = new_features.shape[1]
            new_features = new_features.reshape(batch_size * num_rois, -1, num_features)

            if self.tr_mode == 'NoTr':
                pass
            elif self.tr_mode == 'Normal':
                new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, new_features.shape[1], 1)  # (L, B, C)
                new_features = self.transformer_encoders[i](new_features, pos=pos_emb)
                new_features = new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
            elif self.tr_mode == 'Residual':
                tr_new_features = new_features.permute(1, 0, 2).contiguous()  # (L, B, C)
                pos_emb = self.pos_embeddings[i].unsqueeze(1).repeat(1, tr_new_features.shape[1], 1)  # (L, B, C)
                tr_new_features = self.transformer_encoders[i](tr_new_features, pos=pos_emb)
                tr_new_features = tr_new_features.permute(1, 0, 2).contiguous()  # (B, L, C)
                new_features = new_features + tr_new_features
            else:
                raise NotImplementedError

            cls_weights = self.cls_weight_module[i](new_features)  # (B, L, 1)
            cls_features = new_features * cls_weights
            cls_features_list.append(cls_features)

            reg_weights = self.reg_weight_module[i](new_features)  # (B, L, 1)
            reg_features = new_features * reg_weights
            reg_features_list.append(reg_features)

        cls_features = torch.cat(cls_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)
        reg_features = torch.cat(reg_features_list, dim=1)  # (B x N, \sum(grid_size^3), C)

        return cls_features.contiguous(), reg_features.contiguous()
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask = None,
                src_key_padding_mask = None,
                pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output