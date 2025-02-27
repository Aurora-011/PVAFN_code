import os

import torch
import torch.nn as nn
import numpy as np
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils


class Detector3DTemplate(nn.Module):
    #模型配置信息，目标类别的数量，数据集对象
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())#创建一个缓冲区，用于追踪全局训练步数

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]#拓扑信息；通过这个顺序调用各个模型
        #如果列表中的模块是类的成员方法，你可以使用点操作符调用它们。例如，如果 dense_head 是一个类的成员方法，你可以通过 self.dense_head.method_name() 的方式调用其中的方法

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    #根据模型的模块拓扑结构，依次构建各个模块，并将其添加到模型中。
    #在构建过程中，根据需要的模块和数据集的相关信息，逐步更新 model_info_dict 字典。最后，返回模型的模块列表
    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,#原始点特征数量
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,#点特征数量/通道数
            'grid_size': self.dataset.grid_size,#网格大小
            'point_cloud_range': self.dataset.point_cloud_range,#点云范围
            'voxel_size': self.dataset.voxel_size,#体素大小
            'depth_downsample_factor': self.dataset.depth_downsample_factor#深度下采样因子
        }
        #根据拓扑信息建立网络，并保存在module_list中
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    #VFE体素特征编码
    #这段代码的作用是根据模型配置中的 VFE 设置，创建 VFE 模块对象，并更新 model_info_dict 中的相关信息。
    #最后，返回 VFE 模块对象和更新后的 model_info_dict。如果模型配置中没有定义 VFE 模块，则返回 None 和原始的 model_info_dict
    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        #如果基础配置中有vfe的话，构建VFE模块对象
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        #获取vfe模块的输出特征维度，并更新model_info_dict['num_point_features']
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    #构建3D特征提取的骨干网络
    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],#输入通道数（即之前构建的VFE模块输出的特征维度）
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features#被更新为网络输出维度
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None#如果channels属性不存在则赋值为none
        return backbone_3d_module, model_info_dict

    #构建特征映射到鸟瞰图模型
    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    #构建2D骨干网络
    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict.get('num_bev_features', None)
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    #PFE点云特征提取编码模块
    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict
    
    #构建密集头模块
    #负责在网络中生成密集的预测结果，通常接收来自其他层的特征表示，将其转换为目标的位置、类别、置信度等预测结果
    #输出是一个密集的特征图或预测图，每个位置都对应着输入图像或特征图的一个区域，并包含了该区域内目标的预测信息
    #用途：密集头通常用于生成全局的目标预测结果，覆盖整个输入图像或特征图，用于全局目标检测任务
    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'] if 'num_bev_features' in model_info_dict else self.model_cfg.DENSE_HEAD.INPUT_FEATURES,
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    #构建点头模块
    #负责在点云数据中生成目标的预测结果。接收点云数据作为输入，对点云进行处理和预测
    #输出是每个点的目标位置、类别、置信度等预测结果，它能够对点云中的每个点进行分类、定位和预测
    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict
        #如果USE_POINT_FEATURES_BEFORE_FUSION为True，就将其赋值给input_channels，False则将num_point_features赋值给input_channels
        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            #如果模型配置中的点头设置 CLASS_AGNOSTIC 为 False，则 num_class 取 self.num_class 的值，表示目标类别的数量。
            #如果 CLASS_AGNOSTIC 为 True，则 num_class 取 1，表示模型将对目标进行类别不可知的预测
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
            #如果模型配置中的 ROI_HEAD 设置为 True，则 predict_boxes_when_training 取 True，表示在训练时预测边界框，反之不预测边界框
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    #负责在候选区域中生成目标的预测结果
    #接收来自区域生成模块（如RPN）的候选区域（通常是边界框），对这些区域进行处理和预测
    #通过对候选区域进行进一步的分类、定位和预测，提高目标检测的准确性和效率
    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            #以下为修改内容，还原请去掉注释即可
            #backbone_channels= model_info_dict.get('backbone_channels', None),
            #point_cloud_range=model_info_dict['point_cloud_range'],
            #voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    #前向传播，在各个子类中定义
    def forward(self, **kwargs):
        raise NotImplementedError

    #后处理方法用于将模型的输出转换为更具有可读性或可解释性的形式，或者对预测结果进行进一步过滤、修正或筛选，以满足特定的需求或任务要求
    def post_processing(self, batch_dict):
        #batch_cls_preds表示目标类别预测结果，三种：表示每个批次的每个边界框的类别预测结果（可能是多类别）or 
        #表示每个批次的每个边界框的类别预测结果（可能是多类别）的列表 or 表示每个批次的每个边界框的类别预测结果的列表，其中每个元素对应一个类别的预测结果
        #multihead_label_mapping 一个列表，包含每个类别预测结果的映射关系。每个元素是一个列表，表示对应类别的映射关系
        #batch_box_preds边界框预测结果 表示每个批次的每个边界框的预测结果（包括边界框的位置和其他附加信息） or
        #表示每个批次的每个边界框的预测结果（包括边界框的位置和其他附加信息）的列表
        #cls_preds_normalized 布尔值，表示 batch_cls_preds 是否已经归一化
        #batch_index: 可选参数，表示每个样本所属的批次索引
        #has_class_labels: 布尔值，表示输入数据是否包含类别标签
        #roi_labels: 边界框的类别标签，形状为 (B, num_rois)，取值范围为 1 到 num_classes
        #batch_pred_labels: 边界框的预测标签，形状为 (B, num_boxes, 1)
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            #如果 batch_dict 中存在 batch_index，则根据 batch_index 筛选出当前样本对应的边界框预测结果，否则根据 index 进行筛选
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            #获取边界框预测结果 box_preds，并将其保存到 src_box_preds 中
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            #如果 batch_cls_preds 是列表类型，则表示有多个类别的预测结果，分别获取每个类别的预测结果，
            #并保存到 cls_preds 和 src_cls_preds 中。如果 cls_preds 未经过归一化，则对每个类别的预测结果分别进行 sigmoid 函数处理
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]#检查cls_preds 的形状的第二个维度是否等于 1 或者等于 self.num_class

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            #如果 batch_cls_preds 是列表类型，则表示有多个类别的预测结果，
            # 分别获取每个类别的预测结果，并保存到 cls_preds 和 src_cls_preds 中。
            # 如果 cls_preds 未经过归一化，则对每个类别的预测结果分别进行 sigmoid 函数处理
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            #判断是否使用NMS极大值抑制，如果配置：
            #进行多类别的 NMS 过程。根据 cls_preds 和 box_preds 进行多类别的 NMS，
            # 得到预测分数、预测标签和预测框。若配置中指定输出原始得分，则使用原始的类别预测结果计算得分。
            # 最终将多个类别的预测结果合并为一个结果，得到最终的预测分数 final_scores、预测标签
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            #如果没有配置NMS：
            #进行类别预测结果的后处理。首先，根据类别预测结果 cls_preds 获取最大得分和对应的标签。
            # 如果输入数据中存在类别标签，则使用标签进行替换，否则将标签加一作为预测标签。
            # 然后，根据最大得分和边界框预测结果 box_preds 进行类别无关的非极大值抑制（NMS），得到选中的边界框索引和对应的得分。
            # 如果配置中指定输出原始得分，则使用原始的类别预测结果计算得分。最终将选中的边界框的得分、
            # 标签和框保存为最终的预测分数 final_scores、预测标签 final_labels 和预测框 final_boxes
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1 
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            #根据阈值列表和预测框信息，更新召回率字典 recall_dict        
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )        

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    #这段代码的作用是计算边界框预测结果与真实边界框之间的召回率，根据阈值列表统计满足条件的预测结果数量，并更新召回率字典
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict
        #从 data_dict 中获取当前批次的边界框预测结果 rois 和真实边界框 gt_boxes
        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]
        #如果召回率字典 recall_dict 为空，说明是第一次调用该方法，需要初始化召回率字典。
        #首先创建一个键为 'gt' 值为 0 的项，并根据阈值列表 thresh_list 创建相应的 'roi_<thresh>' 和 'rcnn_<thresh>' 的项，初始值均为 0
        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        #对真实边界框 gt_boxes 进行处理，去除没有目标的无效边界框，得到有效的边界框
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1 #k为索引总数
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1#当k大于0且当前cur_gt[k] 中所有元素和为0时，代表该边界框是一个空的边界框，则总数减1，继续往下遍历
        cur_gt = cur_gt[:k + 1]#得到有效边界框，范围是0到k+1

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                #计算box_preds和cur_gt的IoU信息（重叠度）
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])
            #遍历阈值列表 thresh_list，对每个阈值进行召回率计算
            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    #该方法用于根据输入的模型状态字典 model_state_disk 更新模型的状态字典，并根据需要对 spconv 模块的权重形状进行适应性调整。
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            #如果键 key 在 spconv_keys 中，并且同时存在于 state_dict 中，且其形状与 val 的形状不同，说明需要对 spconv 模块的权重形状进行调整
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:#这意味着只加载 update_model_state 中出现的键值对，而不会保留原模型状态字典中未出现的键。
            self.load_state_dict(update_model_state)
        else:#这样做可以保留原模型状态字典中未出现的键，并更新那些在 update_model_state 中出现的键的对应值
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    #从文件中加载模型参数，并根据需要将参数加载到CPU或GPU上
    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        if not pre_trained_path is None:#如果有，则加载额外的预训练检查点
            pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
            pretrain_model_state_disk = pretrain_checkpoint['model_state']
            model_state_disk.update(pretrain_model_state_disk)
            
        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    #从文件中加载模型参数和优化器状态
    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
