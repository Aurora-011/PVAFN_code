import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
    #原版
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
    
    #新版
    # def forward(self, batch_dict):
    #     """
    #     Args:
    #         batch_dict:
    #             encoded_spconv_tensor: sparse tensor
    #     Returns:
    #         batch_dict:
    #             spatial_features:

    #     """
    #     encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
    #     encoded_spconv_tensor4x = batch_dict['encoded_spconv_tensor4x']
    #     spatial_features = encoded_spconv_tensor.dense()
    #     spatial_features4x = encoded_spconv_tensor4x.dense()
    #     N, C, D, H, W = spatial_features.shape
    #     N4, C4, D4, H4, W4 = spatial_features4x.shape
    #     spatial_features = spatial_features.view(N, C * D, H, W)
    #     spatial_features4x = spatial_features.view(N4, C4 * D4, H4, W4)
    #     batch_dict['spatial_features'] = spatial_features
    #     batch_dict['spatial_features4x'] = spatial_features4x
    #     batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
    #     batch_dict['spatial_features_stride4x'] = batch_dict['encoded_spconv_tensor_stride4x']
    #     return batch_dict