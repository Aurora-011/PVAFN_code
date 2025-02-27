# PVAFN: Point-Voxel Attention Fusion Network with Multi-Pooling Enhancing

## Detection Framework
The overall detection framework is shown below.
(1) Point-Voxel Attention Fusion Module (PVAFM);
(2) Multi-Pooling Enhancement Module (MPE). 
PVAFM combines self-attention and point-voxel attention modules to adaptively combine point features with voxel-BEV fusion features for set abstraction.
MPE combines the RoI clustering pooling head and the RoI pyramid pooling head for proposal refinement.

![](./tools/images/figure2.png)

## Getting Started

```
conda create -n spconv2 python=3.8
conda activate spconv2
conda install cudatoolkit=11.3.1 cudnn=8.2 pytorch=1.10 torchvision=0.11
pip install numpy==1.24.4 protobuf==3.19.6 scikit-image==0.19.2 waymo-open-dataset-tf-2-5-0 spconv-cu113 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython prefetch-generator
```

### Dependency
Our released implementation is tested on.
+ Ubuntu 18.04
+ Python 3.8.18
+ PyTorch 1.11.0
+ Spconv 1.2.1
+ NVIDIA CUDA 11.3
+ 4x Tesla V100 GPUs

### Prepare dataset

Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
PVAFN
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

### Installation

```
git clone https://github.com/hailanyi/PVAFN.git
cd PVAFN
python3 setup.py develop
```

### Training

Single GPU train:
```
cd tools
python train.py --cfg_file cfgs/kitti_models/pvafn.yaml
```
Multiple GPU train: 

You can modify the gpu number in the dist_train.sh and run
```
cd tools
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --cfg_file cfgs/kitti_models/pvafn.yaml --launcher pytorch
```

### Evaluation

```
cd tools
python3 test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

## License

This code is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [PV-RCNN](https://github.com/sshaoshuai/PV-RCNN), some codes are from [Pyramid-RCNN](https://github.com/PointsCoder/Pyramid-RCNN)

[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

[PV-RCNN](https://github.com/sshaoshuai/PV-RCNN)

[Pyramid-RCNN](https://github.com/PointsCoder/Pyramid-RCNN)