import logging
import os
import pickle
import random
import shutil
import subprocess
import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

#用于检查输入参数x是否为Numpy数组类型，并将其转换为pytorch的张量类型
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False
    #np.array([1,2,3]) => (tensor([1., 2., 3.]), True)

#用于将输入值val限制在指定的周期范围内
def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans

#根据给定的name值从字典info中删除特定的条目，并返回过滤后的字典info
def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info

#用于沿着Z轴旋转点云数据中的点，返回旋转后的点云数据
def rotate_points_along_z(points, angle):
    #B表示批次大小，即点云数据中的样本数，N表示每个样本中点的数量，3+C表示每个点的特征维度，3是xyz，C是其他特征向量（颜色/法向量等）
    #例：points = np.array([
    #     [[1, 2, 3, 4], [5, 6, 7, 8]],
    #     [[9, 10, 11, 12], [13, 14, 15, 16]]])
    # 其中，形状为（2，2，4）；shape[0]表示里面二维数组的总数，shape[1]表示单个二维数组的行数，shape[2]表示里面二维数组的列数
    #另解：shape[0]表示最外围的数组的维数，shape[1]表示次外围的数组的维数，数字不断增大，维数由外到内
    """
    Args:
        points: (B, N, 3 + C) 
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    #使用 torch.stack 函数将九个张量沿着第一个维度（行）进行堆叠。这将创建一个形状为 (B, 9) 的张量;
    #使用 view 函数将形状为 (B, 9) 的张量重新调整为形状为 (-1, 3, 3) 的张量。
    #这里的 -1 表示根据其他维度的大小自动推断，以使总元素数保持不变。因此，这一步将生成一个形状为 (B, 3, 3) 的张量
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    #将点云数据中的点坐标 points[:, :, 0:3] 与旋转矩阵 rot_matrix 相乘，得到旋转后的点坐标 points_rot
    #points[:, :, 0:3] 表示选取 points 数组中的前三列，即点的三维坐标（x、y、z）。
    # 这样选择之后，points[:, :, 0:3] 的形状为 (B, N, 3)，其中 B 是样本数，N 是每个样本中的点数
    #rot_matrix 是旋转矩阵，形状为 (B, 3, 3)
    #根据矩阵相乘的规则，points[:, :, 0:3] 的最后一个维度大小为 3，与 rot_matrix 的倒数第二个维度大小相匹配，因此可以进行相乘操作
    #结果为旋转后的点坐标 points_rot，形状与 points[:, :, 0:3] 保持一致，即 (B, N, 3)
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)#切片操作，【start,stop,step】
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)#将前三个特征维度与第三列之后的特征维度进行拼接
    return points_rot.numpy() if is_numpy else points_rot

#根据给定的绕 z 轴旋转的角度，计算并返回对应的旋转矩阵
def angle2matrix(angle):
    """
    Args:
        angle: angle along z-axis, angle increases x ==> y
    Returns:
        rot_matrix: (3x3 Tensor) rotation matrix
    """

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.tensor([
        [cosa, -sina, 0],
        [sina, cosa,  0],
        [   0,    0,  1]
    ])
    return rot_matrix

#用于根据给定的范围对点云数据进行掩码处理，生成一个布尔型掩码，表示哪些点在给定的范围内
def mask_points_by_range(points, limit_range):
    #返回的mask是一个张量，如【false,true,false】
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask

#用于计算体素中心的坐标
def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)体素坐标，点数量N
        downsample_times:下采样倍数
        voxel_size:体素大小
        point_cloud_range:点云范围，包含六个元素【xmin,ymin,zmin,xmax,ymax,zmax】

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz) 将 voxel_coords 的坐标顺序调整为 (z, y, x)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

#set_random_seed 函数为 Python、NumPy 和 PyTorch 库中的随机数生成器设置了相同的种子，
#以确保在相同种子下运行时生成的随机数序列是可重复的。此函数在需要保证实验或训练的可重现性时非常有用
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True#将 PyTorch 库中的 cuDNN 库的随机性设置为确定性模式
    torch.backends.cudnn.benchmark = False#将 PyTorch 库中的 cuDNN 库的自动调优模式设置为关闭，以确保在使用 cuDNN 加速时的一致性

#为每个工作器设置了与基准种子值和工作器标识相关的随机种子
def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)

# 函数计算出了使用 np.pad 函数进行填充时的填充参数，以便将输入数据填充到期望的大小。填充参数是一个元组，指定了在数据的前面和后面分别填充多少个值
def get_pad_params(desired_size, cur_size):
    #desired_size期望的填充后的输出大小; cur_size当前的大小。应始终小于或等于 desired_size
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)#表示在数据前面不填充，而在数据后面填充diff个值

    return pad_params

#用于根据给定的目标名称列表和使用的类别列表，筛选出符合条件的目标索引
def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

#为使用Slurm作业调度系统进行分布式训练的代码提供必要的初始化操作
def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    #tcp_port表示TCP端口号;local_rank表示本地进程的排名
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank

#这个函数的目的是为使用基于PyTorch的分布式训练的代码提供必要的初始化操作
def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank

#用于获取当前分布式训练环境的信息。它会返回当前进程的排名（rank）、总进程数（world_size）以及每台机器上的GPU数量
def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size

#用于在分布式环境中合并多个结果，并按特定顺序返回合并后的结果列表
def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results

#根据给定的索引将点数据散布到指定形状的张量中，以便在一些计算任务中进行数据重排或数据转换
def scatter_point_inds(indices, point_inds, shape):
    #indices参数表示用于指定散布位置的索引张量。
    #point_inds参数表示要散布的点数据张量。
    #shape参数表示目标张量的形状
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret

#根据稀疏张量生成体素到点索引的映射张量
def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device#获取稀疏张量的设备信息和批次大小
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape#获取稀疏张量的空间形状（体素的维度）
    indices = sparse_tensor.indices.long()#将稀疏张量的索引转换为长整型
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)#创建一个与索引张量形状相同的点索引张量，用于表示点在稀疏张量中的位置
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor

#创建共享内存数组
def sa_create(name, var):
    #name:共享内存数组的名称，var：要创建共享内存数组的原始数组
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x
    #共享内存数组（SharedArray）是一种在多进程或多线程之间共享数据的机制。它允许多个进程或线程直接访问相同的内存块，而无需进行数据的复制或传输

#用于计算和存储平均值和当前值
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    #AverageMeter 类通常用于在训练或评估过程中跟踪损失函数或指标的平均值和当前值。它提供了一种方便的方式来计算和存储这些值，并能够在每个步骤更新并计算平均值