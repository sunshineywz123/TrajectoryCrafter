import numpy as np
import cv2
import PIL
from PIL import Image
import os
from datetime import datetime
import pdb
import torch.nn.functional as F
import numpy as np
import os
import cv2
import copy
from scipy.interpolate import UnivariateSpline, interp1d
import numpy as np
import PIL.Image
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Optional
import cv2
import PIL
import numpy
import skimage.io
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
import open3d as o3d
idx=0
world_points_idx=0
trans_world_idx=0
def tensor_to_point_cloud(tensor,colors=None, filename="point_cloud.ply"):
    """
    将 PyTorch 张量转换为 Open3D 点云并保存为 PLY 文件。

    Args:
        tensor (torch.Tensor): 形状为 [1, H, W, 3, 1] 的 PyTorch 张量。
        filename (str): 保存的文件名。
    """
    # 1. 移除维度为 1 的维度，并确保数据类型正确
    points = tensor.squeeze().float()

    # 2. 将 PyTorch 张量转换为 NumPy 数组
    points = points.cpu().numpy()

    # 3. 重塑数组，使得形状为 [N, 3]，其中 N 是点的数量
    points = points.reshape(-1, 3)

    # 4. 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. 保存点云到 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"点云已保存到 {filename}")
    return pcd

def read_video_frames(video_path, process_length, stride, max_res, dataset="open"):
    if dataset == "open":
        print("==> processing video: ", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
        # original_height, original_width = vid.get_batch([0]).shape[1:3]
        # height = round(original_height / 64) * 64
        # width = round(original_width / 64) * 64
        # if max(height, width) > max_res:
        #     scale = max_res / max(original_height, original_width)
        #     height = round(original_height * scale / 64) * 64
        #     width = round(original_width * scale / 64) * 64

        # FIXME: hard coded
        width = 1024
        height = 576

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0

    return frames


def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [
            np.array(Image.open(os.path.join(folder_name, path)))
            for folder_name, path in zip(folder, data)
        ]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(
        images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '10'}
    )


def sphere2pose(c2ws_input, theta, phi, r, device, x=None, y=None):
    c2ws = copy.deepcopy(c2ws_input)
    # c2ws[:,2, 3] = c2ws[:,2, 3] - radius

    # 先沿着世界坐标系z轴方向平移再旋转
    c2ws[:, 2, 3] -= r
    if x is not None:
        c2ws[:, 1, 3] += y
    if y is not None:
        c2ws[:, 0, 3] -= x

    theta = torch.deg2rad(torch.tensor(theta)).to(device)
    sin_value_x = torch.sin(theta)
    cos_value_x = torch.cos(theta)
    rot_mat_x = (
        torch.tensor(
            [
                [1, 0, 0, 0],
                [0, cos_value_x, -sin_value_x, 0],
                [0, sin_value_x, cos_value_x, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    phi = torch.deg2rad(torch.tensor(phi)).to(device)
    sin_value_y = torch.sin(phi)
    cos_value_y = torch.cos(phi)
    rot_mat_y = (
        torch.tensor(
            [
                [cos_value_y, 0, sin_value_y, 0],
                [0, 1, 0, 0],
                [-sin_value_y, 0, cos_value_y, 0],
                [0, 0, 0, 1],
            ]
        )
        .unsqueeze(0)
        .repeat(c2ws.shape[0], 1, 1)
        .to(device)
    )

    c2ws = torch.matmul(rot_mat_x, c2ws)
    c2ws = torch.matmul(rot_mat_y, c2ws)
    # c2ws[:,2, 3] = c2ws[:,2, 3] + radius
    return c2ws


def generate_traj_specified(c2ws_anchor, theta, phi, d_r, d_x, d_y, frame, device):
    # Initialize a camera.
    thetas = np.linspace(0, theta, frame)
    phis = np.linspace(0, phi, frame)
    rs = np.linspace(0, d_r, frame)
    xs = np.linspace(0, d_x, frame)
    ys = np.linspace(0, d_y, frame)
    c2ws_list = []
    for th, ph, r, x, y in zip(thetas, phis, rs, xs, ys):
        c2w_new = sphere2pose(
            c2ws_anchor,
            np.float32(th),
            np.float32(ph),
            np.float32(r),
            device,
            np.float32(x),
            np.float32(y),
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws


def txt_interpolation(input_list, n, mode='smooth'):
    x = np.linspace(0, 1, len(input_list))
    if mode == 'smooth':
        f = UnivariateSpline(x, input_list, k=3)
    elif mode == 'linear':
        f = interp1d(x, input_list)
    else:
        raise KeyError(f"Invalid txt interpolation mode: {mode}")
    xnew = np.linspace(0, 1, n)
    ynew = f(xnew)
    return ynew


def generate_traj_txt(c2ws_anchor, phi, theta, r, frame, device):
    # Initialize a camera.
    """
    The camera coordinate sysmte in COLMAP is right-down-forward
    Pytorch3D is left-up-forward
    """

    if len(phi) > 3:
        phis = txt_interpolation(phi, frame, mode='smooth')
        phis[0] = phi[0]
        phis[-1] = phi[-1]
    else:
        phis = txt_interpolation(phi, frame, mode='linear')

    if len(theta) > 3:
        thetas = txt_interpolation(theta, frame, mode='smooth')
        thetas[0] = theta[0]
        thetas[-1] = theta[-1]
    else:
        thetas = txt_interpolation(theta, frame, mode='linear')

    if len(r) > 3:
        rs = txt_interpolation(r, frame, mode='smooth')
        rs[0] = r[0]
        rs[-1] = r[-1]
    else:
        rs = txt_interpolation(r, frame, mode='linear')
    # rs = rs*c2ws_anchor[0,2,3].cpu().numpy()

    c2ws_list = []
    for th, ph, r in zip(thetas, phis, rs):
        c2w_new = sphere2pose(
            c2ws_anchor, np.float32(th), np.float32(ph), np.float32(r), device
        )
        c2ws_list.append(c2w_new)
    c2ws = torch.cat(c2ws_list, dim=0)
    return c2ws


class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        self.dtype = torch.float32
        return

    def forward_warp(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
        mask=False,
        twice=False,
        load_pcd=False,
        pcd_path=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device).to(self.dtype)
        mask1 = mask1.to(self.device).to(self.dtype)
        if len(depth1)!=0:
            assert depth1.shape == (b, 1, h, w)
            depth1 = depth1.to(self.device).to(self.dtype)
        transformation1 = transformation1.to(self.device).to(self.dtype)
        transformation2 = transformation2.to(self.device).to(self.dtype)
        intrinsic1 = intrinsic1.to(self.device).to(self.dtype)
        intrinsic2 = intrinsic2.to(self.device).to(self.dtype)

        # 计算变换后的3D点坐标

        global idx
        if load_pcd:
            # pcd = o3d.io.read_point_cloud('/nas/users/yuanweizhong/monst3r/my_data/airport/scene_pointcloud_{}.ply'.format(idx))
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd_points = np.asarray(pcd.points)
            trans_points1=torch.from_numpy(pcd_points).to(self.device).to(self.dtype).reshape(b,h,w,3,1)
            #trans_points1 = torch.matmul(intrinsic2,torch.from_numpy(pcd_points).to(self.device).to(self.dtype).reshape(b,h,w,3,1))
        #rgb 是frame1
            ones_4d = torch.ones(size=(b,h,w,1,1)).to(self.device).to(self.dtype)
            # 转换为齐次坐标
            trans_points1_homo = torch.cat([trans_points1, ones_4d], dim=3)  # (b, h, w, 4, 1)
            # trans_4d = transformation2
            trans_4d = torch.bmm(
                transformation2, torch.linalg.inv(transformation1)
            )  # (b, 4, 4)  
            # 应用变换矩阵
            trans_world_homo = torch.matmul(trans_4d, trans_points1_homo)  
            trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
            trans_points1 = torch.matmul(intrinsic2,trans_world)
        else:
            trans_points1 = self.compute_transformed_points(
                depth1, transformation1, transformation2, intrinsic1, intrinsic2,frame=(frame1+1)/2
            ) # (b, h, w, 3, 1)
        #rgb 是frame1
        #trans_points1 是frame1的3D点坐标
        #生成带颜色的点云
        
        colors = (frame1+1)/2
        # colors.shape (b,3,h,w)
        # pcd =tensor_to_point_cloud(torch.matmul(torch.linalg.inv(intrinsic2),trans_points1),colors=colors.permute(0,2,3,1).reshape(b*h*w,3),filename='tmp3/pcd_color_{:04d}.ply'.format(idx))
        idx+=1

        # import ipdb; ipdb.set_trace()
        # 将3D点投影到2D平面,进行透视除法
        trans_coordinates = (
            trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        )
        # 获取变换后的深度值
        trans_depth1 = trans_points1[:, :, :, 2, 0]
        # 创建网格坐标
        grid = self.create_grid(b, h, w).to(trans_coordinates)
        # 计算光流场,即坐标偏移量
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid
        if not twice:
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            if mask:
                warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)
            return warped_frame2, mask2, None, flow12

        else:
            # 使用双线性插值进行第一次变形，得到变形后的帧和遮罩
            warped_frame2, mask2 = self.bilinear_splatting(
                frame1, mask1, trans_depth1, flow12, None, is_image=True
            )
            # 注释掉的清理点代码
            # warped_frame2, mask2 = self.clean_points(warped_frame2, mask2)
            
            # 对光流场进行双线性插值变形,得到变形后的光流
            warped_flow, _ = self.bilinear_splatting(
                flow12, mask1, trans_depth1, flow12, None, is_image=False
            )
            
            # 使用变形后的帧和光流进行第二次变形,得到最终的变形帧
            # twice_warped_frame1, _ = self.bilinear_splatting(
            #     warped_frame2,
            #     mask2,
            #     depth1.squeeze(1),
            #     -warped_flow,
            #     None,
            #     is_image=True,
            # )

            twice_warped_frame1, mask3 = self.bilinear_splatting(
                warped_frame2,
                mask2,
                depth1.squeeze(1),
                -warped_flow,
                None,
                is_image=True,
            )
            # import ipdb; ipdb.set_trace()
            # return twice_warped_frame1, warped_frame2, None, None
            return twice_warped_frame1, mask3, None, None

    def compute_transformed_points(
        self,
        depth1: torch.Tensor,
        transformation1: torch.Tensor,
        transformation2: torch.Tensor,
        intrinsic1: torch.Tensor,
        intrinsic2: Optional[torch.Tensor],
        frame: Optional[torch.Tensor] = None,
    ):
        """
        Computes transformed position for each pixel location
        """
        # 如果设置了分辨率,检查深度图的尺寸是否匹配
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        # 获取深度图的尺寸信息
        b, _, h, w = depth1.shape
        # 如果没有提供intrinsic2,使用intrinsic1的副本
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        # 计算从transformation1到transformation2的变换矩阵
        transformation = torch.bmm(
            transformation2, torch.linalg.inv(transformation1)
        )  # (b, 4, 4)

        # 创建x方向的一维坐标数组
        x1d = torch.arange(0, w)[None]
        # 创建y方向的一维坐标数组
        y1d = torch.arange(0, h)[:, None]
        # 将x坐标扩展为二维网格
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        # 将y坐标扩展为二维网格
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        # 创建全1的二维数组
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        # 扩展ones_2d为四维张量
        ones_4d = ones_2d[None, :, :, None, None].repeat(
            [b, 1, 1, 1, 1]
        )  # (b, h, w, 1, 1)
        # 将x,y坐标和1堆叠成齐次坐标
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[
            None, :, :, :, None
        ]  # (1, h, w, 3, 1)

        # 计算内参矩阵的逆矩阵
        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        # 扩展内参逆矩阵为四维
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        # 扩展内参矩阵为四维
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        # 调整深度图的维度
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        # 扩展变换矩阵为四维
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        # 计算未归一化的位置向量
        unnormalized_pos = torch.matmul(
            intrinsic1_inv_4d, pos_vectors_homo
        )  # (b, h, w, 3, 1)
        # 计算世界坐标系中的点
        #world_points的含义camera 0的坐标系下的坐标
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        # global world_points_idx
        # tensor_to_point_cloud(world_points.reshape(-1,3),colors=frame.permute(0,2,3,1).reshape(-1,3),filename='tmp3/world_points_{:04d}.ply'.format(world_points_idx))
        # world_points_idx+=1
        # 转换为齐次坐标
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        # 应用变换矩阵
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        # 提取变换后的3D坐标
        #trans_world的含义camera 1的坐标系下的坐标
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        # global trans_world_idx
        
        # tensor_to_point_cloud(trans_world.reshape(-1,3),colors=frame.permute(0,2,3,1).reshape(-1,3),filename='tmp3/trans_world_points_{:04d}.ply'.format(trans_world_idx))
        # trans_world_idx+=1
        # 计算变换后的归一化坐标
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(
        self,
        frame1: torch.Tensor,
        mask1: Optional[torch.Tensor],
        depth1: torch.Tensor,
        flow12: torch.Tensor,
        flow12_mask: Optional[torch.Tensor],
        is_image: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用双线性插值进行图像扭曲变换。

        Args:
            frame1 (torch.Tensor): 输入图像1，形状为 (b, c, h, w)。
            mask1 (Optional[torch.Tensor]): 输入图像1的掩码，形状为 (b, 1, h, w)，其中1表示已知，0表示未知。默认为None。
            depth1 (torch.Tensor): 输入图像1的深度图，形状为 (b, 1, h, w)。
            flow12 (torch.Tensor): 从图像1到图像2的光流场，形状为 (b, 2, h, w)。
            flow12_mask (Optional[torch.Tensor]): 光流场的掩码，形状为 (b, 1, h, w)，其中1表示有效光流，0表示无效光流。默认为None。
            is_image (bool): 如果为True，则输出将被裁剪到 (-1, 1) 范围。默认为False。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - warped_frame2 (torch.Tensor): 扭曲后的图像2，形状为 (b, c, h, w)。
                - mask2 (torch.Tensor): 扭曲后的图像2的掩码，形状为 (b, 1, h, w)，其中1表示已知，0表示未知。

        """
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        # 获取输入图像的尺寸信息
        b, c, h, w = frame1.shape
        # 如果没有提供mask1,创建全1的掩码
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        # 如果没有提供flow12_mask,创建全1的掩码
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        # 创建网格并将其移动到与frame1相同的设备上
        grid = self.create_grid(b, h, w).to(frame1)
        # 计算变换后的位置
        trans_pos = flow12 + grid

        # 将变换位置偏移1个单位
        trans_pos_offset = trans_pos + 1
        # 计算变换位置的下界(向下取整)
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        # 计算变换位置的上界(向上取整)
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        # 将偏移位置限制在有效范围内
        trans_pos_offset = torch.stack(
            [
                torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        # 将下界位置限制在有效范围内
        trans_pos_floor = torch.stack(
            [
                torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )
        # 将上界位置限制在有效范围内
        trans_pos_ceil = torch.stack(
            [
                torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
                torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1),
            ],
            dim=1,
        )

        # 计算西北方向的权重
        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        # 计算西南方向的权重
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1])
        )
        # 计算东北方向的权重
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )
        # 计算东南方向的权重
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * (
            1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1])
        )

        # 限制深度值的范围
        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        # 对深度值进行对数变换
        log_depth1 = torch.log(1 + sat_depth1)
        # 计算深度权重
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        # 计算西北方向的最终权重
        weight_nw = torch.moveaxis(
            prox_weight_nw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        # 计算西南方向的最终权重
        weight_sw = torch.moveaxis(
            prox_weight_sw * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        # 计算东北方向的最终权重
        weight_ne = torch.moveaxis(
            prox_weight_ne * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )
        # 计算东南方向的最终权重
        weight_se = torch.moveaxis(
            prox_weight_se * mask1 * flow12_mask / depth_weights.unsqueeze(1),
            [0, 1, 2, 3],
            [0, 3, 1, 2],
        )

        # 创建用于存储变形后帧的零张量
        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(
            frame1
        )
        # 创建用于存储变形权重的零张量
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(
            frame1
        )

        # 调整frame1的轴顺序
        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        # 创建批次索引
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        # 将西北方向的值累加到结果中
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_nw,
            accumulate=True,
        )
        # 将西南方向的值累加到结果中
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            frame1_cl * weight_sw,
            accumulate=True,
        )
        # 将东北方向的值累加到结果中
        warped_frame.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_ne,
            accumulate=True,
        )
        # 将东南方向的值累加到结果中
        warped_frame.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            frame1_cl * weight_se,
            accumulate=True,
        )

        # 累加西北方向的权重
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
            weight_nw,
            accumulate=True,
        )
        # 累加西南方向的权重
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
            weight_sw,
            accumulate=True,
        )
        # 累加东北方向的权重
        warped_weights.index_put_(
            (batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
            weight_ne,
            accumulate=True,
        )
        # 累加东南方向的权重
        warped_weights.index_put_(
            (batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
            weight_se,
            accumulate=True,
        )

        # 调整变形后帧的轴顺序
        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        # 调整权重的轴顺序
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        # 裁剪变形后的帧，去除边界
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        # 裁剪权重，去除边界
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        # 创建掩码，标记有效区域
        mask = cropped_weights > 0
        # 根据是否为图像设置填充值
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        # 计算最终的变形帧
        warped_frame2 = torch.where(
            mask, cropped_warped_frame / cropped_weights, zero_tensor
        )
        # 转换掩码格式
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=-1, max=1)
        return warped_frame2, mask2

    def clean_points(self, warped_frame2, mask2):
        warped_frame2 = (warped_frame2 + 1.0) / 2.0
        mask = 1 - mask2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0) * 255.0
        mask = mask.cpu().numpy()
        kernel = numpy.ones((5, 5), numpy.uint8)
        mask_erosion = cv2.dilate(numpy.array(mask), kernel, iterations=1)
        mask_erosion = PIL.Image.fromarray(numpy.uint8(mask_erosion))
        mask_erosion_ = numpy.array(mask_erosion) / 255.0
        mask_erosion_[mask_erosion_ < 0.5] = 0
        mask_erosion_[mask_erosion_ >= 0.5] = 1
        mask_new = (
            torch.from_numpy(mask_erosion_)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        warped_frame2 = warped_frame2 * (1 - mask_new)
        return warped_frame2 * 2.0 - 1.0, 1 - mask_new[:, 0:1, :, :]

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def read_image(path: Path) -> torch.Tensor:
        image = skimage.io.imread(path.as_posix())
        return image

    @staticmethod
    def read_depth(path: Path) -> torch.Tensor:
        if path.suffix == '.png':
            depth = skimage.io.imread(path.as_posix())
        elif path.suffix == '.npy':
            depth = numpy.load(path.as_posix())
        elif path.suffix == '.npz':
            with numpy.load(path.as_posix()) as depth_data:
                depth = depth_data['depth']
        else:
            raise RuntimeError(f'Unknown depth format: {path.suffix}')
        return depth

    @staticmethod
    def camera_intrinsic_transform(
        capture_width=1920, capture_height=1080, patch_start_point: tuple = (0, 0)
    ):
        start_y, start_x = patch_start_point
        camera_intrinsics = numpy.eye(4)
        camera_intrinsics[0, 0] = 2100
        camera_intrinsics[0, 2] = capture_width / 2.0 - start_x
        camera_intrinsics[1, 1] = 2100
        camera_intrinsics[1, 2] = capture_height / 2.0 - start_y
        return camera_intrinsics

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device
