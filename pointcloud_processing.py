import numpy as np
import open3d as o3d
import torch

def depth_map_to_point_cloud(depth_image, intrinsic_matrix):
    """
    将深度图转换为点云，支持批处理。

    参数:
    depth_image (torch.Tensor): 深度图，形状为(B,1,H,W)的张量。
    intrinsic_matrix (torch.Tensor): 相机内参矩阵，形状为(B,3,3)的张量。

    返回值:
    torch.Tensor: 点云，形状为(B,H*W,3)的张量。
    """
    # 确保输入是torch.Tensor
    if isinstance(depth_image, np.ndarray):
        depth_image = torch.from_numpy(depth_image)
    if isinstance(intrinsic_matrix, np.ndarray):
        intrinsic_matrix = torch.from_numpy(intrinsic_matrix)
    
    batch_size, _, height, width = depth_image.shape
    device = depth_image.device
    
    # 创建批量网格点
    y_grid, x_grid = torch.meshgrid(torch.arange(height, device=device), 
                                    torch.arange(width, device=device), 
                                    indexing='ij')
    
    # 扩展为批量形式
    x_grid = x_grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B,H,W)
    y_grid = y_grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B,H,W)
    
    points_list = []
    
    for b in range(batch_size):
        fx = intrinsic_matrix[b, 0, 0]
        fy = intrinsic_matrix[b, 1, 1]
        cx = intrinsic_matrix[b, 0, 2]
        cy = intrinsic_matrix[b, 1, 2]
        
        z = depth_image[b, 0]  # (H,W)
        x = (x_grid[b] - cx) / fx * z  # (H,W)
        y = (y_grid[b] - cy) / fy * z  # (H,W)
        
        # 堆叠为点云
        points = torch.stack([x, y, z], dim=-1)  # (H,W,3)
        points = points.reshape(-1, 3)  # (H*W,3)
        points_list.append(points)
    
    return torch.stack(points_list)  # (B,H*W,3)


def mask_point_cloud(point_cloud, mask):
    """
    根据mask过滤点云，支持批处理。

    参数:
    point_cloud (torch.Tensor): 点云，形状为(B,H*W,3)的张量。
    mask (torch.Tensor): mask，形状为(B,1,H,W)的张量。

    返回值:
    list: 过滤后的点云列表，每个元素是一个形状为(N_i,3)的张量，其中N_i是每个样本中有效点的数量。
    """
    # 确保输入是torch.Tensor
    if isinstance(point_cloud, np.ndarray):
        point_cloud = torch.from_numpy(point_cloud)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    batch_size = point_cloud.shape[0]
    mask_flat = mask.reshape(batch_size, -1)  # (B,H*W)
    
    masked_points_list = []
    for b in range(batch_size):
        # 获取当前样本的有效点
        valid_points = point_cloud[b][mask_flat[b]]  # (N_i,3)
        masked_points_list.append(valid_points)
    
    return masked_points_list


def calculate_bounding_box_center(masked_point_clouds):
    """
    计算点云的3D边界框中心，支持批处理。

    参数:
    masked_point_clouds (list): 点云列表，每个元素是一个形状为(N_i,3)的张量。

    返回值:
    torch.Tensor: 边界框的中心点坐标，形状为(B,3)的张量。
    """
    batch_size = len(masked_point_clouds)
    centers = []
    
    for b in range(batch_size):
        points = masked_point_clouds[b]
        
        # 处理空点云的情况
        if len(points) == 0:
            centers.append(torch.zeros(3, device=points.device))
            continue
        
        # 转换为numpy进行Open3D处理
        points_np = points.detach().cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        bounding_box = pcd.get_axis_aligned_bounding_box()
        center = bounding_box.get_center()
        
        # 转回torch张量
        center_tensor = torch.tensor(center, device=points.device)
        centers.append(center_tensor)
    
    return torch.stack(centers)  # (B,3)


def process_point_cloud(depth_image, intrinsic_matrix, mask):
    """
    从深度图和内参矩阵处理点云数据，并返回3D Bounding Box的中心点，支持批处理。

    参数:
        depth_image (torch.Tensor): 深度图像数据，形状为(B,1,H,W)的张量。
        intrinsic_matrix (torch.Tensor): 内参矩阵，形状为(B,3,3)的张量。
        mask (torch.Tensor): 掩码图像数据，形状为(B,1,H,W)的张量。

    返回值:
        torch.Tensor: 3D Bounding Box的中心点坐标，形状为(B,3)的张量。
    """
    # 步骤1: 从深度图反投影得到点云
    point_cloud = depth_map_to_point_cloud(depth_image, intrinsic_matrix)  # (B,H*W,3)

    # 步骤2: 应用Mask
    masked_point_clouds = mask_point_cloud(point_cloud, mask)  # list of (N_i,3)

    # 步骤3: 计算3D Bounding Box并获取中心点
    centers = calculate_bounding_box_center(masked_point_clouds)  # (B,3)

    return centers


# 示例
if __name__ == '__main__':
    # 批处理示例数据
    batch_size = 2
    height, width = 3, 3
    
    # 创建批量深度图
    depth_image = torch.tensor([[[[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]],
                               
                               [[[2, 3, 4],
                                 [5, 6, 7],
                                 [8, 9, 10]]]], dtype=torch.float32)  # (2,1,3,3)
    
    # 创建批量内参矩阵
    intrinsic_matrix = torch.tensor([[[500, 0, 160],
                                     [0, 500, 120],
                                     [0, 0, 1]],
                                    
                                    [[510, 0, 165],
                                     [0, 510, 125],
                                     [0, 0, 1]]], dtype=torch.float64)  # (2,3,3)
    
    # 创建批量掩码
    mask = torch.tensor([[[[True, False, True],
                          [False, True, False],
                          [True, False, True]]],
                        
                        [[[False, True, False],
                          [True, False, True],
                          [False, True, False]]]], dtype=torch.bool)  # (2,1,3,3)
    
    centers = process_point_cloud(depth_image, intrinsic_matrix, mask)
    
    print("\n3D Bounding Box 中心:\n", centers)