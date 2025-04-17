import numpy as np
import torch
from scipy.linalg import expm
import cv2

def calculate_rotation_matrix(a, b):
    """
    计算将向量 b 旋转到向量 a 的旋转矩阵 (使用罗德里格斯公式).

    参数:
    a (numpy.ndarray): 目标向量.
    b (numpy.ndarray): 原始向量.

    返回值:
    numpy.ndarray: 旋转矩阵.
    """
    # 归一化向量
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    # 计算旋转轴
    v = np.cross(b, a)
    s = np.linalg.norm(v)
    c = np.dot(b, a)

    # 构造旋转矩阵
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix


def transform_point(point, rotation_matrix):
    """
    使用旋转矩阵变换点.

    参数:
    point (numpy.ndarray): 要变换的点.
    rotation_matrix (numpy.ndarray): 旋转矩阵.

    返回值:
    numpy.ndarray: 变换后的点.
    """
    transformed_point = np.dot(rotation_matrix, point)
    return transformed_point


def calculate_camera_transformation(enlarged_masks_center, image_center, K):
    """
    计算相机对应旋转矩阵，将 OB 变换到 OA，A 点变换为相机图像中心.

    参数:
    enlarged_masks_center (numpy.ndarray): enlarged_masks 的中心点坐标，形状为 (N, 2).
    image_center (numpy.ndarray): 图像的中心点坐标，形状为 (N, 2).
    K (torch.Tensor): 相机内参矩阵，形状为 (N, 3, 3).

    返回值:
    tuple: 包含旋转矩阵和变换后的 A 点的元组，每个元素都是长度为 N 的列表.
    """
    # 获取样本数量
    num_samples = enlarged_masks_center.shape[0]
    
    # 初始化结果列表
    rotation_matrices = []
    transformed_points = []
    transformed_points_camera = []
    # 对每个样本进行处理
    for i in range(num_samples):
        # 1. 获取当前样本的数据
        A = np.array(enlarged_masks_center[i], dtype=np.float32)
        B = np.array(image_center[i], dtype=np.float32)
        K_i = K[i].cpu().numpy()  # 将当前样本的 K 转换为 NumPy 数组
        
        # 2. 转换为齐次坐标
        A_homogeneous = np.append(A, 1)
        B_homogeneous = np.append(B, 1)
        
        # 3. 计算 K 的逆矩阵
        K_inv = np.linalg.inv(K_i)
        
        # 4. 计算相机坐标系下的 A 和 B
        A_camera = np.dot(K_inv, A_homogeneous)
        B_camera = np.dot(K_inv, B_homogeneous)
        
        # 5. 计算旋转矩阵
        rotation_matrix = calculate_rotation_matrix(A_camera, B_camera)
        
        # 6. 将 A 点变换到相机图像中心
        A_transformed = transform_point(A_camera, rotation_matrix)
        #获取经过内参K变换后的A点
        A_transformed_camera = np.dot(K_i, A_transformed)
        A_transformed_camera=A_transformed_camera[:2]/A_transformed_camera[2]
        
        # 7. 将结果添加到列表中
        rotation_matrices.append(rotation_matrix)
        transformed_points.append(A_transformed)
        transformed_points_camera.append(A_transformed_camera)
    
    return rotation_matrices, transformed_points, transformed_points_camera


# 示例用法
if __name__ == "__main__":
    # 示例数据 (批量处理)
    batch_size = 2
    
    # 创建批量的 enlarged_masks_center 和 image_center
    enlarged_masks_center = np.random.randint(0, 300, size=(batch_size, 2))
    image_center = np.random.randint(0, 500, size=(batch_size, 2))
    
    # 创建批量的相机内参矩阵 K
    K = torch.zeros((batch_size, 3, 3), dtype=torch.float64)
    for i in range(batch_size):
        K[i] = torch.tensor([[1000, 0, 320],
                             [0, 1000, 240],
                             [0, 0, 1]], dtype=torch.float64)
    
    # 计算相机变换
    rotation_matrices, transformed_points, transformed_points_camera = calculate_camera_transformation(enlarged_masks_center, image_center, K)
    
    # 打印结果 (仅显示第一个样本的结果)
    print("第一个样本的旋转矩阵:\n", rotation_matrices[0])
    print("enlarged_masks_center:\n", enlarged_masks_center[0])
    print("image_center:\n", image_center[0])
    print("第一个样本变换后的 A 点 transformed_points:\n", transformed_points[0])
    print("第一个样本变换后2d的 A 点 transformed_points_camera:\n", transformed_points_camera[0])
    print(f"总共处理了 {len(rotation_matrices)} 个样本")