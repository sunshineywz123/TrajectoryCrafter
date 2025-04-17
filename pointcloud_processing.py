import numpy as np
import open3d as o3d

def depth_map_to_point_cloud(depth_image, intrinsic_matrix):
    """
    将深度图转换为点云。

    参数:
    depth_image (numpy.ndarray): 深度图，numpy数组格式。
    intrinsic_matrix (numpy.ndarray): 相机内参矩阵，3x3的numpy数组。

    返回值:
    numpy.ndarray: 点云，Nx3的numpy数组。
    """
    height, width = depth_image.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / fx
    y = (y - cy) / fy

    z = depth_image
    x = x * z
    y = y * z

    points = np.stack([x, y, z], axis=-1)
    return points.reshape(-1, 3)


def mask_point_cloud(point_cloud, mask):
    """
    根据mask过滤点云。

    参数:
    point_cloud (numpy.ndarray): 点云，Nx3的numpy数组。
    mask (numpy.ndarray): mask，布尔类型的numpy数组，与深度图尺寸相同。

    返回值:
    numpy.ndarray: 过滤后的点云。
    """
    masked_point_cloud = point_cloud[mask.flatten()]
    return masked_point_cloud


def calculate_bounding_box_center(point_cloud):
    """
    计算点云的3D边界框中心。

    参数:
    point_cloud (numpy.ndarray): 点云，Nx3的numpy数组。

    返回值:
    numpy.ndarray: 边界框的中心点坐标，1x3的numpy数组。
    """
    if len(point_cloud) == 0:
        return None  # 处理点云为空的情况

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    bounding_box = pcd.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()
    return np.array(center)
def process_point_cloud(depth_image, intrinsic_matrix, mask):
    """
    从深度图和内参矩阵处理点云数据，并返回3D Bounding Box的中心点。

    Args:
        depth_image (numpy.ndarray): 深度图像数据，二维数组。
        intrinsic_matrix (numpy.ndarray): 内参矩阵，三维数组。
        mask (numpy.ndarray): 掩码图像数据，二维数组。

    Returns:
        numpy.ndarray: 3D Bounding Box的中心点坐标，一维数组。

    """
    # 步骤1: 从深度图反投影得到点云
    point_cloud = depth_map_to_point_cloud(depth_image, intrinsic_matrix)

    # 步骤2: 应用Mask
    masked_point_cloud = mask_point_cloud(point_cloud, mask)

    # 步骤3: 计算3D Bounding Box并获取中心点
    center = calculate_bounding_box_center(masked_point_cloud)

    return center
# 示例
if __name__ == '__main__':
    # 示例数据
    depth_image = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]], dtype=np.float32)  # 示例深度图
    intrinsic_matrix = np.array([[500, 0, 160],
                                 [0, 500, 120],
                                 [0, 0, 1]])  # 示例相机内参
    mask = np.array([[True, False, True],
                     [False, True, False],
                     [True, False, True]], dtype=bool)  # 示例mask

    center = process_point_cloud(depth_image, intrinsic_matrix, mask)


    print("\n3D Bounding Box 中心:\n", center)