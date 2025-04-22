import numpy as np
from scipy.spatial.transform import Rotation as R

def slerp_rotation_matrix(R1, R2, num_frames):
    """
    在两个旋转矩阵之间进行球面线性插值。

    参数:
    R1 (numpy.ndarray): 起始旋转矩阵。
    R2 (numpy.ndarray): 目标旋转矩阵。
    num_frames (int): 插值的帧数。

    返回值:
    list: 包含插值旋转矩阵的列表。
    """
    # 将旋转矩阵转换为四元数
    q1 = R.from_matrix(R1).as_quat()
    q2 = R.from_matrix(R2).as_quat()

    # 初始化插值旋转矩阵列表
    interpolated_rotations = []

    for i in range(num_frames + 1):
        # 计算插值参数 t
        t = i / num_frames

        # 球面线性插值（SLERP）
        q_interp = slerp(q1, q2, t)

        # 将插值四元数转换为旋转矩阵
        R_interp = R.from_quat(q_interp).as_matrix()
        interpolated_rotations.append(R_interp)

    return interpolated_rotations

def slerp(q1, q2, t):
    """
    在两个四元数之间进行球面线性插值。

    参数:
    q1 (numpy.ndarray): 起始四元数。
    q2 (numpy.ndarray): 目标四元数。
    t (float): 插值参数，范围 [0, 1]。

    返回值:
    numpy.ndarray: 插值后的四元数。
    """
    # 计算四元数之间的点积
    dot = np.dot(q1, q2)

    # 确保点积在有效范围内
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # 如果四元数非常接近，则进行线性插值
    if dot > 0.9995:
        q_interp = q1 + t * (q2 - q1)
        return q_interp / np.linalg.norm(q_interp)

    # 计算插值角度
    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    # 计算插值四元数的系数
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    # 计算插值四元数
    q_interp = (s0 * q1) + (s1 * q2)
    return q_interp

if __name__ == "__main__":
    # 定义两个旋转矩阵
    R1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 初始旋转矩阵（单位矩阵）
    R2 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # 目标旋转矩阵

    # 定义插值帧数
    num_frames = 50

    # 进行球面线性插值
    interpolated_rotations = slerp_rotation_matrix(R1, R2, num_frames)

    # 打印结果
    for i, R_interp in enumerate(interpolated_rotations):
        print(f"Frame {i}:")
        print(R_interp)
        print("-" * 20)