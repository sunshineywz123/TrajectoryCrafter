import torch

def orient_camera_to_centers(centers, pose_s_in):
    """
    调整相机位姿，使每一帧相机都指向对应的3D中心点。

    参数:
    centers (torch.Tensor): 3D中心点坐标，shape (35, 3)。
    pose_s_in (torch.Tensor): 原始相机位姿，shape (35, 4, 4)。

    返回值:
    torch.Tensor: 调整后的相机位姿，shape (35, 4, 4)。
    """
    num_frames = centers.shape[0]
    adjusted_poses = []

    for i in range(num_frames):
        center = centers[i]
        pose = pose_s_in[i]

        # 1. 计算相机位置
        camera_position = pose[:3, 3]

        # 2. 计算方向向量
        direction_vector = center - camera_position
        direction_vector = direction_vector / torch.norm(direction_vector)  # 归一化

        # 3. 计算旋转矩阵
        # 定义相机默认朝向 (通常是Z轴负方向)
        default_forward = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

        # 计算旋转轴 (使用叉积)
        rotation_axis = torch.cross(default_forward, direction_vector)
        rotation_axis = rotation_axis / torch.norm(rotation_axis)  # 归一化

        # 计算旋转角度 (使用点积)
        rotation_angle = torch.acos(torch.dot(default_forward, direction_vector))

        # 使用Rodrigues公式计算旋转矩阵
        k = rotation_axis
        theta = rotation_angle
        K = torch.tensor([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]], dtype=torch.float32)
        rotation_matrix = (
            torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
        )

        print(f'rotation_matrix: {rotation_matrix}')
        # 4. 应用变换
        # 创建新的位姿矩阵
        new_pose = pose.clone()
        new_pose[:3, :3] = rotation_matrix @ pose[:3, :3]  # 应用旋转

        adjusted_poses.append(new_pose)

    return torch.stack(adjusted_poses)

# 示例
if __name__ == '__main__':
    # 示例数据
    num_frames = 35
    centers = torch.randn(num_frames, 3)  # 随机生成3D中心点坐标
    pose_s_in = torch.randn(num_frames, 4, 4)  # 随机生成原始相机位姿
    # 确保位姿矩阵的右下角元素为1
    pose_s_in[:, 3, 3] = 1.0

    # 调整相机位姿
    adjusted_poses = orient_camera_to_centers(centers, pose_s_in)

    print("原始相机位姿:\n", pose_s_in)
    print("\n调整后的相机位姿:\n", adjusted_poses)