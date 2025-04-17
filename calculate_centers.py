import cv2
import numpy as np
import torch

def calculate_mask_center(mask):
    """
    计算二值图像 mask 的中心点.

    参数:
    mask (numpy.ndarray 或 torch.Tensor): 二值图像，像素值为 0 或 255.
                                         可以是单个掩码或批量掩码 [batch_size, channels, height, width].

    返回值:
    tuple 或 list of tuples: 中心点坐标 (x, y)，如果 mask 为空则返回 None.
                            如果输入是批量掩码，则返回每个掩码的中心点列表.
    """
    if mask is None or (isinstance(mask, np.ndarray) and mask.size == 0):
        return None

    # 如果输入是 PyTorch 张量，转换为 numpy 数组
    if isinstance(mask, torch.Tensor):
        # 检查是否是批量掩码 [batch_size, channels, height, width]
        if len(mask.shape) == 4:
            batch_size = mask.shape[0]
            centers = []
            for i in range(batch_size):
                # 提取单个掩码并转换为 numpy
                single_mask = mask[i, 0].cpu().numpy().astype(np.uint8)
                center = calculate_mask_center(single_mask)
                centers.append(center)
            return centers
        else:
            # 单个掩码，转换为 numpy
            mask = mask.cpu().numpy().astype(np.uint8)

    # 计算图像矩
    moments = cv2.moments(mask)

    # 检查 m00 是否为零，避免除以零的错误
    if moments["m00"] == 0:
        return None

    # 计算中心点坐标
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    return (center_x, center_y)


def calculate_image_center(image):
    """
    计算图像的中心点.

    参数:
    image (numpy.ndarray 或 torch.Tensor): 输入图像，可以是单个图像或批量图像.

    返回值:
    tuple 或 list of tuples: 中心点坐标 (x, y).
                            如果输入是批量图像，则返回每个图像的中心点列表.
    """
    # 如果输入是 PyTorch 张量
    if isinstance(image, torch.Tensor):
        # 检查是否是批量图像 [batch_size, channels, height, width]
        if len(image.shape) == 4:
            batch_size = image.shape[0]
            centers = []
            for i in range(batch_size):
                # 获取单个图像的高度和宽度
                height, width = image.shape[2], image.shape[3]
                center_x = width // 2
                center_y = height // 2
                centers.append((center_x, center_y))
            return centers
        else:
            # 单个图像
            height, width = image.shape[1], image.shape[2]
    else:
        # numpy 数组
        height, width = image.shape[:2]  # 获取图像的高度和宽度
    
    center_x = width // 2  # 使用整数除法
    center_y = height // 2 # 使用整数除法
    return (center_x, center_y)


# 示例用法
if __name__ == "__main__":
    # 创建一个示例 mask (你需要替换成你自己的 mask)
    # 假设 mask 是一个 numpy 数组，像素值为 0 或 255
    enlarged_masks = np.zeros((200, 200), dtype=np.uint8)
    enlarged_masks[50:150, 50:150] = 255  # 创建一个简单的矩形 mask

    # 创建一个示例图像 (你需要替换成你自己的图像)
    image = np.zeros((300, 400, 3), dtype=np.uint8)  # 创建一个 300x400 的黑色图像

    # 计算 mask 的中心点
    enlarged_masks_center = calculate_mask_center(enlarged_masks)

    # 计算图像的中心点
    image_center = calculate_image_center(image)

    # 打印结果
    print("enlarged_masks 中心点:", enlarged_masks_center)
    print("图像中心点:", image_center)

    # 可视化 mask 和中心点 (可选)
    if enlarged_masks_center is not None:
        cv2.circle(enlarged_masks, enlarged_masks_center, 5, (128), -1)  # 在 mask 上标记中心点
        # cv2.imshow("Enlarged Masks with Center", enlarged_masks)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # PyTorch 张量示例
    if torch.cuda.is_available():
        print("\nPyTorch 张量示例:")
        # 创建一个批量掩码 [batch_size, channels, height, width]
        batch_masks = torch.zeros((35, 1, 288, 512), dtype=torch.uint8)
        # 在每个掩码中央创建一个矩形
        batch_masks[:, :, 100:200, 200:300] = 255
        
        # 计算批量掩码的中心点
        batch_centers = calculate_mask_center(batch_masks)
        print(f"批量掩码中心点 (前5个): {batch_centers[:5]}")
        
        # 计算批量图像的中心点
        batch_image_centers = calculate_image_center(batch_masks)
        print(f"批量图像中心点 (前5个): {batch_image_centers[:5]}")