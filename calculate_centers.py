import cv2
import numpy as np

def calculate_mask_center(mask):
    """
    计算二值图像 mask 的中心点.

    参数:
    mask (numpy.ndarray): 二值图像，像素值为 0 或 255.

    返回值:
    tuple: 中心点坐标 (x, y)，如果 mask 为空则返回 None.
    """
    if mask is None or mask.size == 0:
        return None

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
    image (numpy.ndarray): 输入图像.

    返回值:
    tuple: 中心点坐标 (x, y).
    """
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
        cv2.imshow("Enlarged Masks with Center", enlarged_masks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()