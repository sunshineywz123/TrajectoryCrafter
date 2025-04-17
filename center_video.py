import cv2
import numpy as np

def expand_video_with_mask(input_video_path, input_mask_path, output_video_path, output_mask_path):
    """
    根据 2D Mask 的中心，将视频画面进行平移和扩图，并将扩图区域置零。

    Args:
        input_video_path (str): 输入视频的路径。
        input_mask_path (str): 输入 Mask 视频的路径。
        output_video_path (str): 输出视频的路径。
        output_mask_path (str): 输出 Mask 视频的路径。
    """

    # 读取视频
    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        print("无法打开输入视频")
        return

    # 读取 Mask 视频
    mask_capture = cv2.VideoCapture(input_mask_path)
    if not mask_capture.isOpened():
        print("无法打开输入 Mask 视频")
        video_capture.release()
        return

    # 获取视频的宽度、高度和帧率
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # 定义 VideoWriter 用于保存输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    mask_writer = cv2.VideoWriter(output_mask_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    while True:
        # 读取视频帧
        ret, frame = video_capture.read()
        if not ret:
            break

        # 读取 Mask 帧
        ret_mask, mask_frame = mask_capture.read()
        if not ret_mask:
            print("Mask 视频提前结束")
            break

        # 将 Mask 转换为灰度图
        mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

        # 计算 Mask 的中心
        moments = cv2.moments(mask_gray)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:
            center_x = frame_width // 2
            center_y = frame_height // 2

        # 计算平移量
        shift_x = frame_width // 2 - center_x
        shift_y = frame_height // 2 - center_y

        # 创建平移矩阵
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # 平移图像和 Mask
        translated_frame = cv2.warpAffine(frame, translation_matrix, (frame_width, frame_height))
        translated_mask = cv2.warpAffine(mask_gray, translation_matrix, (frame_width, frame_height))

        # 创建一个与图像大小相同的黑色图像，用于填充扩图区域
        black_background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        black_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # 将平移后的图像放置在黑色背景上，实现扩图区域置零
        x_offset = max(0, -shift_x)
        y_offset = max(0, -shift_y)
        translated_frame_x_start = max(0, shift_x)
        translated_frame_y_start = max(0, shift_y)

        # 计算有效区域的宽度和高度
        valid_width = frame_width - abs(shift_x)
        valid_height = frame_height - abs(shift_y)

        # 确保切片区域大小一致
        black_background[y_offset:y_offset + valid_height,
                         x_offset:x_offset + valid_width] = \
            translated_frame[translated_frame_y_start:translated_frame_y_start + valid_height,
                             translated_frame_x_start:translated_frame_x_start + valid_width]

        black_mask[y_offset:y_offset + valid_height,
                   x_offset:x_offset + valid_width] = \
            translated_mask[translated_frame_y_start:translated_frame_y_start + valid_height,
                            translated_frame_x_start:translated_frame_x_start + valid_width]

        # 写入输出视频
        video_writer.write(black_background)
        mask_writer.write(black_mask)

    # 释放资源
    video_capture.release()
    mask_capture.release()
    video_writer.release()
    mask_writer.release()
    cv2.destroyAllWindows()

    print("视频处理完成")
input_path = '/nas/datasets/bullet_demo/rollerblade'
# 示例用法
input_video_path = input_path + "/rgb.mp4"  # 替换为你的输入视频路径
input_mask_path = input_path + "/mask.mp4"  # 替换为你的输入 Mask 视频路径
output_video_path = input_path + "/output.mp4"  # 替换为你的输出视频路径
output_mask_path = input_path + "/output_mask.mp4"  # 替换为你的输出 Mask 视频路径

expand_video_with_mask(input_video_path, input_mask_path, output_video_path, output_mask_path)