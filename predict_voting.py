import os
import argparse
from ultralytics import YOLO
from utils.voting_utils import ransac_vote, visualize_voting
import cv2
import numpy as np


def predict_with_voting(model_path, image_path, conf_threshold=0.25):
    """
    使用投票机制进行预测
    """
    # 加载训练好的模型
    model = YOLO(model_path)

    # 进行预测
    results = model(source=image_path, conf=conf_threshold)

    if len(results) == 0:
        print("未检测到目标")
        return None

    # 提取所有检测结果的关键点
    all_keypoints = []
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            kpts_data = result.keypoints.data.cpu().numpy()
            all_keypoints.append(kpts_data)

    if not all_keypoints:
        print("未检测到关键点")
        return None

    # 合并所有关键点
    combined_kpts = np.concatenate(all_keypoints, axis=0)

    # 使用RANSAC投票
    final_keypoint = ransac_vote(combined_kpts, conf_threshold=0.5)

    # 可视化结果
    visualize_voting(image_path, combined_kpts, final_keypoint)

    print(f"检测到 {len(combined_kpts)} 个关键点")
    print(f"最终投票结果: {final_keypoint}")

    return final_keypoint


def main():
    parser = argparse.ArgumentParser(description='使用投票机制的YOLO姿态估计')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='模型路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像或文件夹路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')

    args = parser.parse_args()

    # 检查输入路径是文件还是文件夹
    if os.path.isdir(args.image):
        # 如果是文件夹，遍历其中的图片文件
        for filename in os.listdir(args.image):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_file = os.path.join(args.image, filename)
                predict_with_voting(args.model, image_file, args.conf)
    elif os.path.isfile(args.image):
        # 如果是文件，直接处理
        predict_with_voting(args.model, args.image, args.conf)
    else:
        print(f"输入路径 {args.image} 无效，请检查路径是否正确。")


if __name__ == '__main__':
    main()