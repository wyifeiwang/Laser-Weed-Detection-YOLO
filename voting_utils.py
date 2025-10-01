# utils/voting_utils.py
import numpy as np
from sklearn.linear_model import RANSACRegressor


def ransac_vote(keypoints, conf_threshold=0.5):
    """
    使用RANSAC对多个检测结果的关键点进行投票
    keypoints: shape [N, 1, 3] 其中N是检测到的目标数，最后一个维度是[x, y, confidence]
    """
    if len(keypoints) == 0:
        return None

    # 过滤低置信度的关键点
    valid_kpts = keypoints[keypoints[:, 0, 2] > conf_threshold]
    if len(valid_kpts) < 2:
        # 如果点数太少，直接返回平均值
        return np.mean(valid_kpts[:, 0, :2], axis=0) if len(valid_kpts) > 0 else None

    # 提取坐标
    points = valid_kpts[:, 0, :2]

    # 使用RANSAC找到一致性最好的点
    try:
        model = RANSACRegressor(random_state=42, min_samples=2)
        X = np.arange(len(points)).reshape(-1, 1)
        model.fit(X, points)

        # 获取inlier点（一致性最好的点）
        inlier_mask = model.inlier_mask_
        if np.any(inlier_mask):
            inlier_points = points[inlier_mask]
            # 取inlier点的中值作为最终预测
            final_kpt = np.median(inlier_points, axis=0)
            return final_kpt
        else:
            return np.median(points, axis=0)
    except:
        # 如果RANSAC失败，退回平均值
        return np.mean(points, axis=0)


def visualize_voting(image_path, keypoints, final_kpt):
    """可视化投票结果"""
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return

    # 绘制所有检测到的关键点
    for kpt in keypoints:
        x, y, conf = kpt[0]
        color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)  # 绿色：高置信度，红色：低置信度
        cv2.circle(img, (int(x), int(y)), 5, color, -1)

    # 绘制最终投票结果
    if final_kpt is not None:
        cv2.circle(img, (int(final_kpt[0]), int(final_kpt[1])), 10, (255, 0, 0), -1)  # 蓝色：最终结果

    cv2.imwrite('voting_result.jpg', img)
    return img