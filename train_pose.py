# train_pose.py
from ultralytics import YOLO


def train_model():
    # 加载预训练的姿态估计模型
    model = YOLO('weights/yolo11n-pose.pt')

    # 训练模型
    results = model.train(
        data='data/weed.yaml',
        epochs=500,
        imgsz=640,
        batch=32,
        device='cpu',
        project='runs/train',
        name='weed_pose'
    )

    print('关键点检测模型训练完毕')
    return results


if __name__ == '__main__':
    train_model()