# 创建 merge_annotations.py
import os
import glob
import numpy as np


def merge_bbox_keypoints(bbox_dir, kpt_dir, output_dir):
    """
    合并边界框和关键点标注
    bbox_dir: 边界框标签目录
    kpt_dir: 关键点坐标目录
    output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有边界框文件
    bbox_files = glob.glob(os.path.join(bbox_dir, "*.txt"))

    for bbox_file in bbox_files:
        filename = os.path.basename(bbox_file)
        kpt_file = os.path.join(kpt_dir, filename)

        print(f"处理: {filename}")

        # 读取边界框数据
        with open(bbox_file, 'r') as f:
            bbox_lines = f.readlines()

        # 读取关键点数据
        if os.path.exists(kpt_file):
            with open(kpt_file, 'r') as f:
                kpt_lines = f.readlines()
        else:
            print(f"警告: 找不到关键点文件 {kpt_file}")
            continue

        # 合并数据
        output_lines = []
        for i, (bbox_line, kpt_line) in enumerate(zip(bbox_lines, kpt_lines)):
            # 解析边界框
            bbox_data = bbox_line.strip().split()
            if len(bbox_data) < 5:
                continue

            # 解析关键点
            kpt_data = kpt_line.strip().split()
            if len(kpt_data) < 2:
                continue

            # 创建YOLO姿态估计格式
            # 格式: class_id center_x center_y width height kpt1_x kpt1_y kpt1_conf
            new_line = f"{' '.join(bbox_data)} {kpt_data[0]} {kpt_data[1]} 2\n"
            output_lines.append(new_line)

        # 写入合并后的文件
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w') as f:
            f.writelines(output_lines)

        print(f"已创建: {output_file}")


# 使用示例
if __name__ == "__main__":
    # 假设你的数据这样组织：
    bbox_dir = r"C:\Users\ww\PyCharmMiscProject\.venv\lu\labels" # 边界框标签目录
    kpt_dir = r"C:\Users\ww\PyCharmMiscProject\.venv\lu\labeis-n"  # 关键点坐标目录
    output_dir = r"C:\Users\ww\PyCharmMiscProject\.venv\lu\labeis-e"  # 输出目录

    merge_bbox_keypoints(bbox_dir, kpt_dir, output_dir)