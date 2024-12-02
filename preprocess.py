import cv2
import os
import shutil
import random


def mask_to_yolo(image_path, mask_path, output_dir, class_id=0):
    # 读取原始图像和对应的 mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 获取图像尺寸
    img_height, img_width = image.shape[:2]

    # 二值化 mask
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"There are {len(contours)} apples")

    yolo_annotations = []

    for contour in contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 计算归一化后的中心坐标和宽高
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        # 保存为 YOLO 格式
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # 保存为文本文件
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    with open(output_file, "w") as f:
        f.write("\n".join(yolo_annotations))

    print(f"Annotations saved to {output_file}")


def batch_transfer(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有图片和 mask
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file)

            if os.path.exists(mask_path):
                mask_to_yolo(image_path, mask_path, output_dir)


# 训练集目录，生成的验证集目录，分割度（比如0.2代表82分）
def generate_val(train_dir, val_dir, split_ratio):
    # 定义目录
    images_dir = os.path.join(train_dir, "images")
    labels_dir = os.path.join(train_dir, "labels")
    val_images_dir = os.path.join(val_dir, "images")
    val_labels_dir = os.path.join(val_dir, "labels")

    # 创建验证集目录
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    # 随机选择验证集
    val_size = int(len(image_files) * split_ratio)
    val_files = random.sample(image_files, val_size)

    for val_file in val_files:
        # 图片和对应标签文件的路径
        image_path = os.path.join(images_dir, val_file)
        label_path = os.path.join(labels_dir, os.path.splitext(val_file)[0] + ".txt")

        # 验证集目标路径
        val_image_path = os.path.join(val_images_dir, val_file)
        val_label_path = os.path.join(val_labels_dir, os.path.splitext(val_file)[0] + ".txt")

        # 移动图片和标签到验证集
        shutil.move(image_path, val_image_path)
        if os.path.exists(label_path):
            shutil.move(label_path, val_label_path)

    print(f"验证集创建完成，共 {val_size} 个样本。")