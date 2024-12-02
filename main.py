import os
from ultralytics import YOLO

import preprocess

# 运行重复加载OpenMP，tensorflow和pytorch的可能会冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 数据集路径
dataset_yaml = "D:/Dataset/detection/apples.yaml"
train_dir = "D:/Dataset/detection/train"
val_dir = "D:/Dataset/detection/val"
test_dir = "D:/Dataset/detection/test"
results_dir = "D:/Dataset/detection/results"


def train_model():
    # 加载预训练的YOLO模型
    model = YOLO("yolo11n.pt")

    # 指定训练集并训练，批次为20，结果放到runs/detect里面
    model.train(data=dataset_yaml, epochs=20, project="runs/detect", name="apples_train")

    return model


def val_model(model):
    # 在验证集上验证
    results = model.val()
    print("验证完成")


def save_model(model):
    # 保存模型，onnx是用ProtoBuf的序列化方式存储的
    success = model.export(format="onnx")


def predict_model(model, image_path):
    # 试试跑一个图片看看效果
    results = model(image_path)

    # 获取检测框数量（即苹果数量）
    apple_count = len(results[0].boxes)  # YOLOv8的检测结果保存在results[0].boxes

    # 输出结果
    print(f"苹果有{apple_count}个")


def load_model(model_path):
    model = YOLO(model_path)
    return model


def transfer_yolo():
    train_dir = "D:/Dataset/detection/train/images"
    mask_dir = "D:/Dataset/detection/train/masks"
    output_dir = "D:/Dataset/detection/train/labels"
    preprocess.batch_transfer(train_dir, mask_dir, output_dir)


def generate_val():
    preprocess.generate_val(train_dir, val_dir, 0.2)


def my_model():
    model = train_model()
    val_model(model)
    predict_model(model, "images/20150919_174151_image1.png")
    save_model(model)


if __name__ == '__main__':
    print("训练开始!")
    my_model()
