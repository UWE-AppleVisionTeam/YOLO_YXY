# YOLO_YXY
YOLO模型实现目标检测

1. 数据集用明尼苏达的苹果数据集(url: https://conservancy.umn.edu/items/e1bb4015-e92a-4295-822c-d21d277ecfbd)
2. 该项目只使用了detection这部分的数据
3. YOLO的目标检测需要数据集提供的Label（标签）必须是方形盒子，明尼苏达苹果数据集里则是已经数出来的苹果二值化后的灰度图（mask文件夹下），所以使用前需要注意对数据做处理
4. detection文件夹里面没有val（验证集），因此代码里也有函数方法去分割出来验证集的
5. 最终结果直接对results计数他的boxes的数量就可以得出苹果数量

函数方法使用顺序（推荐）：
transfer_yolo()
generate_val()
my_model()
