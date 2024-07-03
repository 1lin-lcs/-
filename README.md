# 说明

该项目分为三个文件夹

- `code`：包含模型代码、训练代码和测试代码

- `result`：包含模型训练中每一轮的结果

- `model`：阿里云盘分享连接，包含ResNetRGA50、ResNetAtt50和ResNet50训练到的最高的正确率的`pth`文件

## 文件说明

`RGA.py`和`Attention.py`分别为RGA模块和注意力机制

`ResNetRGA.py`和`ResNetAtt.py`分别使用RGA模块和注意力机制的`ResNet50`

`ResNet.py`是未修改的`ResNe50`

`Mean_std_batch.py`分批计算均值和标准差

`LoadModelParam.py`用来加载模型文件

`train.py`和`test.py`分别是训练和测试代码

## 使用

训练只要使用`python ./train.py`即可，默认使用`ResNetRGA`，在`train.py`中定位到`test=Resnet50_RGA(10)`这行可以修改使用的模型

`Mean_std_batch.py`会对变量`TRAIN_DATASET_PATH`下的所有`png`图片进行计算

# 依赖

该项目总共的依赖

- `torch`

- `os`

- `opencv-python`

- `glob`

- `tqdm`

- `thop`

- `csv`

- `pillow`

- `tensorboardX`

- `torchvision`

## model文件夹说明

Q:为什么不直接上传模型？

A:因为模型文件大小超过了git上传限制的100MB，而且阿里云盘不给分享
