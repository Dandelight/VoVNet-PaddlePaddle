# VoVNet

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
   * [3.3 准备模型](#33-准备模型)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
* [5. 模型推理部署](#5-模型推理部署)
* [6. 自动化测试脚本](#6-自动化测试脚本)
* [7. LICENSE](#7-license)
* [8. 参考链接与文献](#8-参考链接与文献)


## 1. 简介

这是基于`PaddlePaddle`和`PaddleClas`的`VoVNet`实现。

**论文:** [An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection](https://arxiv.org/abs/1904.09730)

**参考repo:** [VoVNet.pytorch](https://github.com/stigma0617/VoVNet.pytorch)

## 2. 数据集和复现精度

数据集为`ImageNet-1K`，训练集包含1281167张图像，验证集包含50000张图像。您可以从[ImageNet 官网](https://image-net.org/)申请下载数据。

| 模型      | top1 acc (参考精度) | top1 acc (模型转换精度) | top1 acc (实际训练) | 权重 \| 训练日志 |
|:---------:|:------:|:----------:|:----------:|:----------:|
| VoVNet39 | 0.7677   |  0.76122  | 0.75904 | best_model.pdparams \| train.log |

权重及训练日志下载地址：[百度网盘](链接：https://pan.baidu.com/s/1W7H_oBMqlxE_VjtCtYdxTA?pwd=xrub)

## 3. 准备数据与环境

### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * RTX3090
- 框架：
  - PaddlePaddle == 2.2.2

* 下载代码

```bash
git clone https://github.com/Dandelight/VoVNet-PaddlePaddle
cd VoVNet-PaddlePaddle
git checkout -b vovnet
```

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

如果您已经拥有`ImageNet-1K`数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。

下载后请将数据集挂载到`dataset/ILSVRC2012`文件夹下。

```shell
# 在本项目根目录下执行
ln –s ${ImageNet根目录} ./dataset/ILSVRC2012
```

挂载后数据的组织形式如下。

```
PaddleClas/dataset/ILSVRC2012/
|_ train/
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val/
|  |_ ILSVRC2012_val_00000001.JPEG
|  |_ ...
|  |_ ILSVRC2012_val_00050000.JPEG
|_ train_list.txt
|_ val_list.txt
```

如果只是希望快速体验模型训练功能，可以参考：[飞桨训推一体认证（TIPC）开发文档](https://github.com/PaddlePaddle/models/blob/tipc/docs/tipc_test/README.md)


### 3.3 准备模型


如果您希望直接体验评估或者预测推理过程，可以直接根据第2章的内容下载提供的预训练模型，直接体验模型评估、预测、推理部署等内容。


## 4. 开始使用


### 4.1 模型训练

* 单机多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/train.py \
    -c ./ppcls/configs/ImageNet/VoVNet/VoVNet39.yaml
```

部分训练日志如下所示。

```
[2022/04/19 15:14:20] root INFO: [Train][Epoch 70/90][Iter: 750/1252]lr: 0.00100, top1: 0.78320, top5: 0.92243, CELoss: 1.84433, loss: 1.84433, batch_cost: 0.78969s, reader_cost: 0.02191, ips: 324.17870 images/sec, eta: 5:36:10
[2022/04/19 15:14:28] root INFO: [Train][Epoch 70/90][Iter: 760/1252]lr: 0.00100, top1: 0.78327, top5: 0.92241, CELoss: 1.84431, loss: 1.84431, batch_cost: 0.78985s, reader_cost: 0.02191, ips: 324.11334 images/sec, eta: 5:36:06
[2022/04/19 15:14:36] root INFO: [Train][Epoch 70/90][Iter: 770/1252]lr: 0.00100, top1: 0.78330, top5: 0.92244, CELoss: 1.84447, loss: 1.84447, batch_cost: 0.79003s, reader_cost: 0.02188, ips: 324.03792 images/sec, eta: 5:36:03
```

### 4.2 模型评估

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus="0,1,2,3" \
    tools/eval.py \
    -c ./ppcls/configs/ImageNet/VoVNet/VoVNet39.yaml \
    -o Global.pretrained_model=$TRAINED_MODEL
```

### 4.3 模型预测

```shell
python tools/infer.py \
    -c ./ppcls/configs/ImageNet/VoVNet/VoVNet39.yaml \
    -o Infer.infer_imgs=./deploy/images/ILSVRC2012_val_00020010.jpeg \
    -o Global.pretrained_model=$TRAINED_MODEL
```
<div align="center">
    <img src="./deploy/images/ILSVRC2012_val_00020010.jpeg" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 246, 210, 209, 171], 'scores': [0.71461, 0.00499, 0.00289, 0.00234, 0.00232], 'file_name': './deploy/images/ILSVRC2012_val_00020010.jpeg', 'label_names': ['Weimaraner', 'Great Dane', 'German short-haired pointer', 'Chesapeake Bay retriever', 'Italian greyhound']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.71461`。

## 5. 模型推理部署

### 5.1 基于Inference的推理

可以参考[模型导出](./docs/zh_CN/inference_deployment/export_model.md)，

将该模型转为 inference 模型只需运行如下命令：

```shell
python tools/export_model.py \
    -c ./ppcls/configs/ImageNet/VoVNet/VoVNet39.yaml \
    -o Global.save_inference_dir=./deploy/models/class_VAN_tiny_ImageNet_infer \
    -o Global.pretrained_model=$TRAINED_MODEL
```

### 5.2 基于Serving的服务化部署

Serving部署教程可参考：[链接](./deploy/paddleserving/readme.md)。


## 6. 自动化测试脚本

**详细日志在test_tipc/output**

TIPC: [test_tipc/README.md](./test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考<https://github.com/LDOUBLEV/AutoLog>。

```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```

进行TIPC：进行中

* 更多详细内容，请参考：[TIPC测试文档](./test_tipc/README.md)。

## 7. LICENSE

本项目在[Apache 2.0 license](./LICENSE)许可证下发布。

## 8. 参考链接与文献

1. An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection: https://arxiv.org/abs/1904.09730
2. stigma0617/VoVNet.pytorch: https://github.com/stigma0617/VoVNet.pytorch
3. PaddlePaddle/PaddleClas: https://github.com/PaddlePaddle/PaddleClas
4. flytocc/PaddleClas_VAN-Classification: https://github.com/flytocc/PaddleClas_VAN-Classification
