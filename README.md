# RetinaFace in PyTorch

This is a personal extra code from [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) which is [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). 
I tried to replace mobilenet0.25 backbone with mobilenetv3_small and mobilenetv3_large in [torchvision](https://pytorch.org/vision/0.8/models.html). Each size of backbone model mobilenet0.25, mobilenetv3_small and mobilenetv3_large is 1.7M, 1.4M and 11.7M.

## WiderFace Val Performance in single scale When using Mobilenet0.25, MobilenetV3_small, MobilenetV3_large as backbone net.
| Style | easy | medium | hard |
|:-|:-:|:-:|:-:|
| Mobilenet0.25 (original image scale) | 89.63% | 87.31% | 72.80% |
| MobilenetV3_small (original image scale) | 88.86% | 85.56% | 69.90% |
| MobilenetV3_large (original image scale) | 93.21% | 91.63% | 80.43% |

## Inference time for each resolutions
## Intel i5-8500 CPU, NVIDIA 1060GTX 6GB GPU
| Backbone | VGA(640*480) | HD(1920*1080) | 4K(4096*2160)
|:-|:-:|:-:|:-:|
| Mobilenet0.25(GPU) | 7.45ms | 18.42ms | 73.46ms |
| MobilenetV3_small(GPU) | 8.66ms | 18.29ms | 72.96ms |
| MobilenetV3_large(GPU) | 12.17ms | 55.59ms | 231.84ms |
| Mobilenet0.25(CPU) | 48.05ms | 347.51ms | 1471.12ms |
| MobilenetV3_small(CPU) | 74.47ms | 449.50ms | 2407.47ms |

### Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [TensorRT](#tensorrt)
- [References](#references)

## Installation
##### Clone and install
1. git clone https://github.com/sglee487/Pytorch_Retinaface.git

2. Pytorch version 1.1.0+ and torchvision 0.9.1+ are needed.

3. Codes are based on Python 3

##### Data
1. Download the [WIDERFACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell
  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
```
ps: wider_val.txt only include val file names but not label information.

##### Data1
We also provide the organized dataset we used as in the above directory structure.

Link: from [google cloud](https://drive.google.com/open?id=11UGV3nbVv1x9IC--_tK3Uxf7hA6rlbsS) or [baidu cloud](https://pan.baidu.com/s/1jIp9t30oYivrAvrgUgIoLQ) Password: ruck

## Training
We provide mobilenet0.25, mobilenetv3_small, mobilenetv3_large as backbone network to train model.
```Shell
  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      mobilenetv3sm_Final.pth
      mobilenetv3lg_Final.pth
```
1. Before training, you can check network configuration (e.g. batch_size, min_sizes and steps etc..) in ``data/config.py and train.py``.

2. Train the model using WIDER FACE:
  ```Shell
  CUDA_VISIBLE_DEVICES=0 python train.py --network mobile0.25 or mobilenetv3sm or mobilenetv3lg
  ```


## Evaluation
### Evaluation widerface val
1. Generate txt file
```Shell
python test_widerface.py --trained_model weight_file --network mobile0.25 or mobilenetv3sm or mobilenetv3lg
```
2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
```Shell
cd ./widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
3. You can also use widerface official Matlab evaluate demo in [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)
### Evaluation FDDB

1. Download the images [FDDB](https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH) to:
```Shell
./data/FDDB/images/
```

2. Evaluate the trained model using:
```Shell
python test_fddb.py --trained_model weight_file --network mobile0.25 or mobilenetv3sm or mobilenetv3lg
```

3. Download [eval_tool](https://bitbucket.org/marcopede/face-eval) to evaluate the performance.

<p align="center"><img src="curve/1.jpg" width="640"\></p>

### Evaluation inference time from random image
```shell
python measure_inferenceTime.py --trained_model weight_file --network mobile0.25 or mobilenetv3sm or mobilenetv3lg
```

## TensorRT
-[TensorRT](https://github.com/wang-xinyu/tensorrtx/tree/master/retinaface)

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
