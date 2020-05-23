# YOLOv4: Optimal Speed and Accuracy of Object Detection

#### 该仓库收录于[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)

# 说明
- 非常感谢[Tianxiaomo](https://github.com/Tianxiaomo)等大佬的工作
- [原仓库](https://github.com/Tianxiaomo/pytorch-YOLOv4)的训练模块未更新完毕，正在跟进
- 源码基于[yolov3](https://github.com/bobo0810/PytorchNetHub/tree/master/Yolov3_pytorch)源码修改，有基础更易理解

# 接下来工作
 
- [x] 推理部分注释
- [x] 原生NMS精简为Pytorch官方接口（仅本仓库）

# 环境

| python | pytorch | system  |
|------------|-------------|--------|
| 3.6        | 1.5       | Ubuntu |


# 代码结构
```
YOLOv4_Pytorch
│
└───data
│   │   coco.names
│   
└───img
│   │   ...
│   
└───tool
│   │   ...
│   
└───utils
│   │   ...
│   
└───demo.py
└───models.py 
└───yolov4.weights

```

# 推理

- 1、下载[原仓库](https://github.com/Tianxiaomo/pytorch-YOLOv4)预训练权重yolov4.weights,放到仓库根目录

  baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)

  google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

- 2、运行demo.py即可，结果为img/predictions.jpg

  参数修改：demo.py的main方法

![image](https://github.com/bobo0810/YOLOv4_Pytorch/blob/master/img/predictions.jpg)




 # 参考

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```

