# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
from utils.utils import *
from tool.darknet2pytorch import Darknet
def detect(cfgfile, weightfile, imgfile):
    # 根据 配置文件 初始化网络
    m = Darknet(cfgfile)
    # 打印网络
    m.print_network()
    # 加载 模型权重
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    # 默认使用 coco类别
    namesfile = 'data/coco.names'

    # 默认CPU
    use_cuda = 0
    if use_cuda:
        m.cuda()

    # 读取 测试图片并转为 RGB通道
    img = Image.open(imgfile).convert('RGB')
    # 测试图像 调整尺度，以便输入网络
    sized = img.resize((m.width, m.height))
    # 统计第二次运行结果 的时间更稳定，更具代表性？
    for i in range(2):
        start = time.time()
        #默认CPU
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    # 加载类别名称，为 bbox打类别标签
    class_names = load_class_names(namesfile)
    # 将bbox及类别 绘制到 测试图像并保存
    plot_boxes(img, boxes, 'img/predictions.jpg', class_names)

if __name__ == '__main__':

    cfgfile='cfg/yolov4.cfg' # 网络框架配置文件
    weightfile='yolov4.weights' # 预训练权重
    imgfile='img/dog.jpg' # 测试图像路径
    # 开始检测
    detect(cfgfile, weightfile, imgfile)

