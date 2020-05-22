import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.autograd import Variable



def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    #boxes [XXX,7]  0~3:预测bbox坐标  4:包含物体的概率   5:分类概率最大值 6:分类概率最大值的对应下标
    # 没有符合条件的bbox,预测为空
    if len(boxes) == 0:
        return boxes
    # 取出 每个bbox 对应的包含物体的概率
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    # sortIds是 所有bbox 包含物体概率值 由于 上一步的1-概率 导致结果是 从大到小排序的下标
    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    # 按照 包含物体概率值从大到小的顺序  将boxes装入out_boxes.
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        # 排除多余bbox
        if box_i[4] > 0:
            out_boxes.append(box_i)
            # 在装入的过程中将当前bbox 和 剩余bbox进行NMS非极大值抑制，排除多余bbox
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                # 若为多余bbox,则 将  包含物体概率值=0.下次循环时过滤掉
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    '''
     # 对某一尺度特征图 提取bbox
    :param output:  19x19 38x38 76x76其中一种尺度特征图  eg:[1,255,76,76]
    :param conf_thresh: 类别置信度
    :param num_classes:   80：COCO类别数
    :param anchors: anchors具体尺度
    :param num_anchors:  一个单元格要预测的不同尺度anchor数量
    :param only_objectness:
    :param validation:
    :return:
    '''
    anchor_step = len(anchors) // num_anchors
    # eg：若output为[255,76,76]，则扩展一维，变为[1,255,76,76]
    if len(output.shape) == 3:
        output = np.expand_dims(output, axis=0)
    batch = output.shape[0]
    # 验证特征图是否合规    每个单元格（特征图的一个元素）预测3（num_anchors）个anchor, 每个anchor需要同时提供 4+1+80（4：anchor坐标，1:包含物体的概率以过滤背景，80：数据集的类别）
    # 网络预测[1,255,76,76]   [76,76]指一张特征图   255:(4+1+80)x3网络预测的坐标和分类概率
    assert (output.shape[1] == (4+1 + num_classes) * num_anchors)
    h = output.shape[2]
    w = output.shape[3]
    # 保存最终bbox
    all_boxes = []
    # output[85,17328]
    output = output.reshape(batch * num_anchors, 5 + num_classes, h * w).transpose((1, 0, 2)).reshape(
        5 + num_classes,
        batch * num_anchors * h * w)
    # grid_x,grid_y均是0~75   预测时的左上角坐标
    grid_x = np.expand_dims(np.expand_dims(np.linspace(0, w - 1, w), axis=0).repeat(h, 0), axis=0).repeat(
        batch * num_anchors, axis=0).reshape(
        batch * num_anchors * h * w)
    grid_y = np.expand_dims(np.expand_dims(np.linspace(0, h - 1, h), axis=0).repeat(w, 0).T, axis=0).repeat(
        batch * num_anchors, axis=0).reshape(
        batch * num_anchors * h * w)
    # xs，ys 即单元格中anchor的中心点坐标=网格左上角坐标 + 相对网格左上角坐标的偏移量（网络预测的anchor坐标 前两个即相对网格左上角坐标的偏移量）
    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y

    anchor_w = np.array(anchors).reshape((num_anchors, anchor_step))[:, 0]
    anchor_h = np.array(anchors).reshape((num_anchors, anchor_step))[:, 1]
    anchor_w = np.expand_dims(np.expand_dims(anchor_w, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * num_anchors * h * w)
    anchor_h = np.expand_dims(np.expand_dims(anchor_h, axis=1).repeat(batch, 1), axis=2) \
        .repeat(h * w, axis=2).transpose(1, 0, 2).reshape(batch * num_anchors * h * w)
    # ws,hs 即 在anchor中心点上的宽、高
    ws = np.exp(output[2]) * anchor_w
    hs = np.exp(output[3]) * anchor_h

    # 包含物体的概率，以过滤背景
    det_confs = sigmoid(output[4])
    # 80个类别的分类概率
    cls_confs = softmax(output[5:5 + num_classes].transpose(1, 0))
    # 80个类别概率中  最大的概率值
    cls_max_confs = np.max(cls_confs, 1)
    # 80个类别概率中  最大概率值的下标，用于确定具体类别名称
    cls_max_ids = np.argmax(cls_confs, 1)

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    for b in range(batch):
        # 一张输入图片 对应的预测bbox
        boxes = []
        for cy in range(h):
            for cx in range(w):
                # [76,76]特征图的每个元素  预测3个anchor
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    # 找到对应的 包含物体的概率det_conf
                    det_conf = det_confs[ind]
                    # 包含物体的概率
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
                    # 当 包含物体的概率  大于设定阈值时，则需要该anchor

                    if conf > conf_thresh:
                        # anchor中心点坐标 bcx bcy及 对应的宽高bw bh
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        # 该anchor 在80个分类中概率值的最大值
                        cls_max_conf = cls_max_confs[ind]
                        # 该anchor 在80个分类中概率值的最大值的下标
                        cls_max_id = cls_max_ids[ind]
                        # [bcx / w, bcy / h, bw / w, bh / h]:预测bbox坐标       det_conf:包含物体的概率   cls_max_conf:分类概率最大值 cls_max_id:分类概率最大值的对应下标
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes

def plot_boxes(img, boxes, savename=None, class_names=None):
    '''
    将bbox及类别 绘制到 测试图像并保存
    :param img: 测试图像
    :param boxes: bbox结果
    :param savename: 测试结果的图像名字
    :param class_names: 类别数组
    :return:
    '''
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img




def load_class_names(namesfile):
    '''
    加载类别名称
    '''
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names



def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    '''
    预测阶段
    :param model: 模型
    :param img: 预测图像
    :param conf_thresh: 类别阈值
    :param nms_thresh:  非极大值抑制阈值，用于去掉 重复且较低质量的bbox
    :param use_cuda:   0:CPU   1:GPU
    '''
    # 调整为 推理模式，不记录梯度，速度更快 占用更少
    model.eval()
    t0 = time.time()

    # 读取图像，调整格式并归一化
    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        # batch:1  channel:3
        img = img.view(1, 3, height, width)
        # 除以255，归一化
        img = img.float().div(255.0)
    elif type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    elif type(img) == np.ndarray and len(img.shape) == 4:
        img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    if use_cuda:
        img = img.cuda()


    # 网络输出 三组不同尺度特征图（19x19 38x38 76x76）
    list_boxes = model(img)
    # 任意9个簇和3个比例
    anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    num_anchors = 9
    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    strides = [8, 16, 32]
    anchor_step = len(anchors) // num_anchors
    boxes = []
    # 对每种尺度特征图 提取bbox
    for i in range(3):
        masked_anchors = []
        for m in anchor_masks[i]:
            masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
        masked_anchors = [anchor / strides[i] for anchor in masked_anchors]
        boxes.append(get_region_boxes(list_boxes[i].data.numpy(), 0.6, 80, masked_anchors, len(anchor_masks[i])))
    # 多张图像同时预测
    if img.shape[0] > 1:
        bboxs_for_imgs = [ boxes[0][index] + boxes[1][index] + boxes[2][index] for index in range(img.shape[0])]
        # 分别对每一张图片的结果进行nms
        boxes = [nms(bboxs, nms_thresh) for bboxs in bboxs_for_imgs]
    # 单张图像预测
    else:
        boxes = boxes[0][0] + boxes[1][0] + boxes[2][0]
        # pytorch现已提供 原生NMS方法，我以后修改
        boxes = nms(boxes, nms_thresh)
    return boxes

