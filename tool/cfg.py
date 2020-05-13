import sys
import torch
from utils.utils import convert2cpu


def parse_cfg(cfgfile):
    '''
    解析配置文件，将数据读取为 规范格式，并未构建网络
    '''
    blocks = []
    # 读取 配置文件
    fp = open(cfgfile, 'r')
    block = None
    line = fp.readline()
    # 按行读取
    while line != '':
        #  删除 string 字符串末尾的指定字符（默认为空格）
        line = line.rstrip()
        # 该行'' 或者 #被注释，则忽略该行，继续下一行
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        # [ 开始一个新区块
        elif line[0] == '[':
            # block不为空，则直接
            if block:
                blocks.append(block)
            # 每个区块为dict
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        # 读取下一行
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    # blocks为List,由多个dict组成。 每个dict是一个[]区块
    return blocks


def print_cfg(blocks):
    '''
    打印网络框架信息（每层网络结构、卷积核数、输入特征图尺度及通道数、输出特征图尺度及通道数）
    '''
    print('layer     filters    size              input                output');
    # 每个模块输入的特征图尺度
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    # 记录每一层的 卷积通道数
    out_filters = []
    # 记录每一层的 输出特征图尺度
    out_widths = []
    out_heights = []
    # 网络层数计数
    ind = -2
    for block in blocks:
        ind = ind + 1
        # net模块 规定输入图像的尺度，不算网络层数
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            # 计算需要pad的数值
            pad = (kernel_size - 1) // 2 if is_pad else 0
            # 计算 卷积之后的 特征图尺度
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
                height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height,
                filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (
                ind, 'avg', prev_width, prev_height, prev_filters, prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width // stride
            height = prev_height // stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]])
                assert (prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            elif len(layers) == 4:
                print('%5d %-6s %d %d %d %d' % (ind, 'route', layers[0], layers[1], layers[2], layers[3]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]] == out_widths[layers[2]] == out_widths[layers[3]])
                assert (prev_height == out_heights[layers[1]] == out_heights[layers[2]] == out_heights[layers[3]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + out_filters[
                    layers[3]]
            else:
                print("route error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                        sys._getframe().f_code.co_name, sys._getframe().f_lineno))

            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))


def load_conv(buf, start, conv_model):
    '''
    加载 yolov4.weights 中卷积层的预训练权重
    :param buf: yolov4.weights权重
    :param start: 该卷积层权重 位于yolov4.weights的位置下标处
    :param conv_model: 该层卷积层参数
    :return:
    '''
    # numel()  返回tensor的元素个数
    # num_w卷积层的权重个数    num_b偏置项个数
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    #  读取卷积层 权重(有 偏置项)
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    return start



def load_conv_bn(buf, start, conv_model, bn_model):
    '''
    加载 yolov4.weights 中卷积层和BN的预训练权重
    :param buf:   yolov4.weights权重
    :param start:   该卷积层权重 位于yolov4.weights的位置下标处
    :param conv_model: 该层卷积层参数
    :param bn_model: BN参数
    :return:
    '''
    # numel()  返回tensor的元素个数
    # num_w卷积层的权重个数    num_b偏置项个数
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    # 读取bn 权重
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    # 读取卷积层 权重(无 偏置项)
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).reshape(conv_model.weight.data.shape));
    start = start + num_w
    # 读取完成后，下标后移
    return start



def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]));
    start = start + num_w
    return start



