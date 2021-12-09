''''*!
    * \date 2021/12/4
    *
    * \author Yang, Tao
    * Contact: 627871875@qq.com
    *
    *
    * \note
*'''
import numpy as np
import json
import tensorflow as tf
from keras import backend as K
from keras.models import Model

# from yolo3 import model

path = 'D:/dataset/train.json'


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, np.float32)

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], np.float32)
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], np.float32)
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_loss(yolo_outputs, y_true, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors) // 3  # default setting
    # yolo_outputs = args[:num_layers]
    # y_true = args[num_layers:]
    a = K.shape(yolo_outputs)[1:3] * 8
    # TODO 这里的input——shape 没那么简单
    input_shape = K.cast(K.shape(yolo_outputs)[1:3] * 8, np.float32)
    grid_shapes = [K.cast(K.shape(yolo_outputs)[1:3], np.float32)]
    loss = 0
    m = K.shape(yolo_outputs)[0]  # batch size, tensor
    mf = K.cast(m, np.float32)

    object_mask = y_true[..., 4:5]  # confidence
    true_class_probs = y_true[..., 5:]  # class
    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs,
                                                 anchors[0:3], num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    a = y_true[..., :2]
    raw_true_xy = y_true[..., :2] * grid_shapes[::-1] - grid
    raw_true_wh = K.log(y_true[..., 2:4] / anchors[0:3] * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
    box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]  # area

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(np.float32, size=1, dynamic_size=True)  # all images label
    object_mask_bool = K.cast(object_mask, 'bool')

    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])  # object_mask_bool (confidence)
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
        return b + 1, ignore_mask

    _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)

    # K.binary_crossentropy is helpful to avoid exp overflow.
    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                   from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                      (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    loss += xy_loss + wh_loss + confidence_loss + class_loss
    if print_loss:
        loss = tf.print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                        message='loss: ')
    return loss


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)  batch_size max_boxes=20  5
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    anchor_mask = [[0, 1, 2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # shape=(m, 5, 2)

    m = true_boxes.shape[0]  # 图片数量
    # grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    grid_shapes = [32, 32]
    a = len(anchor_mask[0])
    y_true = np.zeros((m, grid_shapes[0], grid_shapes[0], len(anchor_mask[0]), 5 + num_classes), dtype='float32')

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # dims = 1 3 2
    anchor_maxes = anchors / 2.  # 这里还有3个anchor
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0 #(m,5)

    for b in range(m):
        # 归一化 + 缩放比例
        true_boxes[..., 0:2] = (boxes_xy / input_shape[b, 0]) * (256 /input_shape[b, 0])
        true_boxes[..., 2:4] = (boxes_wh / input_shape[b, 1]) * (256 /input_shape[b, 0])
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]] * (256 /input_shape[b, 0]) # num_ground_box 2  （2,2） #这个尺寸不定，原图中有多少个groundTruth，就有几个WH
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)  # num_ground_box 1  2
        box_maxes = wh / 2.
        box_mins = -box_maxes  # num_ground_box 1 2

        # 计算交并比（IOU）
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]  # t 1
        anchor_area = anchors[..., 0] * anchors[..., 1]  # 1 3
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # t 3

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            if n in anchor_mask[0]:
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[0]).astype('int32')  # cell_x
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[0]).astype('int32')  # cell_y
                k = anchor_mask[0].index(n)  # n 在anchor_mask的索引
                c = true_boxes[b, t, 4].astype('int32')  # class_id
                y_true[b, j, i, k, 0:4] = true_boxes[b, t, 0:4]  # box  num_img 32 32 anchor_num (5 + num_class)
                y_true[b, j, i, k, 4] = 1  # confidence
                y_true[b, j, i, k, 5 + c] = 1  # class

    return y_true


def read_json(path):
    true_boxes = []
    imgs_shape = []
    with open(path) as f:
        datas = json.load(f)
        for data in datas:  # 每张图像数据
            img_shape = []
            img_shape.append(data["height"])
            img_shape.append(data["width"])
            imgs_shape.append(img_shape)
            box = []
            box_data = np.zeros((5, 5))
            boxes1 = []
            boxes2 = []
            boxes3 = []
            boxes4 = []
            labels = data['label']
            for label in labels:  # 图片标签数据
                if (label["shape_type"] == "rectangle") and (label["label"] == "0"):
                    for p in label['points']:
                        boxes1.append(p[0])
                        boxes1.append(p[1])
                    boxes1.append(0)
                    box.append(boxes1)
                if (label["shape_type"] == "rectangle") and (label["label"] == "1.6"):
                    for p in label['points']:
                        boxes2.append(p[0])
                        boxes2.append(p[1])
                    boxes2.append(1)
                    box.append(boxes2)
                if (label["shape_type"] == "rectangle") and (label["label"] == "10"):
                    for p in label['points']:
                        boxes3.append(p[0])
                        boxes3.append(p[1])
                    boxes3.append(2)
                    box.append(boxes3)
                if (label["shape_type"] == "rectangle") and (label["label"] == "25"):
                    for p in label['points']:
                        boxes4.append(p[0])
                        boxes4.append(p[1])
                    boxes4.append(3)
                    box.append(boxes4)

            box_data[:len(box)] = box
            true_boxes.append(box_data)
    return true_boxes, imgs_shape


true_boxes, imgs_shape = read_json(path)
true_boxes = np.array(true_boxes)
print(true_boxes.shape)
anchors = [[10, 14], [23, 27], [37, 58]]
# TODO Yolo loss
y_true = preprocess_true_boxes(true_boxes, imgs_shape, anchors, 4)
loss = yolo_loss(y_true,y_true, anchors, 4, ignore_thresh=.5, print_loss=False)
a = 1