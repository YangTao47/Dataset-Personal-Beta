import numpy as np
import json
''''*!
 * file Yolo3-dataset   output 32x32  three-anchor
 * date 2018/03/26
 *
 * author Yang, Tao
 * Contact: 627871875@qq.com
 *
 *
 * note
*'''
# from yolo3 import model

path = 'D:/dataset/train.json'

'''                                   reference                                                         '''
def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])  # [x,y,w,h]

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


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
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # shape=(m, T, 2)

    m = true_boxes.shape[0]  # 图片数量
    # grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    grid_shapes = [32, 32]
    a = len(anchor_mask[0])
    y_true = np.zeros((m, grid_shapes[0], grid_shapes[0], len(anchor_mask[0]), 5 + num_classes), dtype='float32')

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # dims = 1 3 2
    anchor_maxes = anchors / 2.  # 这里还有3个anchor
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # 归一化
        true_boxes[..., 0:2] = boxes_xy / input_shape[b, 0]
        true_boxes[..., 2:4] = boxes_wh / input_shape[b, 1]
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]  # num_ground_box 2
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

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
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
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs)[1:3] * 8, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs)[1:3], K.dtype(y_true[0]))]
    loss = 0
    m = K.shape(yolo_outputs)[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs))

    object_mask = y_true[..., 4:5]  # confidence
    true_class_probs = y_true[..., 5:]  # class

    grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs,
                                                 anchors[anchor_mask[0]], num_classes, input_shape, calc_loss=True)
    pred_box = K.concatenate([pred_xy, pred_wh])

    # Darknet raw box to calculate loss.
    raw_true_xy = y_true[..., :2] * grid_shapes[::-1] - grid
    raw_true_wh = K.log(y_true[..., 2:4] / anchors[anchor_mask[0]] * input_shape[::-1])
    raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
    box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]  # area

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)  # all images label
    object_mask_bool = K.cast(object_mask, 'bool')

    def loop_body(b, ignore_mask):
        true_box = tf.boolean_mask(y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])  # object_mask_bool (confidence)
        iou = box_iou(pred_box[b], true_box)
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
        return b + 1, ignore_mask

    _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
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
        loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                        message='loss: ')
    return loss


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
preprocess_true_boxes(true_boxes, imgs_shape, anchors, 4)
