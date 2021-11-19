import numpy as np
import cv2
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imgaug as ia
import time
import json

def find_files_with_suffix(target_dir, target_suffix="jpg"):
    """ 查找以 target_suffix 为后缀的文件，并返加 """
    find_jpg = []
    find_json = []
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            #shutil.move(os.path.join(root_path, file), os.path.join('new_path', file))
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == ".jpg":
                find_jpg.append(os.path.join(root_path, file))
            if suffix_name == ".json":
                find_json.append(os.path.join(root_path, file))

    return find_jpg, find_json

def get_kp(json):
    kp = []
    for i in range(6):
       if json['label'][i]['shape_type'] == 'point':
           kp.append(json['label'][i]['points'][0])
    return kp

def get_bodbx(json):
    bodbx = []
    for i in range(6):
       if json['label'][i]['shape_type'] == 'rectangle':
           bodbx.append(json['label'][i]['points'])
    return bodbx

all_jpg, all_json = find_files_with_suffix("C:\\Users\\Leaper\\Desktop\\train")

with open("train.json") as fp:
    data = json.load(fp)

kp = get_kp(data[0])
bodbx = get_bodbx(data[0])

#取出当前秒
a = time.gmtime().tm_sec
#设置随机数种子
ia.seed(time.gmtime().tm_sec)

def dataEnforce():
    #读取图片
    img = cv2.imread("C:\\Users\\Leaper\\Desktop\\train\\105.jpg")
    #变换通道
    img = img[:, :, ::-1]
    kps = KeypointsOnImage([
        Keypoint(x=kp[0][0], y=kp[0][1]),
        Keypoint(x=kp[1][0], y=kp[1][1]),
        Keypoint(x=kp[2][0], y=kp[2][1]),
        Keypoint(x=kp[3][0], y=kp[3][1])
    ], shape=img.shape)
    bbs = BoundingBoxesOnImage([
        #目标在图片上的位置
        BoundingBox(x1=bodbx[0][0][0], y1=bodbx[0][0][1], x2=bodbx[0][1][0], y2=bodbx[0][1][1]),
        BoundingBox(x1=bodbx[1][0][0], y1=bodbx[1][0][1], x2=bodbx[1][1][0], y2=bodbx[1][1][1])
    ],shape=img.shape)
    #数据增强
    seq = iaa.SomeOf(4,[
        iaa.LinearContrast((0.5, 1.5)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.ImpulseNoise(0.05),
        iaa.Salt(0.1),
        #iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1)),随机裁剪
        iaa.BlendAlpha( (0.0, 1.0),foreground=iaa.Add(50), background=iaa.Multiply(0.2)),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.GaussianBlur(sigma=(0, 3.0)),
        iaa.AverageBlur(k=((5, 11), (1, 3))),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.2*255),per_channel=0.5),
        iaa.ScaleX((0.8, 1.2)),
        iaa.ScaleY((0.8, 1.2)),
        iaa.Affine(scale={"x":(0.8,1.2),"y":(0.8,1.2)},
                   rotate=(-180, 180),
                   shear=(-8,8)) ],random_order=True)
    #变换后的图片和box
    img_aug,bbs_aug,kps_aug = seq(image=img,bounding_boxes=bbs, keypoints=kps)
    #绘制变换前box在图片上的位置
    img_before = bbs.draw_on_image(img,size=2)
    #绘制图片变换后box在图片上的位置
    img_after = bbs_aug.draw_on_image(img_aug,size=2,color=[255,0,0])
    img_after = kps_aug.draw_on_image(img_after,size=2,color=[255,0,0])
    ia.show_grid([img_before,img_after],rows=1,cols=2)
    b = 1


for i in range(10):
    dataEnforce()

