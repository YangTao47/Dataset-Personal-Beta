''''*!
    * \date 2021/12/4
    *
    * \author Yang, Tao
    * Contact: 627871875@qq.com
    *
    *
    * \note
*'''
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import imgaug as ia
import imgaug.augmenters as iaa


def bounding_box_example():
    # 读取图片
    img = cv2.imread("C:\\Users\\Leaper\\Desktop\\train\\105.jpg")
    # 变换通道
    img = img[:, :, ::-1]
    bbs = BoundingBoxesOnImage([
        # 目标在图片上的位置
        BoundingBox(x1=10, y1=10, x2=10, y2=10)
    ], shape=img.shape)
    # 数据增强
    seq = iaa.Sequential([
        iaa.AdditiveGaussianNoise(scale=0.05 * 255),
        iaa.Affine(translate_px={"x": (10, 100)})
    ])
    # 变换后的图片和box
    img_aug, bbs_aug = seq(image=img, bounding_boxes=bbs)
    # 绘制变换前box在图片上的位置
    img_before = bbs.draw_on_image(img, size=2)
    # 绘制图片变换后box在图片上的位置
    img_after = bbs_aug.draw_on_image(img_aug, size=2, color=[255, 0, 0])
    ia.show_grid([img_before, img_after], rows=1, cols=2)


bounding_box_example()
