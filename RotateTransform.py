#
#   Image and annotations preprocessing for lightnet networks
#   The image transformations work with both Pillow and OpenCV images
#   The annotation transformations work with brambox.annotations.Annotation objects
#   Copyright EAVISE
#

# modified by mileistone
import os
import sys
import math
import random
import collections
import logging as log
import numpy as np
from PIL import Image, ImageOps
from util import BaseTransform, BaseMultiTransform

import codecs
import copy
# import torch
# from torchvision.transforms import functional as F
import inspect

try:
    import cv2
except ImportError:
    log.warn('OpenCV is not installed and cannot be used')
    cv2 = None

__all__ = ['Letterbox', 'RandomCrop', 'RandomCropLetterbox', 'RandomFlip', 'HSVShift', 
           'RandomRotate','RandomPerspective']

#  'ToTensor', 'Normalize', 
# class ToTensor(object):
#     def __call__(self, image, boxes, classes):
#         return F.to_tensor(image), boxes, classes


# class Normalize(object):
#     def __init__(self, mean, std, to_bgr255=True):
#         self.mean = mean
#         self.std = std
#         self.to_bgr255 = to_bgr255

#     def __call__(self, image, boxes, classes):
#         if self.to_bgr255:
#             image = image[[2, 1, 0]] * 255
#         image = F.normalize(image, mean=self.mean, std=self.std)
#         return image, boxes, classes



class Letterbox(BaseMultiTransform):
    """ Transform images and annotations to the right network dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, netInput_size=(800, 800)):
        super().__init__(netInput_size=netInput_size)
        # if self.dimension is None and self.dataset is None:
        #     raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.pad = None
        self.scale = None
        self.fill_color = 127
    

    def __call__(self, data, boxes=None, classes=None):
        # aug data
        if data is None:
            return None
        elif isinstance(data, Image.Image):
            data = self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            data =self._tf_cv(data)
        else:
            log.error(f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
        # aug label
        if boxes is not None and (classes is not None): 
            boxes, classes = self._tf_anno(boxes, classes)
            return data, boxes, classes
        return data

    def _tf_pil(self, img):
        """ Letterbox an image to fit in the network pil lib"""
        net_w, net_h = self.netInput_size
        im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            resample_mode = Image.NEAREST #Image.BILINEAR if self.scale > 1 else Image.ANTIALIAS
            img = img.resize((int(self.scale*im_w), int(self.scale*im_h)), resample_mode)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,)*channels)
        return img

    def _tf_cv(self, img):
        """ Letterbox and image to fit in the network """
        net_w, net_h = self.netInput_size
        im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        channels = img.shape[2] if len(img.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = cv2.copyMakeBorder(img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=(self.fill_color,)*channels)
        return img

    def _tf_anno(self, boxes, classes):
        """ 
        Change coordinates of an annotation, according to the previous letterboxing 
        anno: cx, cy, w, h, theta
        """
        for anno in boxes:
            if self.scale is not None:
                anno[:8] = anno[0]*self.scale, anno[1]*self.scale, anno[2]*self.scale, anno[3]*self.scale, anno[4]*self.scale, anno[5]*self.scale, anno[6]*self.scale, anno[7]*self.scale
            if self.pad is not None:
                anno[0] += self.pad[0]
                anno[1] += self.pad[1]
                anno[2] += self.pad[0]
                anno[3] += self.pad[1]
                anno[4] += self.pad[0]
                anno[5] += self.pad[1]
                anno[6] += self.pad[0]
                anno[7] += self.pad[1]
                # modify for rotate
                # anno[2] += self.pad[0]
                # anno[3] += self.pad[1]
        # print('annos', len(annos), len(annos[0]))
        # exit(0)
        return boxes, classes


class RandomRotate(BaseMultiTransform):
    """ Take random rotate from the image.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(number or list, optional): Argument passed on to :class:`brambox.boxes.util.modifiers.CropModifier`

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, crop_anno=False, fix_angle=-1, intersection_threshold=0.6, fill_color=127):
        super().__init__(crop_anno=crop_anno, fix_angle=fix_angle, fill_color=fill_color, intersection_threshold=intersection_threshold)

    def __call__(self, data, boxes=None, classes=None):
        # aug data
        if data is None:
            return None
        elif isinstance(data, Image.Image):
            data = self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            data =self._tf_cv(data)
        else:
            log.error(f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
        # aug label
        if boxes is not None and (classes is not None): 
            boxes, classes = self._tf_anno(boxes, classes)
            return data, boxes, classes
        return data
    
    def _get_rotate_angle(self):
        if self.fix_angle > -1:
            self.rotate_angle = self.fix_angle
        else:
            self.rotate_angle = int(random.random()*180)


        # 旋转平移矩阵 [2 3]
        self.RT_matrix = cv2.getRotationMatrix2D((self.im_hw[1]/2.0, self.im_hw[0]/2.0), self.rotate_angle, 1)


    def _tf_pil(self, img):
        """ Take random crop from image """
        im_w, im_h = img.size
        self.im_hw = (im_h, im_w)
        
        self._get_rotate_angle()
        '''逆时针旋转rotate_angle度的新Image图像'''
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

        img = img.rotate(self.rotate_angle, fillcolor=(self.fill_color,)*channels)
        # print(inspect.getargspec(img.rotate))

        return img

    def _tf_cv(self, img):
        """ Take random crop from image """
        im_h, im_w = img.shape[:2]
        self.im_hw = (im_h, im_w)

        self._get_rotate_angle()

        # A2 = np.array([[0.5, 0, w/4], [0, 0.5, h/4]], np.float32)
        # d2 = cv2.warpAffine(image, A2, (w, h), borderValue=125)
        # 在d2的基础上绕中心点做旋转
        
        rotate_img = cv2.warpAffine(img, self.RT_matrix, (im_w, im_h), borderValue=(127,127,127))
        # cv2.imshow('img3', rotate_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return rotate_img

    def _tf_anno(self, boxes, classes):
        """ 
        Change coordinates of an annotation, according to the previous crop 
        annos 4个坐标 8个值
        """
        rotate_angle = self.rotate_angle
        cos_theta = math.cos(rotate_angle)
        sin_theta = math.sin(rotate_angle)
        
        # left 可正可负
        imgpoly = shgeo.Polygon([(0, 0), (self.im_hw[1], 0), (self.im_hw[1], self.im_hw[0]), (0, self.im_hw[1])])
        aug_boxes = []        
        aug_labels = [] 

        for i in range(len(boxes)):
            # 对目标进行旋转
            obj   = boxes[i]
            label = classes[i]

            pt1_x = obj[0] * self.RT_matrix[0][0] + obj[1] * self.RT_matrix[0][1] + self.RT_matrix[0][2]
            pt1_y = obj[0] * self.RT_matrix[1][0] + obj[1] * self.RT_matrix[1][1] + self.RT_matrix[1][2]

            pt2_x = obj[2] * self.RT_matrix[0][0] + obj[3] * self.RT_matrix[0][1] + self.RT_matrix[0][2]
            pt2_y = obj[2] * self.RT_matrix[1][0] + obj[3] * self.RT_matrix[1][1] + self.RT_matrix[1][2]
            
            pt3_x = obj[4] * self.RT_matrix[0][0] + obj[5] * self.RT_matrix[0][1] + self.RT_matrix[0][2]
            pt3_y = obj[4] * self.RT_matrix[1][0] + obj[5] * self.RT_matrix[1][1] + self.RT_matrix[1][2]
            
            pt4_x = obj[6] * self.RT_matrix[0][0] + obj[7] * self.RT_matrix[0][1] + self.RT_matrix[0][2]
            pt4_y = obj[6] * self.RT_matrix[1][0] + obj[7] * self.RT_matrix[1][1] + self.RT_matrix[1][2]
            
            rotate_obj = [pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y]
            rotate_gtpoly = shgeo.Polygon([(pt1_x, pt1_y), (pt2_x, pt2_y), (pt3_x, pt3_y), (pt4_x, pt4_y)])

            if (rotate_gtpoly.area <= 0):
                continue

            # 计算旋转之后的框 与原图范围的交集 对框进行裁剪
            inter_poly, half_iou = calchalf_iou(rotate_gtpoly, imgpoly)

            # print('writing...')
            if (half_iou == 1):
                polyInCrop = polyorig2Crop(1, 1, rotate_obj, right=self.im_hw[1] - 1, down=self.im_hw[0] - 1)
                aug_boxes.append(polyInCrop)
                aug_labels.append(label)
                
            ####################
            #elif (half_iou > 0):
            elif (half_iou > self.intersection_threshold):
                ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                out_poly = list(inter_poly.exterior.coords)[0: -1]

                # 相交区域是三角形
                if len(out_poly) < 4:
                    continue

                out_poly2 = []
                for i in range(len(out_poly)):
                    out_poly2.append(out_poly[i][0])
                    out_poly2.append(out_poly[i][1])

                # 相交区域5个点
                if (len(out_poly) == 5):
                    #print('==========================')
                    out_poly2 = GetPoly4FromPoly5(out_poly2)
                elif (len(out_poly) > 5):
                    """
                        if the cut instance is a polygon with points more than 5, we do not handle it currently
                    """
                    continue
                
                out_poly2 = choose_best_pointorder_fit_another(out_poly2, rotate_obj)
                
                # 根据crop 平移坐标 约束边界
                polyInCrop = polyorig2Crop(1, 1, out_poly2, self.im_hw[1] - 1, self.im_hw[0] - 1)
                aug_boxes.append(polyInCrop)
                aug_labels.append(label)

                # # 约束边界
                # for index, item in enumerate(out_poly2):
                #     if (item <= 1):
                #         out_poly2[index] = 1
                #     elif (item >= self.subsize):
                #         out_poly2[index] = self.subsize
        # print('aug_boxes', aug_boxes)
        return aug_boxes, aug_labels


class RandomPerspective(BaseMultiTransform):
    def __init__(self, anglex_rand=(-45, 45), angley_rand=(-45, 45), anglez_rand=(0, 180),  fov_rand=(10,80), fill_color=(127,127, 127)):
        super().__init__(anglex_rand=anglex_rand, angley_rand=angley_rand, anglez_rand=anglez_rand,  fov_rand=(10,80), fill_color=fill_color)

    def __call__(self, data, boxes=None, classes=None):
        # aug data
        if data is None:
            return None
        # elif isinstance(data, Image.Image):
        #     data = self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            data =self._tf_cv(data)
        else:
            log.error(f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
        # aug label
        # if boxes is not None and (classes is not None): 
        #     boxes, classes = self._tf_anno(boxes, classes)
            return data, boxes, classes
        return data

    def rad(self, x):
        return x * np.pi / 180
    
    def get_warpR(self, w, h):
        anglex = random.randint(self.anglex_rand[0], self.anglex_rand[1])
        angley = random.randint(self.angley_rand[0], self.angley_rand[1])
        anglez = random.randint(self.anglez_rand[0], self.anglez_rand[1])
        fov = random.randint(self.fov_rand[0], self.fov_rand[1])

        # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
        z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(self.rad(fov / 2))
        # 齐次变换矩阵
        rx = np.array([[1, 0, 0, 0],
                    [0, np.cos(self.rad(anglex)), -np.sin(self.rad(anglex)), 0],
                    [0, -np.sin(self.rad(anglex)), np.cos(self.rad(anglex)), 0, ],
                    [0, 0, 0, 1]], np.float32)
    
        ry = np.array([[np.cos(self.rad(angley)), 0, np.sin(self.rad(angley)), 0],
                    [0, 1, 0, 0],
                    [-np.sin(self.rad(angley)), 0, np.cos(self.rad(angley)), 0, ],
                    [0, 0, 0, 1]], np.float32)
    
        rz = np.array([[np.cos(self.rad(anglez)), np.sin(self.rad(anglez)), 0, 0],
                    [-np.sin(self.rad(anglez)), np.cos(self.rad(anglez)), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], np.float32)
    
        r = rx.dot(ry).dot(rz)
    
        # 四对点的生成
        pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    
        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, h, 0, 0], np.float32) - pcenter
        p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    
        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)
    
        list_dst = [dst1, dst2, dst3, dst4]
    
        org = np.array([[0, 0],
                        [w, 0],
                        [0, h],
                        [w, h]], np.float32)
    
        dst = np.zeros((4, 2), np.float32)
    
        # 投影至成像平面
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
    
        warpR = cv2.getPerspectiveTransform(org, dst)
        return warpR

    def _tf_pil(self, img):
        pass

    def _tf_cv(self, img):
        """ Take random crop from image """
        im_h, im_w = img.shape[:2]
        warpR = self.get_warpR(im_h, im_w)
        warpR_img = cv2.warpPerspective(img, warpR, (im_h, im_w), borderValue=self.fill_color)
        return warpR_img


class RandomCrop(BaseMultiTransform):
    """ Take random crop from the image.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(number or list, optional): Argument passed on to :class:`brambox.boxes.util.modifiers.CropModifier`

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, jitter=0.25, fix_crop=None, intersection_threshold=0.5, fill_color=127):
        super().__init__(jitter=jitter, fix_crop=fix_crop, fill_color=fill_color, intersection_threshold=intersection_threshold)


    def __call__(self, data, boxes=None, classes=None):
        # aug data
        if data is None:
            return None
        elif isinstance(data, Image.Image):
            data = self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            data =self._tf_cv(data)
        else:
            log.error(f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
        # aug label
        if boxes is not None and (classes is not None): 
            boxes, classes = self._tf_anno(boxes, classes)
            return data, boxes, classes
        return data

    def _tf_pil(self, img):
        """ Take random crop from image """
        im_w, im_h = img.size
        self._get_crop(im_w, im_h)

        if self.if_crop:
            crop_w = self.crop[2] - self.crop[0]
            crop_h = self.crop[3] - self.crop[1]
            img_np = np.array(img)
            channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

            img = img.crop((max(0, self.crop[0]), max(0, self.crop[1]), min(im_w, self.crop[2]-1), min(im_h, self.crop[3]-1)))
            img_crop = Image.new(img.mode, (crop_w, crop_h), color=(self.fill_color,)*channels)
            img_crop.paste(img, (max(0, -self.crop[0]), max(0, -self.crop[1])))

            return img_crop
        else:
            return img

    def _tf_cv(self, img):
        """ Take random crop from image """
        
        # print(img.shape[:2])
        # exit(0)
        im_h, im_w = img.shape[:2]
        self._get_crop(im_w, im_h)

        if self.if_crop:
            crop_w = self.crop[2] - self.crop[0]
            crop_h = self.crop[3] - self.crop[1]
            img_crop = np.ones((crop_h, crop_w) + img.shape[2:], dtype=img.dtype) * self.fill_color

            src_x1 = max(0, self.crop[0])
            src_x2 = min(self.crop[2], im_w)
            src_y1 = max(0, self.crop[1])
            src_y2 = min(self.crop[3], im_h)
            dst_x1 = max(0, -self.crop[0])
            dst_x2 = crop_w - max(0, self.crop[2]-im_w)
            dst_y1 = max(0, -self.crop[1])
            dst_y2 = crop_h - max(0, self.crop[3]-im_h)
            img_crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
            return img_crop
        return img

    def _get_crop(self, im_w, im_h):
        if self.fix_crop is not None:
            # 由setattr 方法定义裁剪区域
            self.crop  = (self.fix_crop[0], self.fix_crop[1], im_w-self.fix_crop[0], im_h-self.fix_crop[1])
            # print('self.crop', self.crop)
            self.if_crop = True
            return
        elif random.random() > 0.5:
            self.if_crop = False
            return 
        else:
            self.if_crop = True
            
            dw, dh = int(im_w* self.jitter), int(im_h* self.jitter)
            crop_left   = random.randint(-dw, dw)
            crop_right  = random.randint(-dw, dw)
            crop_top    = random.randint(-dh, dh)
            crop_bottom = random.randint(-dh, dh)
            
            self.crop  = (crop_left, crop_top, im_w-crop_right, im_h-crop_bottom)

        

    def _tf_anno(self, boxes, classes):
        """ 
        Change coordinates of an annotation, according to the previous crop 
        annos 4个坐标 8个值
        """
        if self.if_crop:
            # 如果标注框被crop 区域裁剪 计算得到相交区域的范围
            crop_left = self.crop[0]
            crop_up   = self.crop[1]
            crop_right= self.crop[2]
            crop_down = self.crop[3]
            # 裁剪之后的图像大小
            
            # left 可正可负
            imgpoly = shgeo.Polygon([(crop_left, crop_up), (crop_right, crop_up), (crop_right, crop_down),(crop_left, crop_down)])
            boxes_aug = []
            labels_aug = [] 

            for i in range(len(boxes)):
                obj   = boxes[i]
                label = classes[i]

                gtpoly = shgeo.Polygon([(obj[0], obj[1]),
                                        (obj[2], obj[3]),
                                        (obj[4], obj[5]),
                                        (obj[6], obj[7])])
                # print('area',gtpoly.area)
                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = calchalf_iou(gtpoly, imgpoly)
                
                if (half_iou == 1):
                    polyInCrop = polyorig2Crop(crop_left + 1, crop_up + 1, obj, crop_right - 1, crop_down - 1)
                    boxes_aug.append(polyInCrop)
                    labels_aug.append(label)

                ####################
                #elif (half_iou > 0):
                elif (half_iou > self.intersection_threshold):
                    ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    inter_poly
                    out_poly = list(inter_poly.exterior.coords)[0: -1]

                    # 相交区域是三角形
                    if len(out_poly) < 4:
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])
                    
                    # 相交区域5个点
                    if (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    
    
                    cnt = np.array([[out_poly2[0][0], out_poly2[0][1]], 
                                    [out_poly2[1][0], out_poly2[1][1]], 
                                    [out_poly2[2][0], out_poly2[2][1]], 
                                    [out_poly2[3][0], out_poly2[3][1]]], dtype=np.int32) # 必须是array数组的形式
                    rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                    
                    if min(rect[1][0], rect[1][1]) < 5:
                        continue

                    out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj)
                    
                    # 根据crop 平移坐标 约束边界
                    polyInCrop = polyorig2Crop(crop_left + 1, crop_up + 1, out_poly2, crop_right - 1, crop_down - 1)
                    boxes_aug.append(polyInCrop)
                    labels_aug.append(label)

                    # # 约束边界
                    # for index, item in enumerate(out_poly2):
                    #     if (item <= 1):
                    #         out_poly2[index] = 1
                    #     elif (item >= self.subsize):
                    #         out_poly2[index] = self.subsize
                    
            return boxes_aug, labels_aug
            
        else:
            return boxes, classes


# no use
class RandomCropLetterbox(BaseMultiTransform):
    """ Take random crop from the image.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(number or list, optional): Argument passed on to :class:`brambox.boxes.util.modifiers.CropModifier`

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dataset, jitter, fill_color=127):
        super().__init__(dataset=dataset, jitter=jitter, fill_color=fill_color)
        self.crop_info = None
        self.output_w = None
        self.output_h = None

    def __call__(self, data):
        if data is None:
            return None
        elif isinstance(data, collections.Sequence):
            return self._tf_anno(data)
        elif isinstance(data, Image.Image):
            return self._tf_pil(data)
        else:
            log.error(f'RandomCrop only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    def _tf_pil(self, img):
        """ Take random crop from image """
        self.output_w, self.output_h = self.dataset.input_dim
        #print('output shape: %d, %d' % (self.output_w, self.output_h))
        orig_w, orig_h = img.size
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        dw = int(self.jitter * orig_w)
        dh = int(self.jitter * orig_h)
        new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
        scale = random.random()*(2-0.25) + 0.25
        if new_ar < 1:
            nh = int(scale * orig_h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * orig_w)
            nh = int(nw / new_ar)

        if self.output_w > nw:
            dx = random.randint(0, self.output_w - nw)
        else:
            dx = random.randint(self.output_w - nw, 0)

        if self.output_h > nh:
            dy = random.randint(0, self.output_h - nh)
        else:
            dy = random.randint(self.output_h - nh, 0)

        nxmin = max(0, -dx)
        nymin = max(0, -dy)
        nxmax = min(nw, -dx + self.output_w - 1)
        nymax = min(nh, -dy + self.output_h - 1)
        sx, sy = float(orig_w)/nw, float(orig_h)/nh
        orig_xmin = int(nxmin * sx)
        orig_ymin = int(nymin * sy)
        orig_xmax = int(nxmax * sx)
        orig_ymax = int(nymax * sy)
        orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
        orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
        output_img = Image.new(img.mode, (self.output_w, self.output_h), color=(self.fill_color,)*channels)
        output_img.paste(orig_crop_resize, (0, 0))
        self.crop_info = [sx, sy, nxmin, nymin, nxmax, nymax]
        return output_img

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous crop """
        sx, sy, crop_xmin, crop_ymin, crop_xmax, crop_ymax = self.crop_info
        for i in range(len(annos)-1, -1, -1):
            anno = annos[i]
            anno_w = anno[2] - anno[0]
            anno_h = anno[3] - anno[1]

            x1 = max(crop_xmin, int(anno[0]/sx))
            x2 = min(crop_xmax, int((anno[0]+anno_w)/sx))
            y1 = max(crop_ymin, int(anno[1]/sy))
            y2 = min(crop_ymax, int((anno[1]+anno_h)/sy))
            w = x2-x1
            h = y2-y1

            if w <= 2 or h <= 2: # or w*h/(anno.width*anno.height/sx/sy) <= 0.5:
                del annos[i]
                continue

            annos[i,0] = x1 - crop_xmin
            annos[i,1] = y1 -crop_ymin
            annos[i,2] = annos[i,0]+ w
            annos[i,3] = annos[i,1]+ h
        return annos


class RandomFlip(BaseMultiTransform):
    """ Randomly flip image.

    Args:
        threshold (Number [0-1]): Chance of flipping the image

    Note:
        Create 1 RandomFlip object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, flipProb):
        self.flipProb = flipProb
        self.flip = False
        self.im_w = None

    def __call__(self, data, boxes=None, classes=None):
        # aug data
        if data is None:
            return None
        elif isinstance(data, Image.Image):
            data = self._tf_pil(data)
        elif isinstance(data, np.ndarray):
            data =self._tf_cv(data)
        else:
            log.error(f'Letterbox only works with <brambox annotation lists>, <PIL images> or <OpenCV images> [{type(data)}]')
        # aug label
        if boxes is not None and (classes is not None): 
            boxes, classes = self._tf_anno(boxes, classes)
            return data, boxes, classes
        return data

    def _tf_pil(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.size[0]
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _tf_cv(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.shape[1]
        if self.flip:
            img = cv2.flip(img, 1)
        return img

    def _get_flip(self):
        self.flip = random.random() < self.flipProb

    def _tf_anno(self,  boxes, classes):
        if self.flip and self.im_w is not None:
            boxes_aug = []
            for i in range(len(boxes)):
                obj   = boxes[i]
                # anno_w = anno[2] - anno[0]
                # anno[0] = self.im_w - anno[0] - anno_w
                obj[0] = self.im_w - obj[0]
                obj[2] = self.im_w - obj[2]
                obj[4] = self.im_w - obj[4]
                obj[6] = self.im_w - obj[6]
                boxes_aug.append([obj[0], obj[1], obj[6], obj[7], obj[4], obj[5], obj[2], obj[3]])
                # boxes_aug.append([obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7]])
            boxes = boxes_aug
        # print('RandomFlip', len(anno))
        return boxes, classes


class HSVShift(BaseTransform):
    """ Perform random HSV shift on the RGB data.

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue
        saturation (Number): Random number between 1,saturation is used to shift the saturation; 50% chance to get 1/dSaturation in stead of dSaturation
        value (Number): Random number between 1,value is used to shift the value; 50% chance to get 1/dValue in stead of dValue

    Warning:
        If you use OpenCV as your image processing library, make sure the image is RGB before using this transform.
        By default OpenCV uses BGR, so you must use `cvtColor`_ function to transform it to RGB.

    .. _cvtColor: https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gself.RT_matrix97ae87e1288a81d2363b61574eb8cab
    """
    def __init__(self, hue, saturation, value):
        super().__init__(hue=hue, saturation=saturation, value=value)

    @classmethod
    def apply(cls, data, hue, saturation, value):
        dh = random.uniform(-hue, hue)
        ds = random.uniform(1, saturation)
        if random.random() < 0.5:
            ds = 1/ds
        dv = random.uniform(1, value)
        if random.random() < 0.5:
            dv = 1/dv

        if data is None:
            return None
        elif isinstance(data, Image.Image):
            return cls._tf_pil(data, dh, ds, dv)
        elif isinstance(data, np.ndarray):
            return cls._tf_cv(data, dh, ds, dv)
        else:
            log.error(f'HSVShift only works with <PIL images> or <OpenCV images> [{type(data)}]')
            return data

    @staticmethod
    def _tf_pil(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.convert('HSV')
        channels = list(img.split())

        def change_hue(x):
            x += int(dh * 255)
            if x > 255:
                x -= 255
            elif x < 0:
                x += 0
            return x

        channels[0] = channels[0].point(change_hue)
        channels[1] = channels[1].point(lambda i: min(255, max(0, int(i*ds))))
        channels[2] = channels[2].point(lambda i: min(255, max(0, int(i*dv))))

        img = Image.merge(img.mode, tuple(channels))
        img = img.convert('RGB')
        return img

    @staticmethod
    def _tf_cv(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        print('beforre hsvshift', img.shape())

        def wrap_hue(x):
            x[x >= 360.0] -= 360.0
            x[x < 0.0] += 360.0
            return x

        img[:, :, 0] = wrap_hue(hsv[:, :, 0] + (360.0 * dh))
        img[:, :, 1] = np.clip(ds * img[:, :, 1], 0.0, 1.0)
        img[:, :, 2] = np.clip(dv * img[:, :, 2], 0.0, 1.0)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = (img * 255).astype(np.uint8)


        return img


# class BoxToTensor(BaseTransform):
#     """ Converts a list of brambox annotation objects to a tensor.

#     Args:
#         dimension (tuple, optional): Default size of the transformed images, expressed as a (width, height) tuple; Default **None**
#         dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
#         max_anno (Number, optional): Maximum number of annotations in the list; Default **50**
#         class_label_map (list, optional): class label map to convert class names to an index; Default **None**

#     Return:
#         torch.Tensor: tensor of dimension [max_anno, 5] containing [class_idx,center_x,center_y,width,height] for every detection

#     Warning:
#         If no class_label_map is given, this function will first try to convert the class_label to an integer. If that fails, it is simply given the number 0.
#     """
#     def __init__(self, dimension=None, dataset=None, max_anno=50, class_label_map=None):
#         super().__init__(dimension=dimension, dataset=dataset, max_anno=max_anno, class_label_map=class_label_map)
#         if self.dimension is None and self.dataset is None:
#             raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')
#         if self.class_label_map is None:
#             log.warn('No class_label_map given. If the class_labels are not integers, they will be set to zero.')

#     def __call__(self, data):
#         if self.dataset is not None:
#             dim = self.dataset.input_dim
#         else:
#             dim = self.dimension
#         return self.apply(data, dim, self.max_anno)

#     @classmethod
#     def apply(cls, data, dimension, max_anno=None):
#         if not isinstance(data, collections.Sequence):
#             raise TypeError(f'BramboxToTensor only works with <brambox annotation list> [{type(data)}]')

#         anno_np = np.array([cls._tf_anno(anno, dimension) for anno in data], dtype=np.float32)

#         if max_anno is not None:
#             anno_len = len(data)
#             if anno_len > max_anno:
#                 raise ValueError(f'More annotations than maximum allowed [{anno_len}/{max_anno}]')

#             z_np = np.zeros((max_anno-anno_len, 5), dtype=np.float32)
#             z_np[:, 0] = -1

#             if anno_len > 0:
#                 return torch.from_numpy(np.concatenate((anno_np, z_np)))
#             else:
#                 return torch.from_numpy(z_np)
#         else:
#             return torch.from_numpy(anno_np)

#     @staticmethod
#     def _tf_anno(anno, dimension):
#         """ Transforms brambox annotation to list """
#         net_w, net_h = dimension
#         anno_w = anno[2] - anno[0]
#         anno_h = anno[3] - anno[1]

#         cx = (anno[0] + (anno_w / 2)) / net_w
#         cy = (anno[1] + (anno_h / 2)) / net_h
#         w = anno_w / net_w
#         h = anno_h / net_h
#         return [cx, cy, w, h, anno[4]]

