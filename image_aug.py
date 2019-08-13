#coding:utf-8
import os
import random
import time
import argparse
import glob

import numpy as np
import cv2
import shutil
import math
import multiprocessing

import RotateTransform as r_transform
import sys


im_size = [360, 640]   # 输出图的大小
netInput_size = [800, 800]

aug_func_dict={
    "Letterbox":    r_transform.Letterbox(netInput_size),
    "RandomCrop":   r_transform.RandomCrop(jitter=0.4),
    "RandomRotate": r_transform.RandomRotate(fix_angle=0),
    "RandomHSV":    r_transform.HSVShift(hue=.1, saturation=1.5, value=1.5),
    "RandomPerspective": r_transform.RandomPerspective()
}
# modify this,
func_names = ["RandomPerspective"]

# filter_postfix = ["*.png", "*.jpg"]

def augImg(image_root, save_root, label_root=None):
    images = glob.glob(os.path.join(image_root, '*.png') )
    # 每个进程 one aug_func
    for func_name in func_names:
        p = multiprocessing.Process(target = aug_process, args = (func_name, images, image_root, save_root, label_root))
        p.start()
        
def augImgLabel():
    pass

def aug_process(aug_func_name, images, image_root, save_root, label_root=None):
    save_dir = os.path.join(save_root, aug_func_name)
    if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

    aug_func = aug_func_dict[aug_func_name]
    letter_box = aug_func_dict["Letterbox"]
    print('{}...'.format(aug_func_name))

    for img_name in images:
        prefix, _ = img_name.split('/')[-1].split('.')

        img = cv2.imread(os.path.join(image_root , img_name))
        im_h, im_w, _ = img.shape

        

        # if aug_func_name == "RandomRotate":
        #     for angel in range(0,180, 10):
                # aug_func.set_value("fix_angle", angel)
                # aug_img = aug_func(img)
                # img_outfile = os.path.join(save_dir, "{}_Rotate_{}.png".format(prefix, angel))
                # # print(img_outfile)
                # cv2.imwrite(img_outfile, aug_img)


        # if aug_func_name == "RandomCrop":
        #     for crop_wh in zip(np.linspace(-int(aug_func.jitter *im_w), int(aug_func.jitter *im_w), 10), 
        #                     np.linspace(-int(aug_func.jitter *im_h), int(aug_func.jitter *im_h), 10)):
        #         crop_wh = (int(crop_wh[0]), int(crop_wh[1]))
        #         aug_func.set_value("fix_crop", crop_wh)
                
        #         aug_img = letter_box(aug_func(img))
        #         img_outfile = os.path.join(save_dir, "{}_crop_{}.png".format(prefix, crop_wh[0]))
        #         # print(img_outfile)
        #         cv2.imwrite(img_outfile, aug_img)

        if aug_func_name == "RandomPerspective":
            for idx in range(10):
                aug_img = aug_func(img)
                img_outfile = os.path.join(save_dir, "{}_persp_{}.png".format(prefix, idx))
                # print(img_outfile)
                cv2.imwrite(img_outfile, aug_img)

 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="input images file dir")
    parser.add_argument("-l", "--labels", help="labels file dir")
    parser.add_argument("-s", "--save", help="result save dir")

    args = parser.parse_args()

    if os.path.isdir(args.images) is False:
        print('no such dir:{}'.format(args.images))
        exit(0)
    
    augImg(args.images, args.save, args.labels)








# def crop_images(data, cropW = 416, cropH = 416):
#     '''
#     1.使用滑窗方法裁剪图像
#     2.复制并翻转图像
#       3.对翻转图像以0.5概率进行高斯模糊
#     '''
#     filename, ann = data
    
#     clas, coords = ann
#     if len(coords) == 0:#no data
#         return
#     print("\nfilename",filename)
#     img = Image.open(os.path.join(args.images, filename))
#     # img=img.load()
#     width, height = img.size
#     level = random.randint(10,20)/10.0
#     img = img.resize((int(width*level), int(height*level)),Image.ANTIALIAS)
#     width, height = img.size
#     # print("width, height", width, height)
#     #print(img_file)
#     coords = np.array(coords)


#     '''label:c_x c_y w h'''
#     w = coords[:,2] * width
#     h = coords[:,3] * height
#     x_min = coords[:,0] * width - w/2.0
#     y_min = coords[:,1] * height- h/2.0
#     x_max = x_min + w
#     y_max = y_min + h
#     prefix, postfix = filename.split('.') 
    
# #     in_file = open(os.path.join(label_dir, prefix + '.txt'))
# #     line = in_file.readline()
# #     while line:
# #         word = line.split(' ')
# #         x_min.append(float(word[4])*level)
# #         y_min.append(float(word[5])*level)
# #         x_max.append(float(word[6])*level)
# #         y_max.append(float(word[7])*level)
# #         line = in_file.readline()
# #     in_file.close()

#     if len(x_min) == 0:
#         return

#     # if((width<=2100)|(height<=1300)):
#     # 	return
#     width_num = int(math.ceil(width * 1.0/cropW))
#     height_num = int(math.ceil(height * 1.0/cropH))
#     # print("width_num, height_num", width_num, height_num)

#     k=1
#     for width_step in range(width_num):
#         for height_step in range(height_num):
#             # print('patch', k)
#             x = int(width_step * (cropW - (width_num * cropW - width)*1.0 / (width_num - 1)))
#             y = int(height_step * (cropH -(height_num * cropH-height)*1.0 / (height_num - 1)))
#             #正常裁切
#             img_crop = os.path.join(args.save,"images/{}_{}.png".format(prefix, k))
#             txt_crop = os.path.join(args.save,"labels/{}_{}.txt".format(prefix, k))
#             file_crop = open(txt_crop, 'w+')
#             #反转
#             img_crop_flip = os.path.join(args.save,"images/{}_{}_flip.png".format(prefix, k))
#             txt_crop_flip = os.path.join(args.save,"labels/{}_{}_flip.txt".format(prefix, k))
#             file_crop_flip = open(txt_crop_flip, 'w+')

#             car_exist = False
#             #x_min图像中目标的最小值
#             for i in range(len(x_min)):
#                 '''中心点在一定缓冲区之内'''
#                 c_x = (x_min[i] + x_max[i])/2.0
#                 c_y = (y_min[i] + y_max[i])/2.0
#                 # print('prefix c_x c_y', prefix, c_x, c_y)
#                 if  (c_x >= x-15)  & (c_x <= x+cropW + 15) & (c_y >= y-10) & (c_y <= y + cropH +10):
#                     # print('there are center')
#                     car_exist = True
#                     size = (cropW, cropH)
#                     #正常裁切
#                     box = [max(x_min[i] - x + 1, 0), 
#                            min(x_max[i] - x - 1, cropW), 
#                            max(y_min[i] - y + 1, 0), 
#                            min(y_max[i] - y - 1, cropH) ]
#                     c_x,c_y,w,h = convert(size, box)
#                     file_crop.write(str(clas[i])+ " " + str(c_x) + " "+ str(c_y) + " "+ str(w) + " "+ str(h)+'\n')
#                     #反转
#                     file_crop_flip.write(str(clas[i])+ " " + str(1 - c_x) + " "+ str(1 - c_y) + " "+ str(w) + " "+ str(h)+'\n')

#             if car_exist is not True:
#                 # print("\n labels/{}_{}.txt".format(prefix, k))
#                 file_crop.close()
#                 os.remove(txt_crop)
#                 #反转
#                 file_crop_flip.close()
#                 os.remove(txt_crop_flip)
#                 continue
#             crop_img = img.crop((x, y, x + cropW, y + cropH))
#             crop_img.save(img_crop)

#             #反转
#             flip_img = crop_img.transpose(Image.FLIP_LEFT_RIGHT)
#             if random.random() > 0.5:
#                flip_img = flip_img.filter(ImageFilter.GaussianBlur(radius=2))
#             flip_img.save(img_crop_flip)
#             # crop_img.show()
#             # time.sleep(1)
#             k = k + 1






