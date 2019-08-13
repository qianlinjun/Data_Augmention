#coding:utf-8
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import math

def NN_interpolation(img, dstH, dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=round((i+1)*(scrH*1./dstH))
            scry=round((j+1)*(scrW*1./dstW))
            retimg[i,j]=img[scrx-1,scry-1]
    return retimg

def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH*1./dstH)-1
            scry=(j+1)*(scrW*1./dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    return retimg

def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0

def BiCubic_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH*1./dstH)
            scry=j*(scrW*1./dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg

def localwarp(img, dstH, dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    # left top right down
    area=[100, 80, 180, 130]
    
    centerX, centerY = (area[0]+area[2])/2, (area[1]+area[3])/2
    dist_w, dist_h = 2*(area[2] - area[0]), 2*(area[3]-area[1])
    scale=2
    dist_area = [centerX - dist_w/2, centerY - dist_h/2, centerX + dist_w/2, centerY + dist_h/2]
    
    

    real_radius=50

    dst=100000

    for i in range(dstH):
        for j in range(dstW):
            # 局部插值
            if all([i >= dist_area[1] , i <= dist_area[3] , j >= dist_area[0] , j <= dist_area[2]]):
                src_x = centerX + (j-centerX)/scale
                src_y = centerY + (i-centerY)/scale
            else:
                src_x = j
                src_y = i
            retimg[i,j]=img[src_y-1, src_x-1]

            # 放大镜的凹凸效果
            # distance = math.sqrt((i - centerX)*(i - centerX) + (j - centerY)*(j - centerY))
            # src_x = int(i - centerX)
            # src_y = int(j - centerY)
            # src_x = int(src_x * (math.sqrt(distance) / real_radius))
            # src_y = int(src_y * (math.sqrt(distance) / real_radius))
            # src_x = src_x + centerX
            # src_y = src_y + centerY

            # sin
            # if  j < (dist_w / 2):
            #     delta = j / dist_w * dst * math.sin((i / dist_h) * math.pi)
            # else:
            #     delta = (1 - j / dist_w) * dst * math.sin((i / dist_h) * math.pi)

            # retimg[i,j]=img[i,j-delta]
    return retimg

# #    int w = mat.cols;
#     int h = mat.rows;
#     cv::Mat t = mat.clone();
#     //cols
#     for (int i = 0; i < w; i++)
#     {
#         //rows
#         for (int j = 0; j < h; j++)
#         {
#             double delta;
#             if (i < (w / 2))
#                 delta = (double)i / (double)w * dst * sin(((double)j / (double)h) * pi);
#             else
#                 delta = (1 - (double)i / (double)w) * dst * sin(((double)j / (double)h) * pi);
#             mat.at(j, i) = t.at(j, i - delta);
#         }
#     }
#     return 0;

import os
if __name__=="__main__":
    data_root=r"F:\qianlinjun\temp\Data_Augmention"
    im_path=os.path.join(data_root, 'test.png')
    image=Image.open(im_path)
    image=np.array(image)
    

    # image1=NN_interpolation(image, image.shape[0]*2,image.shape[1]*2)
    # image1=Image.fromarray(image1.astype('uint8')).convert('RGB')
    # image1.save(os.path.join(data_root, 'NN.png'))

    # image2=BiLinear_interpolation(image,image.shape[0]*2,image.shape[1]*2)
    # image2=Image.fromarray(image2.astype('uint8')).convert('RGB')
    # image2.save(os.path.join(data_root, 'BiLinear.png'))

    # image3=BiCubic_interpolation(image,image.shape[0]*2,image.shape[1]*2)
    # image3=Image.fromarray(image3.astype('uint8')).convert('RGB')
    # image3.save(os.path.join(data_root, 'BiCubic.png'))

    image4=localwarp(image,image.shape[0],image.shape[1])
    image4=Image.fromarray(image4.astype('uint8')).convert('RGB')
    image4.save(os.path.join(data_root, 'warp.png'))