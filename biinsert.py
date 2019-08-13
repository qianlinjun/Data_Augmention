from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import math

def NN_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=round((i+1)*(scrH/dstH))
            scry=round((j+1)*(scrW/dstW))
            retimg[i,j]=img[scrx-1,scry-1]
    return retimg

def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
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
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
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
    

import os
if __name__=="__main__":
    data_root="/media/liesmars/b71625db-4194-470b-a8ab-2d4cf46f4cdd/data/dota/data_aug/"
    im_path=os.path.join(data_root, 'test.png')
    image=np.array(Image.open(im_path))

    image1=NN_interpolation(image,image.shape[0]*2,image.shape[1]*2)
    image1=Image.fromarray(image1.astype('uint8')).convert('RGB')
    image1.save(os.path.join(data_root, 'NN.png'))

    image2=BiLinear_interpolation(image,image.shape[0]*2,image.shape[1]*2)
    image2=Image.fromarray(image2.astype('uint8')).convert('RGB')
    image2.save(os.path.join(data_root, 'BiLinear.png'))

    image3=BiCubic_interpolation(image,image.shape[0]*2,image.shape[1]*2)
    image3=Image.fromarray(image3.astype('uint8')).convert('RGB')
    image3.save(os.path.join(data_root, 'BiCubic.png'))