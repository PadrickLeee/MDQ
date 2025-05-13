import sys
import cv2
import matplotlib.pyplot as plt
import re,os
from PIL import Image
import numpy as np
from tqdm import tqdm


data = []
for line in open("/home/ma-user/modelarts/work/jjw/SFT/lqf/1d-tokenizer/imagenet_convert/txt/file.txt", "r"):  
# 设置文件对象并读取每一行文件
    data.append(line)
print(data[2][3:31])#这个显示的是每个图像的名称
print(data[2][32:41])#这个显示的时该图像的类，我们也是用这个类创建文件夹的

#显示图像

#img = Image.open('D:/Dateset/Alldataset/Imagenet/ILSVRC2012_img_val/{}'.format(data[2][3:31]))
#plt.figure("dog")
#plt.imshow(img)
#plt.show()

for a in tqdm(data):
    a = a.strip()
    nameimage = a[3:31]#获取该图像的名称
    from_img_path = '/home/ma-user/modelarts/work/jjw/SFT/lqf/ILSVRC/Data/CLS-LOC/val/{}'.format(nameimage) # 原来的图片地址
    # print('from_img_path: ', from_img_path)
    im = Image.open(from_img_path)  
    #这个时获取原始验证集ILSVRC2012_img_val中每个图像，因为data中每一行存储一个图像的名称和标签
    path1 = '/home/ma-user/modelarts/work/jjw/SFT/lqf/ILSVRC/Data/CLS-LOC/val/{}'.format(a[32:-1])
    #ILSVRC2012_img_val1时新的验证集，ILSVRC2012_img_val1里面有很多以图像类命名的子文件夹
    im.save(os.path.join(path1, nameimage))#把图像保存在文件中
    im.close()
