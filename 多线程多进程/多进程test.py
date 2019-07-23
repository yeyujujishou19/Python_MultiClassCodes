#coding=utf-8
import threading
import multiprocessing
import time
import os
import cv2
import re   #查找字符串   re.finditer(word, path)]
import numpy as np
from time import sleep, ctime

iDisPlay=1000      # 显示间隔
ithreadNum=8       # 进程数量
#遍历文件夹
list = []
def TraverFolders(rootDir):
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        list.append(path)
        if os.path.isdir(path):
            TraverFolders(path)
    return list

#可以读取带中文路径的图
def cv_imread(file_path,type=0):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    if(type==0):
        if(len(cv_img.shape)==3):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img

#叠加两张图片，输入皆是黑白图，img1是底层图片，img2是上层图片，返回叠加后的图片
def ImageOverlay(img1,img2):
    # 把logo放在左上角，所以我们只关心这一块区域
    h = img1.shape[0]
    w = img1.shape[1]
    rows = img2.shape[0]
    cols = img2.shape[1]
    roi = img1[int((h - rows) / 2):rows + int((h - rows) / 2), int((w - cols) / 2):cols + int((w - cols) / 2)]
    # 创建掩膜
    img2gray = img2.copy()
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    dst = cv2.add(img1_bg, img2)  # 进行融合
    img1[int((h - rows) / 2):rows + int((h - rows) / 2),int((w - cols) / 2):cols + int((w - cols) / 2)] = dst  # 融合后放在原图上
    return img1

#函数功能：处理白边
#找到上下左右的白边位置
#剪切掉白边
#二值化
#将图像放到64*64的白底图像中心
def HandWhiteEdges(img):
    ret, thresh1 = cv2.threshold(img, 249, 255, cv2.THRESH_BINARY)
    # OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 膨胀图像
    thresh1 = cv2.dilate(thresh1, kernel)
    row= img.shape[0]
    col = img.shape[1]
    tempr0 = 0    #横上
    tempr1 = 0    #横下
    tempc0 = 0    #竖左
    tempc1 = 0    #竖右
    # 765 是255+255+255,如果是黑色背景就是0+0+0，彩色的背景，将765替换成其他颜色的RGB之和，这个会有一点问题，因为三个和相同但颜色不一定同
    for r in range(0, row):
        if thresh1.sum(axis=1)[r] != 255 * col:
            tempr0 = r
            break

    for r in range(row - 1, 0, -1):
        if thresh1.sum(axis=1)[r] != 255 * col:
            tempr1 = r
            break

    for c in range(0, col):
        if thresh1.sum(axis=0)[c] != 255 * row:
            tempc0 = c
            break

    for c in range(col - 1, 0, -1):
        if thresh1.sum(axis=0)[c] != 255 * row:
            tempc1 = c
            break

    # 创建全白图片
    imageTemp = np.zeros((64, 64, 3), dtype=np.uint8)
    imageTemp = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2GRAY)
    imageTemp.fill(255)

    if(tempr1-tempr0==0 or tempc1-tempc0==0):   #空图
        return imageTemp    #返回全白图

    new_img = img[tempr0:tempr1, tempc0:tempc1]
    #二值化
    retval,binary = cv2.threshold(new_img,0,255,cv2.THRESH_OTSU)

    #叠加两幅图像
    rstImg=ImageOverlay(imageTemp, binary)
    return rstImg

#函数功能：简单网格
#函数要求：1.无关图像大小；2.输入图像默认为灰度图;3.参数只有输入图像
#返回数据：64*1维特征
def SimpleGridFeature(image):
    '''
    @description:提取字符图像的简单网格特征
    @image:灰度字符图像
    @return:长度为64字符图像的特征向量feature
    @author:RenHui
    '''
    new_img=HandWhiteEdges(image)  #白边处理
    #new_img=image
    #图像大小归一化
    image = cv2.resize(new_img,(64,64))
    img_h = image.shape[0]
    img_w = image.shape[1]

    #二值化
    retval,binary = cv2.threshold(image,0,255,cv2.THRESH_OTSU)

    #定义特征向量
    grid_size1 = 16
    grid_size2 = 8
    grid_size3 = 4
    feature = np.zeros(grid_size1*grid_size1+grid_size2*grid_size2+grid_size3*grid_size3)

    #计算网格大小1
    grid_h1 = binary.shape[0]/grid_size1
    grid_w1 = binary.shape[1]/grid_size1
    for j in range(grid_size1):
        for i in range(grid_size1):
            grid = binary[int(j*grid_h1):int((j+1)*grid_h1),int(i*grid_w1):int((i+1)*grid_w1)]
            feature[j*grid_size1+i] = grid[grid==0].size

    #计算网格大小2
    grid_h2 = binary.shape[0]/grid_size2
    grid_w2 = binary.shape[1]/grid_size2
    for j in range(grid_size2):
        for i in range(grid_size2):
            grid = binary[int(j*grid_h2):int((j+1)*grid_h2),int(i*grid_w2):int((i+1)*grid_w2)]
            feature[grid_size1*grid_size1+j*grid_size2+i] = grid[grid==0].size

    #计算网格大小3
    grid_h3 = binary.shape[0]/grid_size3
    grid_w3 = binary.shape[1]/grid_size3
    for j in range(grid_size3):
        for i in range(grid_size3):
            grid = binary[int(j*grid_h3):int((j+1)*grid_h3),int(i*grid_w3):int((i+1)*grid_w3)]
            feature[grid_size1*grid_size1+grid_size2*grid_size2+j*grid_size3+i] = grid[grid==0].size

    return feature

#存储图像
def saveImage(list,savePath,threadID):
    count = 0  # 文件计数
    countImg = 0  # 图片计数
    for filename in list[threadID::ithreadNum]:
        count += 1
        # -----确定子文件夹名称------------------
        word = r'\\'
        a = [m.start() for m in re.finditer(word, filename)]
        if (len(a) == 5):  # 字文件夹
            strtemp = filename[a[-1] + 1:]  # 文件夹名称-字符名称
            # print(filename)
        # -----确定子文件夹名称------------------

        # -----子文件夹图片特征提取--------------
        if (len(a) == 6):  # 子文件夹下图片
            if ('.jpg' in filename):
                countImg += 1
                if (countImg % iDisPlay == 0):
                    print("进程%d, 共%d个文件，正在处理第%d张图片..." % (threadID, len(list), countImg))
                image = cv_imread(filename, 0)
                feature=SimpleGridFeature(image)
                # cv2.imencode('.jpg', image)[1].tofile(savePath + r'\\' + str(threadID)+ '_'+str(countImg)+ '.jpg')
            else:
                continue


if __name__ == "__main__":
    # -------------------------------------------------------------------------------------------------------
    time_start = time.time()  # 开始计时

    path = r"D:\sxl\处理图片\汉字分类\train85"  # 文件夹路径
    print("遍历图像中，可能需要花一些时间...")
    # list = TraverFolders(path)

    if(os.path.exists(r"E:\sxl_Programs\Python\ANN\npy2\List_21477.npy")):
        list = np.load(r"E:\sxl_Programs\Python\ANN\npy2\List_21477.npy")
    else:
        path=r"E:\2万汉字分类\train21477"     #文件夹路径
        # path=r"D:\sxl\处理图片\汉字分类\train85"     #文件夹路径
        print("遍历图像中，可能需要花一些时间...")
        list=TraverFolders(path)
        np.save(r"E:\sxl_Programs\Python\ANN\npy2\List_21477.npy", list)

    savePath = r"D:\python多线程"
    # 创建进程
    Process=[]
    print("创建进程...")
    for i in range(ithreadNum):
        p = multiprocessing.Process(target = saveImage, args = (list, savePath, i))
        Process.append(p)

    # 启动进程
    print("启动进程...")
    for i in range(ithreadNum):
        Process[i].start()

    for i in range(ithreadNum):
        Process[i].join()

    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id" + str(p.pid))
    print ("END!!!!!!!!!!!!!!!!!")

    time_end=time.time()
    time_h=(time_end-time_start)/3600
    print('用时：%.6f 小时'% time_h)
