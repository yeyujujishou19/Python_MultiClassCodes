#coding=utf-8
from time import sleep, ctime
import threading
import os
import cv2
import re   #查找字符串   re.finditer(word, path)]
import numpy as np

iDisPlay=1      #显示间隔
ithreadNum=5    # 线程数量
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
                    print("线程%d, 共%d个文件，正在处理第%d张图片..." % (threadID, len(list), countImg))
                image = cv_imread(filename, 0)
                cv2.imencode('.jpg', image)[1].tofile(savePath + r'\\' + str(threadID)+ '_'+str(countImg)+ '.jpg')
            else:
                continue


# -------------------------------------------------------------------------------------------------------
path = r"D:\sxl\处理图片\汉字分类\11"  # 文件夹路径
print("遍历图像中，可能需要花一些时间...")
list = TraverFolders(path)

savePath=r"D:\python多线程"
threads = []  #线程列表
#创建线程
print("创建线程...")

for i in range(ithreadNum):
    t = threading.Thread(target=saveImage, args=(list, savePath, i))
    threads.append(t)

#启动线程
print("启动线程...")
for i in range(ithreadNum):
    threads[i].start()


# list =['a','b','c','d','e','f','g']
# for a in list[2::2]:
#     print(a)

