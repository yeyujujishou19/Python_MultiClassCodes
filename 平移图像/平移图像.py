import cv2
import random
import numpy as np

#随机移动图像，黑白图
def randomMoveImage(img):
    img_h = img.shape[0]
    img_w = img.shape[1]
    # 0 上，1 下，2 左，3 右
    idirection=random.randrange(0, 4) #随机产生0,1,2,3
    #随机移动距离
    iPixsNum=random.randrange(10, 30) #随机产生1,2

    if (idirection == 0): #上
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, 0], [0, 1, -iPixsNum]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for h in range(iPixsNum):              # 从上到下
            for w in range(img_w):             # 从左到右
                dst[img_h-h-1, w] = 255

    if (idirection == 1): #下
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, 0], [0, 1, iPixsNum]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for h in range(iPixsNum):              # 从上到下
            for w in range(img_w):             # 从左到右
                dst[h, w] = 255

    if (idirection == 2): #左
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, -iPixsNum], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for w in range(iPixsNum):  # 从左到右
            for h in range(img_h):  # 从上到下
                dst[h, img_w - w - 1] = 255

    if (idirection == 3): #右
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        M = np.float32([[1, 0, iPixsNum], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (img_w, img_h))
        for w in range(iPixsNum):  # 从左到右
            for h in range(img_h):  # 从上到下
                dst[h, w] = 255
    return dst

img = cv2.imread(r'D:\1.jpg',0)
rows,cols = img.shape
# for i in range(10):
dst=randomMoveImage(img)
cv2.imshow("22",dst)
cv2.waitKey(0)