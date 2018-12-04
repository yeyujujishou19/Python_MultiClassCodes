# 引入模块
#!/usr/bin/env python
# coding: utf-8
import os, sys
from PIL import Image

f = open("error.txt", "r")
lines = f.readlines()  # 读取全部内容
i=0
for line in lines:
    i+=1
    line_array = line.split('\t')   #拆分字符串
    line_array2 = line.split('\\')  # 拆分字符串
    first_col=line_array[0]         #分割出待创建文件夹名称
    second_col = line_array[1]      #分割出待移动的图片路径
    second_col = second_col.replace('\\', '/')  #转换字符

    moveToPath=(r"error_Images/%s" %(first_col)) # 定义要移动的目标目录
    target_ImageName=line_array2[-2]+'_'+line_array2[-1] #目标图像名称
    moveToPathImage=os.path.join(moveToPath,target_ImageName)  #合并字符串
    moveFromPath=("%s" %(second_col)) # 定义要移动的图像路径
    moveFromPath = moveFromPath.strip() # 去除首位空格
    moveToPathImage = moveToPathImage.strip()  # 去除首位空格

    if(os.path.exists(moveFromPath)):  #文件存在
        # 复制图片到新文件夹
        # img = Image.open(str(moveFromPath))
        # img.save(moveToPathImage)
        os.remove(str(moveFromPath)) #删除图片
    else:
        print("不存在",moveFromPath)
    a=0
    if(i%100==0):
        print("当前序号%d"% i)
        # print("原路径：",moveFromPath)
        # print("目标路径：", moveToPathImage)

print(len(lines))