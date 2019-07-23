#coding=gbk
import os
import sys


# f = open('D:/2019.07.15印地语新增.txt',"r")   #设置文件对象
# str = f.read()     #将txt文件的所有内容读入到字符串str中
# print(str)
# print(str[1])


#将txt中新旧名称对照表读取到字典中
myDic = {}    #字典
for line in open('D:/2019.07.15印地语新增.txt',"r"): #设置文件对象并读取每一行文件
    new_old_Name = line.split("_")
    myDic[str(new_old_Name[0])] = str(new_old_Name[1][:-1])
print(myDic)

#改变生成图片的文件夹名称
path="D:/2/"    #读取路径
list_dirs = os.listdir(path) #获取子文件夹列表
for sub_dir in list_dirs:
    Olddir = os.path.join(path, sub_dir)
    Newdir = os.path.join(path, myDic[str(sub_dir)])
    os.rename(Olddir, Newdir)
    a=0