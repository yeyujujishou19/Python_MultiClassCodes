#coding=gbk
import os
import sys


# f = open('D:/2019.07.15ӡ��������.txt',"r")   #�����ļ�����
# str = f.read()     #��txt�ļ����������ݶ��뵽�ַ���str��
# print(str)
# print(str[1])


#��txt���¾����ƶ��ձ��ȡ���ֵ���
myDic = {}    #�ֵ�
for line in open('D:/2019.07.15ӡ��������.txt',"r"): #�����ļ����󲢶�ȡÿһ���ļ�
    new_old_Name = line.split("_")
    myDic[str(new_old_Name[0])] = str(new_old_Name[1][:-1])
print(myDic)

#�ı�����ͼƬ���ļ�������
path="D:/2/"    #��ȡ·��
list_dirs = os.listdir(path) #��ȡ���ļ����б�
for sub_dir in list_dirs:
    Olddir = os.path.join(path, sub_dir)
    Newdir = os.path.join(path, myDic[str(sub_dir)])
    os.rename(Olddir, Newdir)
    a=0