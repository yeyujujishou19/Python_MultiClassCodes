#coding=gbk
import os
import sys
def rename():
    myDic={}  #�ֵ�
    path="D:/2019.07.15ӡ��������/"    #��ȡ·��
    name=""  #�����ƿ�ͷ
    startNumber="1"   #��������ʼ���
    fileType=".pdf"   #�ļ���׺��
    print("����������"+name+startNumber+fileType+"�������ļ���")
    count=0
    filelist=os.listdir(path)
    for files in filelist:
        Olddir=os.path.join(path, files)
        if os.path.isdir(Olddir):
            continue
        oldname1=Olddir.split('/')
        oldname2=oldname1[-1][:-4]
        myDic[str(count+1)] = str(oldname2)

        Newdir=os.path.join(path,name+str(count+int(startNumber))+fileType)
        os.rename(Olddir,Newdir)
        count+=1



    print("һ���޸���"+str(count)+"���ļ�")
    print(myDic)

    #���ֵ�д��txt
    write_file = 'D:/2019.07.15ӡ��������.txt'

    print(write_file)
    output = open(write_file, 'w')
    for key, val in myDic.items():
        output.write(str(key))
        output.write('_')
        output.write(str(val))
        output.write('\n')
        # write_str = str(i) + '' + str(list(dict[i])) + '\n'
        # output.write(write_str)
    output.close()


rename()


