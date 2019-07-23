#coding=gbk
import os
import sys
def rename():
    myDic={}  #字典
    path="D:/2019.07.15印地语新增/"    #读取路径
    name=""  #新名称开头
    startNumber="1"   #新名称起始序号
    fileType=".pdf"   #文件后缀名
    print("正在生成以"+name+startNumber+fileType+"迭代的文件名")
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



    print("一共修改了"+str(count)+"个文件")
    print(myDic)

    #将字典写入txt
    write_file = 'D:/2019.07.15印地语新增.txt'

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


