# 引入模块
import os

#创建文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


###################################################
f = open("error.txt", "r")
lines = f.readlines()  # 读取全部内容
for line in lines:
    # print(line)
    #拆分字符串
    line_array = line.split('\t')
    first_col=line_array[0] #分割出待创建文件夹名称
    # print(first_col)
    # 定义要创建的目录
    mkpath=(r"//error_Images/%s" %(first_col))
    # 调用函数，创建文件夹
    mkdir(mkpath)
print(len(lines))