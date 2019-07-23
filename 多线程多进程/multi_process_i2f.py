'''
@description:多进程的图像特征处理
@author:RenHui
'''

import multiprocessing
import os, time, random
import numpy as np
import cv2
import os
import sys
#sys.path.append('Z:/work/MyLibrary/')
#import siannodel.cv
#import siannodel.ml
#import siannodel.mytime
import image2feature

image_dir = 'D:/renhui-share/data/char21000_image/'
data_type = 'test'
save_path = 'D:/renhui-share/data/char21000_data/'
data_name = 'char21000_SGM'

char_set = np.array(os.listdir(image_dir))
np.save(save_path+'char21000_chars.npy',char_set)
char_set_n = len(char_set)

read_process_n = 8
repate_n = 1
data_size = 500000

shuffled = False
wight_edge = True
flutter = False


# 写数据进程执行的代码:
def read_image_to_queue(queue):
    '''
    @description:顺序读取图片到队列
    @queue:先进先出队列
    @author:RenHui
    '''
    print('Process to write: %s' % os.getpid())
    for j,dirname in enumerate(char_set):
        label = np.where(char_set==dirname)[0][0]
        print(j,'read '+dirname+' dir...',siannodel.mytime.current_time() )
        for parent,_,filenames in os.walk(os.path.join(image_dir,dirname,data_type)):
            for filename in filenames:
                if(filename[-4:]!='.jpg'):
                    continue
                image = siannodel.cv.cv_imread(os.path.join(parent,filename),0)
                queue.put((image,label))
    
    for i in range(read_process_n):
        queue.put((None,-1))

    print('read image over!')
    return True
        
# 读数据进程执行的代码:
def extract_feature(queue,lock,count):
    '''
    @description:从队列中取出图片进行特征提取
    @queue:先进先出队列
     lock：锁，在计数时上锁，防止冲突
     count:计数
    @author:RenHui
    '''

    print('Process %s start reading...' % os.getpid())

    global data_n
    features = [] #存放提取到的特征
    labels = [] #存放标签
    flag = True #标志着进程是否结束
    while flag:
        image,label = queue.get()

        if len(features) > data_size or label == -1:

            array_features = np.array(features)
            array_labels = np.array(labels)
            if shuffled:
                array_features,array_labels = siannodel.ml.shuffled_data(array_features,array_labels)
            
            lock.acquire()

            count.value += 1
            str_features_name = data_name+'_'+data_type+str(count.value)+'_features.npy'
            str_labels_name = data_name+'_'+data_type+str(count.value)+'_labels.npy'

            lock.release()

            np.save(save_path+str_features_name,array_features)
            np.save(save_path+str_labels_name,array_labels)   
            print(os.getpid(),'save:',str_features_name)
            print(os.getpid(),'save:',str_labels_name)
            features.clear()
            labels.clear()

        if label == -1:
            break

        for i in range(repate_n):
            feature = image2feature.image2feature(image,wight_edge=wight_edge,flutter=flutter)
            features.append(feature)
            labels.append(label)
    
    print('Process %s is done!' % os.getpid())

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    image_queue = multiprocessing.Queue(maxsize=1000)
    lock = multiprocessing.Lock()
    count = multiprocessing.Value('i',0)


    write_sub_process = multiprocessing.Process(target=read_image_to_queue, args=(image_queue,))

    read_sub_processes = []
    for i in range(read_process_n):
        read_sub_processes.append(
            multiprocessing.Process(target=extract_feature, args=(image_queue,lock,count))
        )

    # 启动子进程pw，写入:
    write_sub_process.start()
    # 启动子进程pr，读取:
    for p in read_sub_processes:
        p.start()
    # 等待进程结束:
    write_sub_process.join()
    for p in read_sub_processes:
        p.join()
