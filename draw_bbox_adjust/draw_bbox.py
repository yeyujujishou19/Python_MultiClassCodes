from PIL import Image, ImageDraw, ImageFont
from skimage import io, color, morphology, filters, draw, measure
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import cv2
import copy
import time

bbox_property_dict = {11: '未设置', 12: '根', 13: '标题', 14: '段落',
                      15: '图片', 151: '图名', 152: '图说',
                      16: '表格', 161: '表名', 162: '表头', 163: '表尾',
                      17: '公式', 18: '参考文献', 19: '引用'}


#创建文件夹
# mainPath  主路径 "D:/"
# bbox_property_dict  字典
def mkdir(mainPath,bbox_property_dict):
    for key in bbox_property_dict:
        dstPath = mainPath
        dstPath +=bbox_property_dict[key]
        isExists = os.path.exists(dstPath) # 判断路径是否存在
        if not isExists: # 如果不存在则创建目录
            os.makedirs(dstPath) # 创建目录操作函数
            print(dstPath + ' 创建成功')
        else:
            print(dstPath + ' 目录已存在') # 如果目录存在则不创建，并提示目录已存在

#剪切并保存roi图像
# 函数功能    截取并保存矩形框
# img_gray   灰度图
# bbox_      矩形框信息
# bbox_property=None    #矩形框信息
def cut_roi(img_gray,image_name, bbox_, bbox_property=None):
    mainPath = '//192.168.107.145/sjck/0411/image_cut/'
    roi = img_gray[bbox_[1]:bbox_[1] + bbox_[3], bbox_[0]:bbox_[0] + bbox_[2]] #截取区域
    ss=bbox_property
    path = bbox_property.strip()      # 去除首位空格
    path = bbox_property.rstrip("'")  # 去除尾部 ' 符号
    path = bbox_property.split(',')   # 按，号分开
    savepath0 =''
    dirpath=''
    for i in range(len(path)):
        s1=path[i].split(':')
        if(i==0):
            dirpath=s1[1]   #文件夹名称
        if(i!=(len(path)-1)):
            savepath0 += s1[0]+s1[1]+'_'
        else:
            savepath0 += s1[0]+s1[1]
    jugSavePath = mainPath + dirpath
    isExists = os.path.exists(jugSavePath)  # 判断路径是否存在
    if not isExists:  # 如果不存在则创建目录
        return
    newimage_name=image_name[:-4] #去除后缀的图像名
    savepath=mainPath+dirpath+'/'+newimage_name+'_'+savepath0+'.jpg'
    io.imsave(savepath, roi)
    #time.sleep(0.5)  # 暂停 1 秒

def draw_bbox_(draw_, bbox_, bbox_property_=None,flag=0):
    '''
    Draw a box.

    Args:
        bbox_     --- Int array: Box information of text blocks.
        box_property_     --- string: The basic information of the block
        draw      --- Image drawing handle
    Returns:
        bbox     --- string: extra bbox.
    '''
    # Define font and color.
    font = ImageFont.truetype('C:\\Windows\\Fonts\\SIMYOU.TTF', 20)
    colors = (0, 0, 255)
    if flag==1:
        colors=(255,0,0)

    # draw bbox
    draw_.line([bbox_[0], bbox_[1], bbox_[0] + bbox_[2], bbox_[1]], fill=colors, width=5) #上横线
    draw_.line([bbox_[0] + bbox_[2], bbox_[1], bbox_[0] + bbox_[2], bbox_[1] + bbox_[3]], fill=colors,  #右竖线
               width=5)
    draw_.line([bbox_[0] + bbox_[2], bbox_[1] + bbox_[3], bbox_[0], bbox_[1] + bbox_[3]], fill=colors, #下横线
               width=5)
    draw_.line([bbox_[0], bbox_[1] + bbox_[3], bbox_[0], bbox_[1]], fill=colors, width=5)  #左竖线
    if bbox_property_:
        draw_.text([bbox_[0], bbox_[1] - 20], bbox_property_, font=font, fill=colors)


def find_bbox(img,bbox_modif,bbox_property):
    '''
    Looking for extra boxes .

    Args:
        img        --- Size m*n float array: gray scale src image.
        bbox_modif ---Int array: Box information of text blocks.
    Returns:
        bbox       --- Int array: Box information of text blocks.
    '''
    # # img = filters.median(img, morphology.square(3))
    # #
    # # img = morphology.erosion(img, morphology.square(3))
    # img = morphology.dilation(img, morphology.rectangle(3, 21))
    # # io.imshow(img)
    # # io.show()
    #
    # temp_bbox=[]
    # label_image=measure.label(img,connectivity=1)
    # regions=measure.regionprops(label_image)
    #
    # for region in regions:
    #     temp=region.bbox
    #     temp_bbox.append([temp[1],temp[0],temp[3],temp[2],1])
    #
    #
    # for i in range(len(temp_bbox)):
    #     if temp_bbox[i][4]==0:
    #         continue
    #     for j in range(i+1,len(temp_bbox)):
    #         if temp_bbox[j][4] == 0:
    #             continue
    #         if abs(temp_bbox[i][1]-temp_bbox[j][1])<=30:
    #             mid_1 = temp_bbox[i][0] + (temp_bbox[i][2]-temp_bbox[i][0])/2
    #             mid_2 = temp_bbox[j][0] + (temp_bbox[j][2]-temp_bbox[j][0])/2
    #             width_1 = (temp_bbox[i][2]-temp_bbox[i][0])/2
    #             width_2 = (temp_bbox[j][2]-temp_bbox[j][0])/2
    #             #if closer
    #             if abs(mid_1 - mid_2)  - (width_1 + width_2) < 50:
    #                 x_0=min(temp_bbox[i][0],temp_bbox[j][0])
    #                 y_0=min(temp_bbox[i][1],temp_bbox[j][1])
    #                 x_1=max(temp_bbox[i][2],temp_bbox[j][2])
    #                 y_1=max(temp_bbox[i][3],temp_bbox[j][3])
    #
    #                 #update bbox
    #                 temp_bbox[i][0]=x_0
    #                 temp_bbox[i][1]=y_0
    #                 temp_bbox[i][2]=x_1
    #                 temp_bbox[i][3]=y_1
    #                 temp_bbox[j][4]=0#Remove the comparable box
    #
    # temp_bbox_1=[]
    # for m in range(len(temp_bbox)):
    #     if temp_bbox[m][4]==1:
    #         temp_bbox_1.append(temp_bbox[m])
    #
    # bbox=[]
    # for n in range(len(temp_bbox_1)):
    #     if temp_bbox_1[n][2]-temp_bbox_1[n][0]<10 or temp_bbox_1[n][3] - temp_bbox_1[n][1]<10:
    #         continue
    #
    #     bbox.append([temp_bbox_1[n][0],temp_bbox_1[n][1],temp_bbox_1[n][2]-temp_bbox_1[n][0],
    #                         temp_bbox_1[n][3] - temp_bbox_1[n][1],1])


    # temp=[]
    # for i in range(len(bbox)):
    #     if bbox[i][4]==0:
    #         continue
    #     # if (bbox[i][1]<(img.shape[0])*2/5 or bbox[i][1]>(img.shape[0]*5)/6) and bbox[i][2]/bbox[i][3]>100 and bbox[i][3]<40 :
    #     #     bbox[i][4]=0
    #     #     temp.append(i)
    #     if bbox[i][1]<(img.shape[0])/6  and bbox[i][2]/bbox[i][3]>30 and bbox[i][3]<50 :
    #         bbox[i][4]=0
    #         temp.append(i)
    #     if  bbox[i][2] / bbox[i][3] > 40 and bbox[i][3]<10:
    #         bbox[i][4] = 0
    #         temp.append(i)
    #
    #     # if bbox[i][1]<(img.shape[0])/5 and bbox[i][3] / bbox[i][2]>10:
    #     #     temp.append(i)
    #     # if bbox[i][3] / bbox[i][2] > 10:
    #     #     bbox[i][4] = 0
    #     #     temp.append(i)
    #
    #
    #     for j in range(i+1,len(bbox)):
    #         if bbox[j][4]==0:
    #             continue
    #         overlap, flag = compute_IoU(bbox[i], bbox[j])
    #         if overlap>=0.9:
    #             if flag==1:
    #                 bbox[i][4]=0
    #                 temp.append(i)
    #             else:
    #                 bbox[j][4]=0
    #                 temp.append(j)
    #
    #         elif overlap>0 and overlap<0.9:
    #             x_0=min(bbox[i][0],bbox[j][0])
    #             y_0=min(bbox[i][1],bbox[j][1])
    #             x_1=max(bbox[i][0]+bbox[i][2],bbox[j][0]+bbox[j][2])
    #             y_1=max(bbox[i][1]+bbox[i][3],bbox[j][1]+bbox[j][3])
    #             bbox[i][0]=x_0
    #             bbox[i][1]=y_0
    #             bbox[i][2]=x_1-x_0
    #             bbox[i][3]=y_1-y_0
    #             bbox[j][4]=0
    #             temp.append(j)
    #         else:
    #             hor_x=abs(bbox[i][1] - bbox[j][1])
    #             ver_y=abs(bbox[i][0] - bbox[j][0])
    #             hor_distance=abs(bbox[i][0]+bbox[i][2]/2-(bbox[j][0]+bbox[j][2]/2))
    #             ver_distance=abs(bbox[i][1]+bbox[i][3]/2-(bbox[j][1]+bbox[j][3]/2))
    #             if (hor_x < 10 and hor_distance-(bbox[i][2]/2+bbox[j][2]/2)<10) or \
    #                     (ver_y<10 and ver_distance-(bbox[i][3]/2+bbox[j][3]/2)<10):
    #                 x_0 = min(bbox[i][0], bbox[j][0])
    #                 y_0 = min(bbox[i][1], bbox[j][1])
    #                 x_1 = max(bbox[i][0] + bbox[i][2], bbox[j][0] + bbox[j][2])
    #                 y_1 = max(bbox[i][1] + bbox[i][3], bbox[j][1] + bbox[j][3])
    #                 bbox[i][0] = x_0
    #                 bbox[i][1] = y_0
    #                 bbox[i][2] = x_1 - x_0
    #                 bbox[i][3] = y_1 - y_0
    #                 bbox[j][4]=0
    #                 temp.append(j)
    #
    #
    #     for k in range(len(bbox_modif)):
    #         for bbox_modif_ in bbox_modif[k]:
    #             overlap, flag = compute_IoU(bbox[i], bbox_modif_)
    #             if overlap > 0.3 and flag==2:
    #                 temp.append(i)
    #
    #             # if bbox_modif_[4] == 13 or bbox_property[k].split(',')[0][4:] == '篇名':
    #             #     center_x_0 = max(bbox[i][0],bbox_modif_[0])
    #             #     center_y_0 = max(bbox[i][1],bbox_modif_[1])
    #             #     center_x_1 = min(bbox[i][0]+bbox[i][2],bbox_modif_[0] + bbox_modif_[2])
    #             #     center_y_1 = min(bbox[i][1]+bbox[i][3],bbox_modif_[1] + bbox_modif_[3])
    #             #
    #             #     if abs(center_x_0 - center_x_1)<10 or abs(center_y_0 - center_y_1)<10:
    #             #         temp.append(i)
    #
    # for l in range(len(bbox)):
    #     for m in range(l + 1, len(bbox)):
    #         # print(overlap)
    #         overlap, flag = compute_IoU(bbox[l], bbox[m])
    #         if overlap >= 0.9:
    #             temp.append(l if flag == 2 else m)
    # bbox_ = []
    # for i in range(len(bbox)):
    #     if i in list(set(temp)):
    #         continue
    #     bbox[i][4]=1
    #     bbox_.append(bbox[i])

    img = filters.median(img, morphology.square(3))

    img = morphology.erosion(img, morphology.square(3))
    img = morphology.dilation(img, morphology.rectangle(3, 17))
    # io.imshow(img)
    # io.show()

    temp_bbox=[]
    label_image=measure.label(img,connectivity=1)
    regions=measure.regionprops(label_image)
    for region in regions:
        temp=region.bbox
        temp_bbox.append([temp[1],temp[0],temp[3],temp[2],1])


    for i in range(len(temp_bbox)):
        if temp_bbox[i][4]==0:
            continue
        for j in range(i+1,len(temp_bbox)):
            if temp_bbox[j][4] == 0:
                continue
            if abs(temp_bbox[i][1]-temp_bbox[j][1])<=30:
                mid_1 = temp_bbox[i][0] + (temp_bbox[i][2]-temp_bbox[i][0])/2
                mid_2 = temp_bbox[j][0] + (temp_bbox[j][2]-temp_bbox[j][0])/2
                width_1 = (temp_bbox[i][2]-temp_bbox[i][0])/2
                width_2 = (temp_bbox[j][2]-temp_bbox[j][0])/2
                #if closer
                if abs(mid_1 - mid_2)  - (width_1 + width_2) < 50:
                    x_0=min(temp_bbox[i][0],temp_bbox[j][0])
                    y_0=min(temp_bbox[i][1],temp_bbox[j][1])
                    x_1=max(temp_bbox[i][2],temp_bbox[j][2])
                    y_1=max(temp_bbox[i][3],temp_bbox[j][3])

                    #update bbox
                    temp_bbox[i][0]=x_0
                    temp_bbox[i][1]=y_0
                    temp_bbox[i][2]=x_1
                    temp_bbox[i][3]=y_1
                    temp_bbox[j][4]=0#Remove the comparable box

    temp_bbox_1=[]
    for m in range(len(temp_bbox)):
        if temp_bbox[m][4]==1:
            temp_bbox_1.append(temp_bbox[m])

    bbox=[]
    for n in range(len(temp_bbox_1)):
        if temp_bbox_1[n][2]-temp_bbox_1[n][0]<10 or temp_bbox_1[n][3] - temp_bbox_1[n][1]<10:
            continue
        bbox.append([temp_bbox_1[n][0],temp_bbox_1[n][1],temp_bbox_1[n][2]-temp_bbox_1[n][0],
                            temp_bbox_1[n][3] - temp_bbox_1[n][1],1])

    temp=[]
    for i in range(len(bbox)):
        if bbox[i][4]==0:
            continue
        # if (bbox[i][1]<(img.shape[0]*2)/5 or bbox[i][1]>(img.shape[0]*4)/5) and bbox[i][2]/bbox[i][3]>40 :
        #     temp.append(i)

        # if  bbox[i][2] / bbox[i][3] > 40:
        #     bbox[i][4]=0
        #     temp.append(i)
        # if bbox[i][1]>(img.shape[0])/2 and bbox[i][3] / bbox[i][2]>10:
        #     bbox[i][4]=0
        #     temp.append(i)
        if bbox[i][3] / bbox[i][2] > 10:
            bbox[i][4] = 0
            temp.append(i)

        for j in range(i+1,len(bbox)):
            if bbox[j][4]==0:
                continue
            overlap, flag = compute_IoU(bbox[i], bbox[j])
            if overlap>0.9:
                if flag==1 and (bbox[i][2]>1300 or bbox[i][3]>1000):
                    bbox[i][4]=0
                    temp.append(i)
                if flag==2 and (bbox[j][2]>1300 or bbox[j][3]>1000):
                    bbox[j][4]=0
                    temp.append(j)
            # if overlap>0.9:
            #     if flag==1 :
            #         bbox[i][4]=0
            #         temp.append(i)
            #     if flag==2 :
            #         bbox[j][4]=0
            #         temp.append(j)

        for k in range(len(bbox_modif)):
            for bbox_modif_ in bbox_modif[k]:
                overlap, flag1 = compute_IoU(bbox[i], bbox_modif_)
                if overlap > 0.9 and flag1==1:
                    bbox[i][4]==0
                    temp.append(i)
                # if bbox_modif_[4] == 13 or bbox_property[k].split(',')[0][4:] == '篇名':
                #     center_x_0 = max(bbox[i][0],bbox_modif_[0])
                #     center_y_0 = max(bbox[i][1],bbox_modif_[1])
                #     center_x_1 = min(bbox[i][0]+bbox[i][2],bbox_modif_[0] + bbox_modif_[2])
                #     center_y_1 = min(bbox[i][1]+bbox[i][3],bbox_modif_[1] + bbox_modif_[3])
                #
                #     if abs(center_x_0 - center_x_1)<50 or abs(center_y_0 - center_y_1)<50:
                #         temp.append(i)


    bbox_ = []
    for i in range(len(bbox)):
        if i in list(set(temp)):
            continue
        bbox[i][4]=1
        bbox_.append(bbox[i])
    return bbox_

def generate_new_label(bbox_property, label):
    '''
    Generate new labels .

    Args:
        bbox_property      --- string: Old label.
        label              --- string: Box information of text blocks.
    Returns:
        bbox_property       --- string: New label.
    '''
    list_property = bbox_property.split(',')
    bbox_property = list_property[0][:4] + label
    for list in list_property[1:]:
        bbox_property += ',' + list

    return bbox_property


def compute_IoU(bbox_1, bbox_2):
    '''
    Calculate the IoU between two blocks .

    Args:
        bbox_1             --- Int array: Box information of text blocks.
        bbox_2             --- Int array: Box information of text blocks.
    Returns:
        Iou                --- Float: The IoU of two blocks.
        flag               --- int  : The mark of the maximum area.
    '''
    # Calculate the coordinate information and area of bbox_1
    bbox_left_up_x = bbox_1[0]
    bbox_left_up_y = bbox_1[1]
    bbox_right_down_x = bbox_1[0] + bbox_1[2]
    bbox_right_down_y = bbox_1[1] + bbox_1[3]
    bbox_area = bbox_1[2] * bbox_1[3]

    # Calculate the coordinate information and area of bbox_2
    bboxP_left_up_x = bbox_2[0]
    bboxP_left_up_y = bbox_2[1]
    bboxP_right_down_x = bbox_2[0] + bbox_2[2]
    bboxP_right_down_y = bbox_2[1] + bbox_2[3]
    bboxP_area = bbox_2[2] * bbox_2[3]

    # Calculating the area of the overlapped
    overlap_width = min(bbox_right_down_x, bboxP_right_down_x) - max(bbox_left_up_x, bboxP_left_up_x)
    overlap_height = min(bbox_right_down_y, bboxP_right_down_y) - max(bbox_left_up_y, bboxP_left_up_y)
    overlap_area = max(overlap_width, 0) * max(overlap_height, 0)

    # Calculating ratio of the overlap to the minimum area block
    IoU = float(overlap_area) / (min(bbox_area, bboxP_area)+1)

    # Get the mark of the maximum area
    flag = 0
    if bbox_area == max(bbox_area, bboxP_area):
        flag = 1
    else:
        flag = 2

    return IoU, flag


def bbox_overlap(bbox_property, bbox, bbox_Paragraph, overlap_thresh=0.4):
    '''
    Remove the duplicate bounding boxes by overlap ratio.

    Args:
        bbox                 --- Int array: Box information of text blocks.
        bbox_Paragraph       --- Int array: Detailed information on paragraph blocks.
    Returns:
        bbox, bbox_Paragraph --- Int array: New bbox after remove overlap .
    '''

    label = []
    # remove overlap between block and block
    for i in range(len(bbox)):
        if bbox_Paragraph[i][0] != ['']:
            continue
        if bbox[i][2] == 0 or bbox[i][3] == 0:
            label.append(i)
            continue
        for j in range(i + 1, len(bbox)):
            if bbox_Paragraph[j][0] != ['']:
                continue
            overlap_1, flag = compute_IoU(bbox[i], bbox[j])
            if overlap_1 > overlap_thresh:
                label.append(i if flag == 2 else j)

    # remove overlap between patch and block
    for i in range(len(bbox)):
        if bbox_Paragraph[i][0] != ['']:
            continue
        if bbox[i][2]== 0 or bbox[i][3] == 0:
            label.append(i)
            continue
        for j in range(len(bbox_Paragraph)):
            bbox_Paragraph_ = bbox_Paragraph[j]
            if bbox_Paragraph_[0] == ['']:
                continue
            for k in range(len(bbox_Paragraph_)):
                overlap_2, _ = compute_IoU(bbox[i], bbox_Paragraph_[k])

                if overlap_2 > overlap_thresh:
                    label.append(i)
    temp_bbox = []
    temp_bboxPar = []
    temp_bboxpro = []

    # remove bbox corresponding to the label
    for i in range(len(bbox)):
        if i in list(set(label)):
            continue
        temp_bbox.append(bbox[i])
        temp_bboxPar.append(bbox_Paragraph[i])
        temp_bboxpro.append(bbox_property[i])
    return temp_bboxpro, temp_bbox, temp_bboxPar


def modify_boundary(img_gray, bbox_property, bbox):
    '''
    Modify the boundary by removing the white edges.

    Args:
        image_gray        --- Size m*n float array: gray scale src image.
        bbox              --- Int array: Box information of text blocks.
    Returns:
        bbox              --- Int array: New bbox after modify the boundary .
    '''
    # Get the ROI  and median filter and binaryzation
    property = bbox_property.split(',')[0][4:]

    roi = img_gray[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    # if bbox is image or form or formula
    if bbox[4] == 15 or bbox[4] == 16 or bbox[4] == 9 or bbox[4] == 17:
        return bbox

    roi = np.float64(roi < 0.8)

    if roi.shape[0]==0 or roi.shape[1]==0:
        return bbox

    if property == '注释':
        bbox_1 = []
        roi = filters.median(roi, morphology.square(3))
        roi = morphology.erosion(roi, morphology.rectangle(3, 3))
        roi = morphology.dilation(roi, morphology.rectangle(31, 51))
        # io.imshow(roi)
        # io.show()
        label_img = measure.label(roi, connectivity=1)
        for region in measure.regionprops(label_img):
            if region.area < 300:
                continue
            minr, minc, maxr, maxc = region.bbox
            if minr == 0 and minc == 0:
                continue
            bbox_ = [minc, minr, maxc, maxr]
            bbox_1.append(bbox_)
        bbox_2 = np.array(bbox_1).reshape(-1, 4)
        if len(bbox_2)==0:
            return bbox
        col_max = np.max(bbox_2, axis=0)
        col_min = np.min(bbox_2, axis=0)
        # io.imshow(roi[col_min[1]:col_max[3],col_min[0]:col_max[2]])
        # io.show()

        bbox[0] = bbox[0] + col_min[0]
        bbox[1] = bbox[1] + col_min[1]
        bbox[2] = col_max[2] - col_min[0]
        bbox[3] = col_max[3] - col_min[1]
    else:
        roi = filters.median(roi, morphology.square(3))
        roi = morphology.erosion(roi, morphology.square(3))
        roi = morphology.dilation(roi, morphology.rectangle(5, 17))
        # io.imshow(roi)
        # io.show()

        # small number of pixels after median filtering
        # don't deal with and return
        if np.sum(roi) < roi.shape[0] * roi.shape[1] * 0.01:
            return bbox

        # Calculate new edges.
        sum_ver = np.concatenate((np.zeros(1), np.sum(roi, 0), np.zeros(1)), axis=0)
        sum_hor = np.concatenate((np.zeros(1), np.sum(roi, 1), np.zeros(1)), axis=0)

        diff_ver = np.diff(sum_ver)
        diff_hor = np.diff(sum_hor)

        x_0 = np.where(diff_ver > 0)[0][0]
        x_1 = np.where(diff_ver < 0)[0][-1]
        y_0 = np.where(diff_hor > 0)[0][0]
        y_1 = np.where(diff_hor < 0)[0][-1]

        # If the change is too large, return
        if (x_1 - x_0) <= 0.2 * bbox[2] or (y_1 - y_0) <= 0.2 * bbox[3]:
            return bbox

        # Update bboxs
        bbox[2] = x_1 - x_0
        bbox[3] = y_1 - y_0
        bbox[0] = bbox[0] + x_0
        bbox[1] = bbox[1] + y_0

    return bbox


def draw_bbox(src_root, root, image_name, bbox_property, bbox, bbox_Paragraph):
    '''
    Draw the box on the image and save the picture to the folder.

    Args:
        image_name        --- string: image name.
        bbox_property     --- string: The basic information of the block
        bbox              --- Int array: Box information of text blocks.
        bbox_Paragraph    --- Int array: Detailed information on paragraph blocks.
    '''
    # print(image_name)
    # open image
    img = io.imread(src_root + image_name)

    img_gray = color.rgb2gray(img)

    img_extra = img_gray.copy()
    img_extra = np.float64(img_extra < 0.8)

    # PIL ImageDraw handle.
    img_obj = Image.fromarray(img)
    draw = ImageDraw.Draw(img_obj)

    # Define font and color.
    font = ImageFont.truetype('C:\\Windows\\Fonts\\SIMYOU.TTF', 20)
    colors = (0, 0, 255)

    # remove overlap bbox
    bbox_property, bbox, bbox_Paragraph = bbox_overlap(bbox_property, bbox, bbox_Paragraph)

    bbox_modif = []
    bbox_new_find = []

    for i in range(len(bbox)):
        bbox_temp = []

        bbox_property_ = bbox_property[i]

        # print(bbox_property_)
        bbox_Paragraph_ = bbox_Paragraph[i]
        # if no information in a paragraph block, use bbox draw block
        if bbox_Paragraph_[0] == ['']:
            bbox_ = bbox[i]
            img_extra[bbox_[1]:bbox_[1] + bbox_[3], bbox_[0]:bbox_[0] + bbox_[2]] = 0.0

            # remove white edges to modify bound .
            bbox_ = modify_boundary(img_gray, bbox_property_, bbox_)
            # draw bbox
            # draw_bbox_(draw, bbox_, bbox_property_,0)
            cut_roi(img_gray,image_name, bbox_, bbox_property_)  ############

            bbox_temp.append(bbox_)
        else:
            for bb in range(len(bbox_Paragraph_)):
                bbox_property_ = generate_new_label(bbox_property_, bbox_property_dict[bbox_Paragraph_[bb][4]])
                bbox_ = [bbox_Paragraph_[bb][0], bbox_Paragraph_[bb][1], bbox_Paragraph_[bb][2], bbox_Paragraph_[bb][3],
                         bbox_Paragraph_[bb][4]]
                img_extra[bbox_[1]:bbox_[1] + bbox_[3], bbox_[0]:bbox_[0] + bbox_[2]] = 0.0

                bbox_ = modify_boundary(img_gray, bbox_property_, bbox_)
                # draw_bbox_(draw, bbox_, bbox_property_,0)
                cut_roi(img_gray,image_name, bbox_, bbox_property_)  ############
                bbox_temp.append(bbox_)
        bbox_modif.append(bbox_temp)
    bbox_new=[]
    # bbox_new = find_bbox(img_extra,bbox_modif,bbox_property)
    # if bbox_new:
    #     for bbox_new_ in bbox_new:
    #         draw_bbox_(draw, bbox_new_, bbox_property_='块属性：新增方框，块类型：横栏',flag=1)
    #         bbox_property_ = '块属性：新增方框，块类型：横栏'
    #         cut_roi(img_gray, bbox_new_, bbox_property_)  ############

    # #save the results of drawing
    # if not os.path.exists(root+'/save'):
    #     os.makedirs(root+'/save')

    ##img_obj.save(root + image_name)#####存储标记框图像

    return bbox_modif, bbox_Paragraph, bbox_property, bbox_new

'''
root      存储主路径
txt_save  存储txt路径
src_root  #读取主路径
txt       读取txt路径
img_dir   #图像名称
'''
def process(root, txt_save, src_root, txt, img_dir):
    # open new txt and save new label
    # fp_w = open(txt_save, 'a')
    # fp_w.write('{\n')
    # fp_w.write('\t"anno":\n')
    # fp_w.write('\t[\n')
    # open txt and read any lines to list
    fp_r = open(txt, 'r')
    lines = fp_r.readlines()

    label = 0  # savs current lines
    location = []  # save left parenthesis position
    bbox_imformations = []

    for line in lines:
        # If the left ,put into the stack
        if '{' in line:
            location.append(label)

        # If the right ,the lrft out of the stack,
        # Store the strings in parentheses into a list
        if '}' in line:
            if label + 1 == len(lines):
                break
            bbox_imformations.append(lines[location[-1] + 1:label])
            location.pop()
        label += 1

    x=0
    for id in tqdm.tqdm(bbox_imformations, desc='Processing ' + img_dir):
        x+=1
        id = [i for i in id if i is not '\n']

        # get image name
        image_name = id[0].strip().strip('\n').strip('，').split(':')[-1].strip(' ').strip('”').strip('“')

        # store bbox_property bbox bbox_Paragraph
        bbox_property = []
        bbox = []
        bbox_Paragraph = []

        # decode bbox_property bbox bbox_Paragraph
        for index in range(1, len(id), 3):
            # decode bbox imformations
            bbox_property_ = id[index].strip()[17:-1].strip('“').strip('.').strip(' ').strip('”').strip('“')
            bbox_ = id[index + 1].strip().strip('\n').split(':')[-1].strip(' ').strip('”').strip('“').strip('.').split(
                ',')
            bbox_Paragraph_ = id[index + 2].strip().strip('\n').split(':')[-1].strip(' ').strip('”').strip('“').strip(
                '.').split('.')
            bbox_Paragraph_ = [i.split(',') for i in bbox_Paragraph_]

            # Convert the string to int
            bbox_ = [int(i) for i in bbox_]

            for bp in bbox_Paragraph_:
                for i in range(len(bp)):
                    if bp[i] == '':
                        continue
                    bp[i] = int(bp[i])

            bbox_property.append(bbox_property_)
            bbox.append(bbox_)
            bbox_Paragraph.append(bbox_Paragraph_)
        # draw bbox and save image
        bbox_modif, bbox_Paragraph, bbox_property, bbox_new = draw_bbox(src_root, root, image_name, bbox_property, bbox,
                                                                        bbox_Paragraph)
        # if image_name.split('.')[0][-2:]=='e1' or x==len(bbox_imformations):
        #     continue
        # fp_w.write('\t\t{\n')
        #
        # fp_w.write('\t\t  "image_id":"' + image_name + '",\n')
        #
        #
        # if bbox != []:
        #     for i in range(len(bbox_property)):
        #         fp_w.write('\t\t\t"bbox_property":"' + bbox_property[i] + '."\n')
        #
        #         fp_w.write('\t\t\t"bbox":"')
        #         string_1 = ''
        #         for temp in bbox_modif[i]:
        #             string_1 += str(temp[0]) + ',' + str(temp[1]) + ',' + str(temp[2]) + ',' + str(temp[3]) + ',' + str(
        #                 temp[4]) + '.'
        #         fp_w.write(string_1 + '"\n')
        #
        #         fp_w.write('\t\t\t"bbox_Paragraph":"')
        #         string_2 = ''
        #         for temp in bbox_Paragraph[i]:
        #             if temp == ['']:
        #                 continue
        #             string_2 += str(temp[0]) + ',' + str(temp[1]) + ',' + str(temp[2]) + ',' + str(temp[3]) + ',' + str(
        #                 temp[4]) + '.'
        #         fp_w.write(string_2 + '"\n')
        #
        # if bbox_new == [] and bbox == []:
        #     fp_w.write('\n')
        #     continue
        # elif bbox != [] and bbox_new == []:
        #     fp_w.write('\t\t\t"bbox_new":""\n')
        # else:
        #     string_3 = ''
        #     fp_w.write('\t\t\t"bbox_new":"')
        #     for temp in bbox_new:
        #         string_3 += str(temp[0]) + ',' + str(temp[1]) + ',' + str(temp[2]) + ',' + str(temp[3]) + ',' + str(temp[4]) + '.'
        #     fp_w.write(string_3 + '"\n')
        # fp_w.write('\t\t},\n')

    # fp_w.write('\t]\n')
    # fp_w.write('}\n')
    #
    fp_r.close()
    # fp_w.close()


def batch_processing():
    dstpath='//192.168.107.145/sjck/0411/image_cut/'
    if not os.path.exists(dstpath):  #创建存储路径
        os.mkdir(dstpath)

    mainPath=dstpath
    mkdir(mainPath, bbox_property_dict)   #创建分块目

    srcpath='//192.168.107.145/sjck/0411/image/'
    id=0

    # #首次存储
    # fileObject = open('ImageDirList.txt', 'w')  # 存储
    # for img_dir0 in os.listdir(srcpath):
    #     fileObject.write(img_dir0)
    #     fileObject.write('\n')
    # fileObject.close()

    #按行读取txt内容
    f = open('ImageDirList.txt', "r")
    lines = f.readlines()  # 读取全部内容
    clone_lines = list(lines)  # 新旧列表内存独立
    for img_dir in lines:
    #for img_dir in os.listdir(srcpath):
        id+=1
        img_dir=img_dir.replace("\n", "") #去除换行符
        src_root = srcpath + img_dir + '/'
        print('%d/%d当前执行目录：%s'%(id,len(lines),src_root))
        root = dstpath
        if not os.path.exists(root):
            os.mkdir(root)

        txt = src_root + img_dir + '.txt'
        txt_save = root + img_dir + '.txt'

        process(root, txt_save, src_root, txt, img_dir)  #处理

        #删除处理过的行
        newclone_lines = list(clone_lines)  # 新旧列表内存独立
        for i in range(len(clone_lines)):
            str=clone_lines[i]
            str = str.replace("\n", "")  # 去除换行符
            if(str == img_dir):
                del newclone_lines[i]  #删除处理过的行
                break

        #存储经过删除行的图像路径
        fileObject = open('ImageDirList.txt', 'w')  # 存储
        for line in newclone_lines:
            fileObject.write(line)
        fileObject.close()
        clone_lines.clear()
        clone_lines=list(newclone_lines)  # 新旧列表内存独立
        newclone_lines.clear()


if __name__ == "__main__":

    # str_1='1703QB03090'
    # if not os.path.exists('./{}/{}'.format(str_1,str_1)): #创建存储路径
    #     os.mkdir('./{}/{}'.format(str_1,str_1))
    # process('./{}/{}/'.format(str_1,str_1),'./{}/{}/{}.txt'.format(str_1,str_1,str_1),'./{}/'.format(str_1),
    #         './{}/{}.txt'.format(str_1,str_1),'{}'.format(str_1))
    batch_processing()
