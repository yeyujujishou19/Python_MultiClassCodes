#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from math import ceil        #向上取整
import time
import os
import random
import tensorflow as tf
from tensorflow.python.framework import graph_util

batch_size=5
inpyNUM=1
n_label = 6           #标签维度

x=np.load(r"E:\sxl_Programs\Python\ANN\npy2\Img21477_features_train_1.npy")
print(x.shape)