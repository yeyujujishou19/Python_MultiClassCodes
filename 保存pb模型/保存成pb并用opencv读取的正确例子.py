# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:04:06 2018


@author: Administrator
"""

import pylab
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util

tf.reset_default_graph()
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print('输入数据：', mnist.train.images)
print('输入数据的shape：', mnist.train.images.shape)
x = tf.placeholder("float", [None, n_input], name='input')
y = tf.placeholder("float", [None, n_classes], name='labels')


def multilayer_perception(x, weights_t, biases_t):
    # 第一层隐藏层
    layer_1 = tf.add(tf.matmul(x, weights_t['h1']), biases_t['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # 第二层隐藏层
    layer_2 = tf.add(tf.matmul(layer_1, weights_t['h2']), biases_t['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # out_layer = tf.matmul(layer_2, weights_t['out']) + biases_t['out']
    out_layer = tf.add(tf.matmul(layer_2, weights_t['out']), biases_t['out'], name="output")
    return out_layer


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),

    'out': tf.Variable(tf.random_normal([n_classes]))
}

# weights_c = {
#        'h1' : tf.constant(tf.random_normal([n_input, n_hidden_1])),
#        'h2' : tf.constant(tf.random_normal([n_hidden_1, n_hidden_2])),
#        'out': tf.constant(tf.random_normal([n_hidden_2, n_classes]))
#        }
#
# biases_c = {
#        'b1' : tf.constant(tf.random_normal([n_hidden_1])),
#        'b2' : tf.constant(tf.random_normal([n_hidden_2])),
#        'out': tf.constant(tf.random_normal([n_classes]))
#        }


print("learn_param")

pred = multilayer_perception(x, weights, biases)

print("multilayer_perception")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver = tf.train.Saver()

savedir = "log1/"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("finished")
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile('expert-graph_t1.pb', mode='wb') as f:

        f.write(constant_graph.SerializeToString())