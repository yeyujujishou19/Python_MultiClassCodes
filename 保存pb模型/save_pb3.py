import tensorflow as  tf
import numpy as np
import os

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', 'model/', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

sess = tf.InteractiveSession()

x = tf.placeholder('float', shape=[None, 5], name="inputs")
y_ = tf.placeholder('float', shape=[None, 1])
w = tf.get_variable('w', shape=[5, 1], initializer=tf.truncated_normal_initializer)
b = tf.get_variable('b', shape=[1], initializer=tf.zeros_initializer)
sess.run(tf.global_variables_initializer())
y = tf.add(tf.matmul(x, w), b, name="outputs")
ms_loss = tf.reduce_mean((y - y_) ** 2)
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(ms_loss)
train_x = np.random.randn(1000, 5)
# let the model learn the equation of y = x1 * 1 + x2 * 2 + x3 * 3
train_y = np.sum(train_x * np.array([1, 2, 3, 4, 5]) + np.random.randn(1000, 5) / 100, axis=1).reshape(-1, 1)
for i in range(FLAGS.training_iteration):
    loss, _ = sess.run([ms_loss, train_step], feed_dict={x: train_x, y_: train_y})
    if i % 100 == 0:
        print("loss is:", loss)
        graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                             ["inputs", "outputs"])
        tf.train.write_graph(graph, ".", FLAGS.work_dir + "liner.pb",
                             as_text=False)
print('Done exporting!')
print('Done training!')