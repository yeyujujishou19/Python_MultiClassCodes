import tensorflow as tf
import  numpy as np
logdir = 'E:/sxl_Programs/Python/Test/model/'
output_graph_path = logdir+'liner.pb'
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def,name="")
    with tf.Session() as sess:
        input = sess.graph.get_tensor_by_name("inputs:0")
        input2 = sess.graph.get_tensor_by_name("inputs2:0")
        output = sess.graph.get_tensor_by_name("outputs:0")
        result = sess.run(output, feed_dict={input: np.reshape([1.0,1.0,1.0,1.0,1.0],[-1,5])})
        print(result)
