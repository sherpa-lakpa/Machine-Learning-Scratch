import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# or if you're on a Unix system simply do export TF_CPP_MIN_LOG_LEVEL=2.

# TF_CPP_MIN_LOG_LEVEL is a TensorFlow environment variable responsible for the logs, to silence INFO logs set it to 1, to filter out WARNING 2 and to additionally silence ERROR logs (not recommended) set it to 3

x1 =  tf.constant(5)
x2 =  tf.constant(5)

# result = x1 * x2
# result = tf.mul(x1,x2)
result = tf.multiply(x1, x2, name=None)
print(result)

# sess = tf.Session()
# print(sess.run(result))
# sess.close()

with tf.Session() as sess:
	output =  sess.run(result)
	print(output)