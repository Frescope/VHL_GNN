# 网络结构，用于模型的视觉特征提取
import tensorflow as tf

FRAME_NUM = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
FRAME_CHANN = 3

def conv3d(name, l_input, w, b):
    return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def visualnet(frame, weights, biases):

    # Convolution Layer
    conv1 = conv3d('conv1', frame, weights['wc1'], biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    # Convolution Layer
    conv2 = conv3d('conv2', pool1, weights['wc2'], biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = conv3d('conv3a', pool2, weights['wc3a'], biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, weights['wc3b'], biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # Convolution Layer
    conv4 = conv3d('conv4a', pool3, weights['wc4a'], biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, weights['wc4b'], biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # Convolution Layer
    conv5 = conv3d('conv5a', pool4, weights['wc5a'], biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    feat_out = conv5
    conv5 = conv3d('conv5b', conv5, weights['wc5b'], biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')
    pool5 = max_pool('pool5', conv5, k=2)

    pool6 = tf.transpose(pool5, perm=(0, 4, 2, 3, 1))
    pool6 = tf.nn.max_pool3d(pool6, ksize=[1, 4, 1, 1, 1], strides=[1, 4, 1, 1, 1], padding='SAME')
    pool6 = tf.transpose(pool6, perm=(0, 4, 2, 3, 1))
    out = pool6
    return out, feat_out
