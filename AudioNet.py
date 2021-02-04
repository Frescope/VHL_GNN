# 网络结构，用于MOSI预训练模型和整体模型
# 输入Emotion Intensity特征与spectrogram，做co-attention，经过卷积层后输出
import tensorflow as tf

SPECTRO_NUM = 6
SPECTRO_HEIGHT = 384
SPECTRO_WIDTH = 400
IS_LENGTH = 384

def audionet(spectro, EIfeat, weights, biases):
    # input: spectro (batch,384,400), EIfeat(batch,384,1)
    # co-attention
    D = spectro  # b*384*400
    D_T = tf.transpose(D, perm=[0,2,1])  # b*400*384
    Q = EIfeat  # b*384*1
    L = tf.matmul(D_T, Q)  # b*400*1
    L_T = tf.transpose(L, perm=[0,2,1])  # b*1*400
    A_Q = tf.nn.softmax(L)  # b*400*1
    A_D = tf.nn.softmax(L_T)  # b*1*400
    C_Q = tf.matmul(D, A_Q)  # b*384*1
    C_D = tf.concat([Q, C_Q], axis=1)  # b*768*1
    C_D = tf.matmul(C_D, A_D)  # b*768*400
    A = tf.concat([D, C_D], axis=1)  # b*1152*400
    spectro_en = tf.expand_dims(A,3)  # emotion-enhanced spectrogram: b*1152*400*1

    # convolution
    conv1 = tf.nn.conv2d(spectro_en, weights['wc1'], [1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    maxpool1 = tf.nn.max_pool(conv1, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')  # b*288*100*16

    conv2 = tf.nn.conv2d(maxpool1, weights['wc2'], [1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    maxpool2 = tf.nn.max_pool(conv2, [1, 4, 2, 1], [1, 4, 2, 1], padding='SAME')  # b*72*50*32

    conv3 = tf.nn.conv2d(maxpool2, weights['wc3'], [1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))
    maxpool3 = tf.nn.max_pool(conv3, [1, 3, 2, 1], [1, 3, 2, 1], padding='SAME')  # b*24*25*64

    conv4 = tf.nn.conv2d(maxpool3, weights['wc4'], [1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))
    maxpool4 = tf.nn.max_pool(conv4, [1, 3, 3, 1], [1, 3, 3, 1], padding='VALID')  # b*8*8*128

    conv5 = tf.nn.conv2d(maxpool4, weights['wc5'], [1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))
    maxpool5 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # b*4*4*256

    audio_out = maxpool5
    return audio_out, maxpool4
