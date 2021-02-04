# 使用AEmoVH作为特征提取器，提取每个分段的特征提供给transformer使用
# 仍然使用旧24s的数据，提取所有分段的最终特征并保存

import os
import time
import numpy as np
import tensorflow as tf
import AudioNet
import VisualNet
import math
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
# global paras
GPU_NUM = 2
load_ckpt_model = True
PRESTEPS = 0
MAXSTEPS = 9000
LR_TRAIN = 1e-6
LR_FINETUNE = 1e-6

EVL_EPOCHS = 1  # epochs for evaluation
BATCH_SIZE = 10
L2_LAMBDA = 0.05  # weightdecay loss
GRAD_THRESHOLD = 10.0  # gradient threshold
MAX_F1 = 0.27

V_NUM = VisualNet.FRAME_NUM
V_HEIGHT = VisualNet.FRAME_HEIGHT
V_WIDTH = VisualNet.FRAME_WIDTH
V_CHANN = VisualNet.FRAME_CHANN

A_NUM = AudioNet.SPECTRO_NUM
A_HEIGHT = AudioNet.SPECTRO_HEIGHT
A_WIDTH = AudioNet.SPECTRO_WIDTH
A_IS_LEN = AudioNet.IS_LENGTH

LABEL_PATH = '//data//linkang//bilibili//label_record_zmn_24s.json'
FEATURE_BASE = '//data//linkang//bilibili//feature//'
visual_model_path = '../../model_HL/mosi_pretrained/sports1m_finetuning_ucf101.model'
audio_model_path = '../../model_HL_v2/mosi_pretrained/MINMSE_0.019'
model_path = '../../model_HL_v3/PM-MF_bilibili_3/MAXF1_0.304_0'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def load_label(label_path):
    file = open(label_path,'r')
    label_record = json.load(file)
    file.close()
    return label_record

def load_data(label_record, feature_base):
    # 装载所有特征，全部放入训练集中
    # 按照每个vid计算一次，形成的特征数组也按照vid存储
    vids = list(label_record.keys())
    data_train = {}
    data_valid = {}
    data_test = {}
    for vid in vids:
        print('-'*20, vid, '-'*20)
        # load data & label
        # label顺序：train-valid-test
        frame_path = feature_base + vid + '//frames_v2_24s.npy'
        spectro_path = feature_base + vid + '//spectros_v2_24s.npy'
        smile_path = feature_base + vid + '//is09_v2_24s.npy'
        frame = np.load(frame_path).reshape((-1, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
        spectro = np.load(spectro_path).reshape((-1, A_NUM, A_HEIGHT, A_WIDTH))
        smile = np.load(smile_path).reshape((-1, A_NUM, A_IS_LEN, 1))
        labels = np.array(label_record[vid]['label'])
        scores = np.array(label_record[vid]['score'])

        temp_train = {}
        temp_train['frame'] = frame
        temp_train['spectro'] = spectro
        temp_train['smile'] = smile
        temp_train['labels'] = labels
        temp_train['scores'] = scores
        temp_train['hl_num'] = int(np.sum(temp_train['labels']))
        temp_train['nhl_num'] = len(temp_train['labels']) - temp_train['hl_num']
        data_train[vid] = temp_train

        print('Data: ',temp_train['frame'].shape)
        print('Scores: ', len(temp_train['scores']))

    return data_train

def get_batch(data,steps,batch_size):
    # 在每个vid的特征中循环若干个batch
    # 全部返回定长分段，这里的batchsize指分段总长，所有batch计算完毕后根据label长度截断即可
    start = steps * batch_size
    end = (steps + 1) * batch_size
    frame = data['frame'][start:end]
    spectro = data['spectro'][start:end]
    smile = data['smile'][start:end]
    score = data['scores'][start:end]

    # padding for tail
    segnum = len(data['frame'])
    if end > segnum:
        frame_pad = np.zeros((end-segnum,V_NUM,V_HEIGHT,V_WIDTH,V_CHANN))
        spectro_pad = np.zeros((end-segnum,A_NUM,A_HEIGHT,A_WIDTH))
        smile_pad = np.zeros((end-segnum,A_NUM,A_IS_LEN,1))
        score_pad = np.zeros((end-segnum))
        frame = np.vstack((frame, frame_pad))
        spectro = np.vstack((spectro, spectro_pad))
        smile = np.vstack((smile, smile_pad))
        score = np.hstack((score,score_pad))

    return frame, spectro, smile, score

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def feat_extract(frames, spectro, smile, weights, biases, audio_weights, audio_biases,
               fusion_weights, fusion_biases, dropout):
    spectrograms = tf.reshape(spectro, shape=(-1, A_HEIGHT, A_WIDTH))  # 6b*384*400
    EIfeat = tf.reshape(smile, shape=(-1, A_IS_LEN, 1))
    audio_out = AudioNet.audionet(spectrograms, EIfeat, audio_weights, audio_biases)  # 6b*4*4*256
    visual_out = VisualNet.visualnet(frames, weights, biases)  # b*1*4*4*128
    visual_out = tf.squeeze(visual_out, axis=1)  # b*4*4*128

    # bilinear pooling
    A = tf.transpose(audio_out, perm=[0, 3, 1, 2])  # 6b*256*4*4
    shape_A = A.get_shape().as_list()
    A = tf.reshape(A, shape=[-1, A_NUM * shape_A[1], shape_A[2] * shape_A[3]])  # b*1536*16
    B = visual_out
    shape_B = B.get_shape().as_list()
    B = tf.reshape(B, shape=[-1, shape_B[1] * shape_B[2], shape_B[3]])  # b*16*128
    I = tf.matmul(A, B)  # b*1536*128
    shape_I = I.get_shape().as_list()
    x = tf.reshape(I, shape=(-1, shape_I[1] * shape_I[2]))  # b*196608
    y = tf.multiply(tf.sign(x), tf.sqrt(tf.abs(x)))  # b*196608
    z = tf.nn.l2_normalize(y, dim=1)  # b*196608

    fc1 = tf.matmul(z, fusion_weights['wd1']) + fusion_biases['bd1']
    return fc1

def run(data, model_path):
    # 每次提取一个vid对应的特征，输入这个vid在data中对应的值，根据label长度截断，返回特征组成的数组
    with tf.Graph().as_default():
        # placeholders
        frames_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,
                                                         V_NUM,
                                                         V_HEIGHT,
                                                         V_WIDTH,
                                                         V_CHANN))
        spectro_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,
                                                          A_NUM,
                                                          A_HEIGHT,
                                                          A_WIDTH))
        smile_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,
                                                        A_NUM,
                                                        A_IS_LEN,
                                                        1))
        labels_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM))
        scores_holder = tf.placeholder(tf.float32, shape=(BATCH_SIZE * GPU_NUM))
        dropout_holder = tf.placeholder(tf.float32,shape=())

        # parameters
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
            }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
            }
        with tf.variable_scope('var_name_audio') as var_scope_audio:
            audio_weights = {
                'wc1': _variable_with_weight_decay('au_wc1', [3, 3, 1, 16], L2_LAMBDA),
                'wc2': _variable_with_weight_decay('au_wc2', [3, 3, 16, 32], L2_LAMBDA),
                'wc3': _variable_with_weight_decay('au_wc3', [3, 3, 32, 64], L2_LAMBDA),
                'wc4': _variable_with_weight_decay('au_wc4', [3, 3, 64, 128], L2_LAMBDA),
                'wc5': _variable_with_weight_decay('au_wc5', [3, 3, 128, 256], L2_LAMBDA),
            }
            audio_biases = {
                'bc1': _variable_with_weight_decay('au_bc1', [16], 0.0000),
                'bc2': _variable_with_weight_decay('au_bc2', [32], 0.0000),
                'bc3': _variable_with_weight_decay('au_bc3', [64], 0.0000),
                'bc4': _variable_with_weight_decay('au_bc4', [128], 0.0000),
                'bc5': _variable_with_weight_decay('au_bc5', [256], 0.0000),
            }
        with tf.variable_scope('var_name_fusion') as var_name_fusion:
            fusion_weights = {
                'wd1': _variable_with_weight_decay('wd1', [196608, 512], L2_LAMBDA),
                'wd2': _variable_with_weight_decay('wd2', [512, 512], L2_LAMBDA),
                'wd3': _variable_with_weight_decay('wd3', [512, 256], L2_LAMBDA),
                'wd4': _variable_with_weight_decay('wd4', [256, 64], L2_LAMBDA),
                'wout': _variable_with_weight_decay('wout', [64, 1], L2_LAMBDA),
            }
            fusion_biases = {
                'bd1': _variable_with_weight_decay('bd1', [512], 0.0000),
                'bd2': _variable_with_weight_decay('bd2', [512], 0.0000),
                'bd3': _variable_with_weight_decay('bd3', [256], 0.0000),
                'bd4': _variable_with_weight_decay('bd4', [64], 0.0000),
                'bout': _variable_with_weight_decay('bout', [1], 0.0000),
            }

        preds_list = []
        scores_list = []
        for gpu_index in range(GPU_NUM):
            with tf.device('/gpu:%d' % gpu_index):
                frames = frames_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :, :]
                spectro = spectro_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :]
                smile = smile_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :]
                labels = labels_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE]
                scores = scores_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE]

                # predict scores
                preds = feat_extract(frames, spectro, smile, weights, biases, audio_weights, audio_biases,
                                   fusion_weights, fusion_biases, dropout_holder)
                preds_list.append(preds)
                scores_list.append(scores)

        saver_overall = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        print('Ckpt Model Restoring: ', model_path)
        saver_overall.restore(sess, model_path)
        print('Ckpt Model Resrtored !')

        # 对于每个视频，提取出全部的最终特征
        vids = list(data.keys())
        for i in range(len(vids)):
            vid = vids[i]
            print('-'*20,i,vid,'-'*20)
            test_step_video = math.ceil(len(data[vid]['frame']) / BATCH_SIZE / GPU_NUM)
            feats_list = []  # 这一视频中提取的全部特征
            check_list = []
            for step in range(test_step_video):
                frame_b, spectro_b, smile_b, score_b = get_batch(data[vid], step, BATCH_SIZE * GPU_NUM)
                feats_temp,check_temp = sess.run([preds_list,scores_list], feed_dict={frames_holder:frame_b,
                                                             spectro_holder:spectro_b,
                                                             smile_holder:smile_b,
                                                             scores_holder:score_b,
                                                             dropout_holder:1.0})
                feats_list += feats_temp
                check_list += check_temp

            # check 检查输出顺序
            scores_check = np.array(check_list).reshape((-1))
            scores_origin = data[vid]['scores']
            temp = scores_check[:len(scores_origin)] - scores_origin
            print('check:',scores_check.shape, data[vid]['scores'].shape)
            scores_check = scores_check[:len(data[vid]['scores'])]
            print(np.sum(scores_check - data[vid]['scores']))

            # output
            features_ovr = np.array(feats_list).reshape((-1, 512))
            features_ovr = features_ovr[:len(data[vid]['frame'])]
            np.save(FEATURE_BASE+vid+'//features_ovr.npy',features_ovr)
            print('Overall Features: ',features_ovr.shape)

    return

def main(self):
    label_record = load_label(LABEL_PATH)
    data = load_data(label_record, FEATURE_BASE)
    print('Data Loaded !')
    run(data, model_path)

if __name__ == '__main__':
    tf.app.run()

