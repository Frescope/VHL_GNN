# 主要用于transformer在VHL问题上的验证工作，tvsum上的实验
# 加载预训练的参数，但是只提取AudioNet与VisualNet倒数第二层卷积操作的输出，余下的卷积层参数随self-attention优化
# 提取audio和visual对应的张量后，存储为numpy形式，后续self-attention网络完成最后的两个卷积层、bilinear pooling以及enoder

import os
import time
import numpy as np
import tensorflow as tf
import AudioNet
import VisualNet
import math
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'
# global paras
GPU_NUM = 2
load_audio_pretrained = True
load_visual_pretrained = True

BATCH_SIZE = 10
L2_LAMBDA = 0.05  # weightdecay loss

V_NUM = VisualNet.FRAME_NUM
V_HEIGHT = VisualNet.FRAME_HEIGHT
V_WIDTH = VisualNet.FRAME_WIDTH
V_CHANN = VisualNet.FRAME_CHANN

A_NUM = 1
A_HEIGHT = AudioNet.SPECTRO_HEIGHT
A_WIDTH = AudioNet.SPECTRO_WIDTH
A_IS_LEN = AudioNet.IS_LENGTH

LABEL_PATH = '//data//linkang//tvsum50//label_record.json'
FEATURE_BASE = '//data//linkang//tvsum50//feature//'
FEATURE_EXT_DIR = '//data//linkang//tvsum50//feature_intermid//'
visual_model_path = '../../model_HL/mosi_pretrained/sports1m_finetuning_ucf101.model'
audio_model_path = '../../model_HL_v2/mosi_pretrained/MINMSE_0.019'

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
    # 装载所有特征，划分测试集
    vids = list(label_record.keys())
    data_train = {}
    data_valid = {}
    data_test = {}
    for vid in vids:
        print('-'*20, vid, '-'*20)
        # load data & label
        frame_path = feature_base + vid + '//frames.npy'
        spectro_path = feature_base + vid + '//spectros.npy'
        smile_path = feature_base + vid + '//is09.npy'
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

        print('Data: ', temp_train['frame'].shape)
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

def feat_extract(frames, spectro, smile, weights, biases, audio_weights, audio_biases):
    spectrograms = tf.reshape(spectro, shape=(-1, A_HEIGHT, A_WIDTH))  # b*384*400
    EIfeat = tf.reshape(smile, shape=(-1, A_IS_LEN, 1))
    _, audio_out = AudioNet.audionet(spectrograms, EIfeat, audio_weights, audio_biases)  # b*8*8*128
    _, visual_out = VisualNet.visualnet(frames, weights, biases)  # b*2*7*7*512
    return audio_out, visual_out

def run(data):
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
        varlist_visual = list(weights.values()) + list(biases.values())
        varlist_audio = list(audio_weights.values()) + list(audio_biases.values())
        varlist_visual_load = list(set(varlist_visual) - {weights['wc5b'], biases['bc5b']})
        varlist_audio_load = list(set(varlist_audio) - {audio_weights['wc5'],audio_biases['bc5']})

        audio_list = []
        visual_list = []
        scores_list = []
        for gpu_index in range(GPU_NUM):
            with tf.device('/gpu:%d' % gpu_index):
                frames = frames_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :, :]
                spectro = spectro_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :]
                smile = smile_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :]
                labels = labels_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE]
                scores = scores_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE]

                # predict scores
                audio_out, visual_out = feat_extract(frames, spectro, smile, weights, biases, audio_weights, audio_biases)
                audio_list.append(audio_out)
                visual_list.append(visual_out)
                scores_list.append(scores)

        saver_audio = tf.train.Saver(varlist_audio_load)  # saver for load pretrained paras
        saver_visual = tf.train.Saver(varlist_visual_load)  # saver for load pretrained paras
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        if load_audio_pretrained:
            print('Mosi Model Restoring: ', audio_model_path)
            saver_audio.restore(sess, audio_model_path)
            print('Mosi Model Resrtored !')
        if load_visual_pretrained:
            print('C3D Model Restoring: ', visual_model_path)
            saver_visual.restore(sess, visual_model_path)
            print('C3D Model Resrtored !')

        # 对于每个视频，提取出全部的最终特征
        vids = list(data.keys())
        for i in range(len(vids)):
            vid = vids[i]
            print('-'*20,i,vid,'-'*20)
            test_step_video = math.ceil(len(data[vid]['frame']) / BATCH_SIZE / GPU_NUM)
            feats_visual_list = []
            feats_audio_list = []
            check_list = []
            for step in range(test_step_video):
                frame_b, spectro_b, smile_b, score_b = get_batch(data[vid], step, BATCH_SIZE * GPU_NUM)
                visual_temp,audio_temp,check_temp = sess.run([visual_list,audio_list,scores_list], feed_dict={frames_holder:frame_b,
                                                             spectro_holder:spectro_b,
                                                             smile_holder:smile_b,
                                                             scores_holder:score_b,
                                                             dropout_holder:1.0})
                feats_visual_list += visual_temp
                feats_audio_list += audio_temp
                check_list += check_temp

            # check 检查输出顺序
            features_audio_ovr = np.array(feats_audio_list).reshape((-1,8,8,128))  # ?*b*8*8*128
            features_visual_ovr = np.array(feats_visual_list).reshape((-1,2,7,7,512))  # ?*b*2*7*7*512
            scores_check = np.array(check_list).reshape((-1))
            scores_origin = data[vid]['scores']
            print('check:',scores_check.shape, scores_origin.shape)
            print('Data:',features_visual_ovr.shape, features_audio_ovr.shape)
            scores_check = scores_check[:len(scores_origin)]
            features_visual_ovr = features_visual_ovr[:len(scores_origin)]
            features_audio_ovr = features_audio_ovr[:len(scores_origin)]
            print(np.sum(scores_check - scores_origin),features_visual_ovr.shape,features_audio_ovr.shape)

            # output
            if not os.path.isdir(FEATURE_EXT_DIR + vid + '//'):
                os.makedirs(FEATURE_EXT_DIR + vid + '//')
            np.save(FEATURE_EXT_DIR+vid+'//features_visual_ovr.npy',features_visual_ovr[:len(data[vid]['labels'])])
            np.save(FEATURE_EXT_DIR+vid+'//features_audio_ovr.npy',features_audio_ovr[:len(data[vid]['labels'])])
            print('Overall Features: ',features_visual_ovr.shape, features_audio_ovr.shape)

    return

def main(self):
    label_record = load_label(LABEL_PATH)
    data = load_data(label_record, FEATURE_BASE)
    print('Data Loaded !')
    run(data)

if __name__ == '__main__':
    tf.app.run()