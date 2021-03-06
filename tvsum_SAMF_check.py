# 用于tvsum的Self-attention实验
# 标准测试方法+6层attention+只训练attention参数+无正负样本+固定学习率+0.25dropout+25seqlen+32head+1interval
# 用原始标签训练，在st标签上测试
# 在每个类别中选择一个验证，一个测试

import os
import time
import numpy as np
import tensorflow as tf
import math
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import random
import logging
import argparse
import Transformer
from Transformer import self_attention

SERVER = 1

class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='3',type=str)
    parser.add_argument('--dropout',default='0.1',type=float)
    if SERVER == 0:
        parser.add_argument('--msd', default='TVSum_SelfAttention', type=str)
    else:
        parser.add_argument('--msd', default='model_tvsum_SA_2', type=str)
hparams = Path()
parser = hparams.parser
hp = parser.parse_args()

if SERVER == 0:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# global paras
PRESTEPS = 0
MAXSTEPS = 24000
MIN_TRAIN_STEPS = 0
WARMUP_STEP = 4000
LR_TRAIN = 1e-7
DROP_OUT = hp.dropout

EVL_EPOCHS = 1  # epochs for evaluation
L2_LAMBDA = 0.005  # weightdecay loss
GRAD_THRESHOLD = 10.0  # gradient threshold
MAX_F1 = 0.35

GPU_NUM = 1
BATCH_SIZE = 4
SEQ_INTERVAL = Transformer.INTERVAL
D_MODEL = Transformer.D_MODEL
SEQ_LEN = Transformer.SEQ_LEN

V_NUM = 2  # 3D卷积的最高一维
V_HEIGHT = 7
V_WIDTH = 7
V_CHANN = 512

A_NUM = 1  # 一个clip的A_NUM个spectro，运算时需要并入batch，保持2D卷积操作的3D输入张量
A_HEIGHT = 8
A_WIDTH = 8
A_CHANN = 128

load_ckpt_model = True

if SERVER == 0:
    # path for JD server
    LABEL_PATH = r'/public/data0/users/hulinkang/tvsum/label_record_st.json'
    INFO_PATH = r'/public/data0/users/hulinkang/tvsum/video_info.json'
    FEATURE_BASE = r'/public/data0/users/hulinkang/tvsum/feature_intermid/'
    visual_model_path = '../model_HL/pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../model_HL/pretrained/MINMSE_0.019'
    model_save_dir = r'/public/data0/users/hulinkang/model_HL/'+hp.msd+'/'
    ckpt_model_path = '../model_HL/TVSum_SelfAttention_0/STEP_30000'
    # ckpt_model_path = '../model_HL/SelfAttention_1/MAXF1_0.286_0'

else:
    # path for USTC server
    LABEL_PATH = '//data//linkang//tvsum50//label_record_st.json'
    INFO_PATH = '//data/linkang//tvsum50//video_info.json'
    FEATURE_BASE = '//data//linkang//tvsum50//feature_intermid//'
    visual_model_path = '../../model_HL/mosi_pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../../model_HL_v2/mosi_pretrained/MINMSE_0.019'
    model_save_dir = r'/data/linkang/model_HL_v3/'+hp.msd+'/'
    # ckpt_model_path = '../../model_HL_v3/model_bilibili_SA_2/STEP_9000'
    ckpt_model_path = '../../model_HL_v3/model_tvsum_SA/STEP_27000'

logging.basicConfig(level=logging.INFO)

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

def load_label_info(label_path,info_path):
    file = open(label_path,'r')
    label_record = json.load(file)
    file.close()
    file = open(info_path,'r')
    video_info = json.load(file)
    file.close()
    # 等级每个类别的视频id
    video_cat = {}
    for vid in video_info.keys():
        category = video_info[vid]['category']
        if category not in video_cat.keys():
            video_cat[category] = []
        video_cat[category].append(vid)
    return label_record, video_cat

def load_data(label_record,feature_base):
    # 只加载数据
    vids =  list(label_record.keys())
    data = {}
    for vid in vids:
        # load data & label
        visual_path = feature_base + vid + r'/features_visual_ovr.npy'
        audio_path = feature_base + vid + r'/features_audio_ovr.npy'
        visual = np.load(visual_path).reshape((-1, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
        audio = np.load(audio_path).reshape((-1, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
        labels = np.array(label_record[vid]['label'])
        scores = np.array(label_record[vid]['score'])
        temp = {}
        temp['visual'] = visual
        temp['audio'] = audio
        temp['labels'] = labels
        temp['scores'] = scores
        data[vid] = temp
    return data

def load_data_old(label_record, feature_base):
    # 装载所有特征，划分测试集
    vids = list(label_record.keys())
    data_train = {}
    data_valid = {}
    data_test = {}
    for vid in vids:
        logging.info('-'*20+str(vid)+'-'*20)
        # load data & label
        visual_path = feature_base + vid + r'/features_visual_ovr.npy'
        audio_path = feature_base + vid + r'/features_audio_ovr.npy'
        visual = np.load(visual_path).reshape((-1, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
        audio = np.load(audio_path).reshape((-1, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
        labels = np.array(label_record[vid]['label'])
        scores = np.array(label_record[vid]['score'])

        # split train & valid & test set
        trindex = label_record[vid]['train_index']
        vdindex = label_record[vid]['valid_index']
        teindex = label_record[vid]['test_index']

        temp_train = {}
        temp_train['visual'] = visual[trindex]
        temp_train['audio'] = audio[trindex]
        temp_train['labels'] = labels[trindex]
        temp_train['scores'] = scores[trindex]
        temp_train['hl_num'] = int(np.sum(temp_train['labels']))
        temp_train['nhl_num'] = len(temp_train['labels']) - temp_train['hl_num']
        data_train[vid] = temp_train

        temp_valid = {}
        temp_valid['visual'] = visual[vdindex]
        temp_valid['audio'] = audio[vdindex]
        temp_valid['labels'] = labels[vdindex]
        temp_valid['scores'] = scores[vdindex]
        temp_valid['hl_num'] = int(np.sum(temp_valid['labels']))
        temp_valid['nhl_num'] = len(temp_valid['labels']) - temp_valid['hl_num']
        data_valid[vid] = temp_valid

        temp_test = {}
        temp_test['visual'] = visual[teindex]
        temp_test['audio'] = audio[teindex]
        temp_test['labels'] = labels[teindex]
        temp_test['scores'] = scores[teindex]
        temp_test['hl_num'] = int(np.sum(temp_test['labels']))
        temp_test['nhl_num'] = len(temp_test['labels']) - temp_test['hl_num']
        data_test[vid] = temp_test

        logging.info('Data(train, valid, test): '+
                     str(temp_train['visual'].shape)+str(temp_valid['audio'].shape)+str(temp_test['audio'].shape))
        logging.info('Scores(train, valid, test): '+
                     str(len(temp_train['scores']))+str(len(temp_valid['scores']))+str(len(temp_test['scores'])))
        logging.info('Labels(train, valid, test): '+
                     str(len(temp_train['labels']))+str(len(temp_valid['labels']))+str(len(temp_test['labels'])))

    return data_train, data_valid, data_test

def split_data(video_cat,data):
    # 随机划分数据集
    data_train = {}
    data_valid = {}
    data_test = {}
    for category in video_cat.keys():
        vlist = video_cat[category]
        vlist.sort()
        valid_ids = [vlist.pop(0)]
        test_ids = [vlist.pop(0)]
        train_ids = vlist
        for vid in train_ids:
            data_train[vid] = data[vid]
        for vid in valid_ids:
            data_valid[vid] = data[vid]
        for vid in test_ids:
            data_test[vid] = data[vid]
    for vid in data_train.keys():
        d = data_train[vid]
        # logging.info('Train '+str(vid)+' (visual,audio,labels,scores): '+
        #              str(d['visual'].shape)+str(d['audio'].shape)+str(d['labels'].shape)+str(d['scores'].shape))
    for vid in data_valid.keys():
        d = data_valid[vid]
        logging.info('Valid '+str(vid) + ' (visual,audio,labels,scores): ' +
                     str(d['visual'].shape) + str(d['audio'].shape) + str(d['labels'].shape) + str(d['scores'].shape))
    for vid in data_test.keys():
        d = data_test[vid]
        logging.info('Test '+str(vid) + ' (visual,audio,labels,scores): ' +
                     str(d['visual'].shape) + str(d['audio'].shape) + str(d['labels'].shape) + str(d['scores'].shape))

    return data_train, data_valid, data_test

def train_scheme_build(data_train,seq_len,interval):
    # 加强随机化的train_scheme，直接根据step确定当前使用哪个序列，找到对应的视频中的对应位置即可
    # train_scheme = [(vid,seq_start,seq_label)]
    seq_list = []
    for vid in data_train:
        label = data_train[vid]['labels']
        vlength = len(label)
        # 对每个视频遍历所有将提取的序列，根据门限确定正负样本，归入相应的列表
        seq_start = 0
        while seq_start + seq_len <= vlength:
            seq_label = label[seq_start:seq_start+seq_len]
            seq_list.append((vid,seq_start,seq_label))
            seq_start += interval
    random.shuffle(seq_list)
    return seq_list

def get_batch_train(data,train_scheme,step,gpu_num,bc,seq_len):
    # 按照train-scheme制作batch，每次选择gpu_num*bc个序列返回即可
    seq_num = len(train_scheme)  # 序列总数

    # 每个step中从一个视频中提取gpu_num*bc个序列，顺序拼接后返回
    visual = []
    audio = []
    score = []
    label = []
    for i in range(gpu_num * bc):
        # 每次提取一个序列
        pos = (step * gpu_num * bc + i) % seq_num  # 序列位置
        vid = train_scheme[pos][0]
        start = train_scheme[pos][1]
        label_seq_orgin = train_scheme[pos][2]
        end = start + seq_len
        visual_seq = data[vid]['visual'][start:end]
        audio_seq = data[vid]['audio'][start:end]
        score_seq = data[vid]['scores'][start:end]
        label_seq = data[vid]['labels'][start:end]
        visual.append(visual_seq)
        audio.append(audio_seq)
        score.append(score_seq)
        label.append(label_seq)
        if np.sum(label_seq - label_seq_orgin) > 0:
            logging.info('\n\nError!',step,pos,vid,i,label_seq,label_seq_orgin,'\n\n')

    # reshape
    visual = np.array(visual).reshape((gpu_num*bc,seq_len,V_NUM,V_HEIGHT,V_WIDTH,V_CHANN))
    audio = np.array(audio).reshape((gpu_num*bc,seq_len,A_NUM,A_HEIGHT,A_WIDTH,A_CHANN))
    score = np.array(score).reshape((gpu_num*bc,seq_len))
    label = np.array(label).reshape((gpu_num*bc,seq_len))

    return visual, audio, score, label

def test_data_build(data_test, seq_len):
    # 按照顺序拼接所有视频的所有分段，保证不同视频的分段不会在同一个序列中出现，因此在序列水平上进行padding
    # 将每个视频的分段数目补齐到seq_len的整数倍，拆分成序列，然后顺序拼接所有序列
    # 测试时每次读取gpu_num*bc个序列，不足的部分再做padding
    visuals = []
    audios = []
    labels = []
    scores = []
    test_ids = []  # 测试的视频顺序
    for key in data_test:
        data = data_test[key]
        vlength = len(data['labels'])  # 视频中的分段数目
        padlen = seq_len - vlength % seq_len
        padlen = padlen % seq_len  # 当vlength是bc的整数倍时，不需要padding
        # 获取数据并padding
        visual = data['visual']
        audio = data['audio']
        score = data['scores']
        visual_pad = np.zeros((padlen,V_NUM,V_HEIGHT,V_WIDTH,V_CHANN))
        audio_pad = np.zeros((padlen,A_NUM,A_HEIGHT,A_WIDTH,A_CHANN))
        score_pad = np.zeros((padlen))
        visual = np.vstack((visual,visual_pad))
        audio = np.vstack((audio,audio_pad))
        score = np.hstack((score,score_pad))
        # 加入列表
        visuals.append(visual)
        audios.append(audio)
        scores.append(score)
        labels.append(data['labels'])  # 未padding
        test_ids.append(key)
    visual_concat = visuals[0]
    audio_concat = audios[0]
    score_concat = scores[0]
    for i in range(1,len(labels)):
        visual_concat = np.vstack((visual_concat, visuals[i]))
        audio_concat = np.vstack((audio_concat, audios[i]))
        score_concat = np.hstack((score_concat,scores[i]))
    data_test_concat = {}
    data_test_concat['visual_concat'] = visual_concat.reshape((-1,seq_len,V_NUM,V_HEIGHT,V_WIDTH,V_CHANN))
    data_test_concat['audio_concat'] = audio_concat.reshape((-1,seq_len,A_NUM,A_HEIGHT,A_WIDTH,A_CHANN))
    data_test_concat['scores_concat'] = score_concat.reshape((-1,seq_len))
    data_test_concat['labels'] = labels  # a list
    logging.info('Test data concatenated: '+
          str(data_test_concat['visual_concat'].shape)+
          str(data_test_concat['audio_concat'].shape)+
          str(data_test_concat['scores_concat'].shape)+
          str(len(data_test_concat['labels'])))

    return data_test_concat, test_ids

def get_batch_test(data_test_concat,step,gpu_num,bc,seq_len):
    # 每次读取gpu_num*bc个序列，在最后一个视频末尾不足的部分做padding，
    # 将N*seq_len形式的score返回到encoder做掩模，score在序列水平和batch水平都进行了padding

    batchsize = gpu_num * bc  # 实质上每个batch要返回的序列个数
    start = step * batchsize
    end = (step + 1) * batchsize
    visual = data_test_concat['visual_concat'][start:end]
    audio = data_test_concat['audio_concat'][start:end]
    score = data_test_concat['scores_concat'][start:end]

    # padding for tail
    segnum = len(data_test_concat['visual_concat'])
    if end > segnum:
        visual_pad = np.zeros((end-segnum,seq_len,V_NUM,V_HEIGHT,V_WIDTH,V_CHANN))
        audio_pad = np.zeros((end-segnum,seq_len,A_NUM,A_HEIGHT,A_WIDTH,A_CHANN))
        score_pad = np.zeros((end-segnum,seq_len))
        visual = np.vstack((visual, visual_pad))
        audio = np.vstack((audio, audio_pad))
        score = np.vstack((score,score_pad))

    return visual, audio, score

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

def conv3d(name, l_input, w, b):
    return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def score_pred(visual,audio,score,visual_weights,visual_biases,audio_weights,audio_biases,
               fusion_weights,fusion_biases,drop_out,training):
    # audio convolution
    audio_feat = tf.reshape(audio,shape=(-1,A_HEIGHT,A_WIDTH,A_CHANN))  # b*8*8*128
    audio_conv5 = tf.nn.conv2d(audio_feat, audio_weights['wc5'], [1, 1, 1, 1], padding='SAME')
    audio_conv5 = tf.nn.relu(tf.nn.bias_add(audio_conv5, audio_biases['bc5']))
    audio_out = tf.nn.max_pool(audio_conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # b*4*4*256

    # visual convolution
    visual_conv5 = conv3d('conv5b', visual, visual_weights['wc5b'], visual_biases['bc5b'])
    visual_conv5 = tf.nn.relu(visual_conv5, 'relu5b')
    visual_pool5 = tf.nn.max_pool3d(visual_conv5,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME')
    visual_pool6 = tf.transpose(visual_pool5, perm=(0, 4, 2, 3, 1))
    visual_pool6 = tf.nn.max_pool3d(visual_pool6, ksize=[1, 4, 1, 1, 1], strides=[1, 4, 1, 1, 1], padding='SAME')
    visual_pool6 = tf.transpose(visual_pool6, perm=(0, 4, 2, 3, 1))  # b*1*4*4*128
    visual_out = tf.squeeze(visual_pool6, axis=1)  # b*4*4*128

    # bilinear pooling
    A = tf.transpose(audio_out,perm=[0,3,1,2])  # b*256*4*4
    shape_A = A.get_shape().as_list()
    A = tf.reshape(A,shape=[-1,A_NUM*shape_A[1],shape_A[2]*shape_A[3]])  # b*256*16
    B = visual_out
    shape_B = B.get_shape().as_list()
    B = tf.reshape(B,shape=[-1,shape_B[1]*shape_B[2],shape_B[3]])  # b*16*128
    I = tf.matmul(A,B)  # b*256*128
    shape_I = I.get_shape().as_list()
    x = tf.reshape(I,shape=(-1,shape_I[1]*shape_I[2]))  # b*32768
    y = tf.multiply(tf.sign(x), tf.sqrt(tf.abs(x)))  # b*32768
    z = tf.nn.l2_normalize(y, dim=1)  # b*32768

    # self-attention
    # z形式为bc*seq_len个clip
    # 对encoder来说每个gpu上输入bc*seq_len*d，即每次输入bc个序列，每个序列长seq_len，每个元素维度为d
    # 在encoder中将输入的序列映射到合适的维度
    seq_input = tf.reshape(z,shape=(BATCH_SIZE,SEQ_LEN,-1))  # bc*seq_len*32768
    logits, attention_list = self_attention(seq_input, score, drop_out, training)  # bc*seq_len
    logits = tf.clip_by_value(tf.reshape(tf.sigmoid(logits), [-1, 1]), 1e-6, 0.999999)  # (bc*seq_len,1)

    # check
    # fc1 = tf.matmul(z, fusion_weights['wd1']) + fusion_biases['bd1']
    # fc1 = tf.nn.relu(fc1)
    # fc1 = tf.layers.dropout(fc1, drop_out)
    # fc2 = tf.matmul(fc1, fusion_weights['wd2']) + fusion_biases['bd2']
    # fc2 = tf.nn.relu(fc2)
    # fc2 = tf.layers.dropout(fc2, drop_out)
    # fc3 = tf.matmul(fc2, fusion_weights['wd3']) + fusion_biases['bd3']
    # fc3 = tf.nn.relu(fc3)
    # fc3 = tf.layers.dropout(fc3, drop_out)
    # fc4 = tf.matmul(fc3, fusion_weights['wd4']) + fusion_biases['bd4']
    # fc4 = tf.nn.relu(fc4)
    # fc4 = tf.layers.dropout(fc4, drop_out)
    # logits = tf.matmul(fc4, fusion_weights['wout']) + fusion_biases['bout']  # b*1
    # logits = tf.clip_by_value(tf.reshape(tf.sigmoid(logits), [-1, 1]), 1e-8, 0.99999999)  # (bc*seq_len,1)
    # attention_list = []

    return logits, attention_list

def tower_loss(name_scope,logits,labels):
    y = tf.reshape(labels,[-1,1])
    # ce = -y * (tf.log(logits)) * (1-logits) ** 2.0 *0.25 - (1 - y) * tf.log(1 - logits) * (logits) ** 2.0 * 0.75
    # loss = tf.reduce_sum(ce)
    ce = -y * (tf.log(logits)) - (1 - y) * tf.log(1 - logits)
    loss = tf.reduce_mean(ce)
    # y_p = tf.reduce_mean(logits * labels)/tf.reduce_mean(labels)
    # y_n = tf.reduce_mean(logits * (1-labels)) / tf.reduce_mean(1-labels)
    # u = 1 - y_p + y_n
    # lp = tf.nn.relu(u)
    # delta = 3
    # condition = tf.less(u, delta)
    # v = tf.square(lp) * 0.5
    # w = lp * delta - delta * delta * 0.5
    # loss = tf.where(condition, x=v, y=w)
    return loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def evaluation_ext(pred_scores, data_test, test_ids, seq_len):
    # 根据预测的分数和对应的标签计算aprf以及mse
    # 输入模型训练时的总bc，用于计算测试数据中填充部分的长度
    # 查看ext_ratio
    preds_c = list(pred_scores[0])
    for i in range(1, len(pred_scores)):
        preds_c = preds_c + list(pred_scores[i])

    ext_ratio = 0.95
    f1_max = 0
    ext_max = ext_ratio
    while round(ext_ratio,2) <= 1.05:
        pos = 0
        label_pred_all = np.array(())
        label_true_all = np.array(())

        for vid in test_ids:
            labels = data_test[vid]['labels'].reshape((-1,))
            # 计算padding，提取preds中的有效预测部分
            vlength = len(labels)
            padlen = seq_len - vlength % seq_len
            padlen = padlen % seq_len  # 当vlength是seq_len的整数倍时，不需要padding
            vlength_pad = vlength + padlen
            # 截取有效的预测部分
            preds = preds_c[pos:pos + vlength_pad]
            preds = np.array(preds).reshape((-1,))
            pos += vlength_pad  # pos按照填充后的长度移动，移动到下一个视频的预测部分起点
            preds = preds[:vlength]  # preds的有效预测长度
            # predict
            hlnum = int(np.sum(labels))
            preds_list = list(preds)
            preds_list.sort(reverse=True)
            threshold = preds_list[hlnum]
            if threshold * ext_ratio <= preds_list[0]:
                threshold = threshold * ext_ratio
            labels_pred = np.zeros_like(preds)
            for i in range(len(labels_pred)):
                if preds[i] > threshold :#and np.sum(labels_pred) < hlnum:
                    labels_pred[i] = 1
            label_true_all = np.concatenate((label_true_all, labels))
            label_pred_all = np.concatenate((label_pred_all, labels_pred))

        a = accuracy_score(label_true_all, label_pred_all)
        p = precision_score(label_true_all, label_pred_all)
        r = recall_score(label_true_all, label_pred_all)
        f = f1_score(label_true_all, label_pred_all)

        if f > f1_max:
            f1_max = f
            ext_max = ext_ratio
        logging.info('Extend: %.10f,%.3f,%.3f,%.3f,%.3f,%d,%d'%(ext_ratio,a,p,r,f,np.sum(label_true_all),np.sum(label_pred_all)))
        ext_ratio += 0.01
    logging.info('Max Ratio: %.2f,%.3f' % (ext_max, f1_max))
    return

def evaluation_simple(pred_scores, data_test, test_ids, seq_len):
    # 根据预测的分数和对应的标签计算aprf以及mse
    # 输入模型训练时的总bc，用于计算测试数据中填充部分的长度
    preds_c = list(pred_scores[0])
    for i in range(1, len(pred_scores)):
        preds_c = preds_c + list(pred_scores[i])

    pos = 0
    label_pred_all = np.array(())
    label_true_all = np.array(())
    for vid in test_ids:
        labels = data_test[vid]['labels'].reshape((-1,))
        # 计算padding，提取preds中的有效预测部分
        vlength = len(labels)
        padlen = seq_len - vlength % seq_len
        padlen = padlen % seq_len  # 当vlength是seq_len的整数倍时，不需要padding
        vlength_pad = vlength + padlen
        # 截取有效的预测部分
        preds = preds_c[pos:pos + vlength_pad]
        preds = np.array(preds).reshape((-1,))
        pos += vlength_pad  # pos按照填充后的长度移动，移动到下一个视频的预测部分起点
        preds = preds[:vlength]  # preds的有效预测长度
        # predict
        hlnum = int(np.sum(labels))
        preds_list = list(preds)
        preds_list.sort(reverse=True)
        threshold = preds_list[hlnum]
        labels_pred = np.zeros_like(preds)
        for i in range(len(labels_pred)):
            if preds[i] > threshold and np.sum(labels_pred) < hlnum:
                labels_pred[i] = 1
        label_true_all = np.concatenate((label_true_all, labels))
        label_pred_all = np.concatenate((label_pred_all, labels_pred))

    a = accuracy_score(label_true_all, label_pred_all)
    p = precision_score(label_true_all, label_pred_all)
    r = recall_score(label_true_all, label_pred_all)
    f = f1_score(label_true_all, label_pred_all)
    logging.info('APRF: %.3f,%.3f,%.3f,%.3f' % (a, p, r, f))
    return a,p,r,f

def model_search(model_save_dir):
    def takestep(name):
        return int(name.split('-')[0].split('S')[-1])
    # 找到要验证的模型名称
    model_to_restore = []
    for root,dirs,files in os.walk(model_save_dir):
        for file in files:
            if file.endswith('.meta'):
                model_name = file.split('.meta')[0]
                model_to_restore.append(os.path.join(root, model_name))
            # if file.startswith('MINMSE'):
            #     model_name = 'MINMSE_0.' + file.split('.')[1]
            #     model_to_restore.append(os.path.join(root, model_name))
            # if file.startswith('MAXF1'):
            #     model_name = 'MAXF1_0.' + file.split('.')[1]
            #     model_to_restore.append(os.path.join(root, model_name))
            # if file.startswith('STEP'):
            #     model_name = file.split('.')[0]
            #     model_to_restore.append(os.path.join(root,model_name))

    model_to_restore = list(set(model_to_restore))
    model_to_restore.sort(key=takestep)
    return model_to_restore

def run_training(data_train, data_test, data_test_concat, test_ids, model_path, test_mode):
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # placeholders
        visual_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,
                                                         SEQ_LEN,
                                                         V_NUM,
                                                         V_HEIGHT,
                                                         V_WIDTH,
                                                         V_CHANN))
        audio_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,
                                                        SEQ_LEN,
                                                        A_NUM,
                                                        A_HEIGHT,
                                                        A_WIDTH,
                                                        A_CHANN))
        scores_holder = tf.placeholder(tf.float32, shape=(BATCH_SIZE * GPU_NUM, SEQ_LEN))
        labels_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM, SEQ_LEN))
        dropout_holder = tf.placeholder(tf.float32,shape=())
        training_holder = tf.placeholder(tf.bool,shape=())

        # parameters
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
            }
            biases = {
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
            }
        with tf.variable_scope('var_name_audio') as var_scope_audio:
            audio_weights = {
                'wc5': _variable_with_weight_decay('au_wc5', [3, 3, 128, 256], L2_LAMBDA),
            }
            audio_biases = {
                'bc5': _variable_with_weight_decay('au_bc5', [256], 0.0000),
            }
        varlist_visual = list(weights.values()) + list(biases.values())
        varlist_audio = list(audio_weights.values()) + list(audio_biases.values())
        # training operations
        lr = noam_scheme(LR_TRAIN,global_step,WARMUP_STEP)
        opt_train = tf.train.AdamOptimizer(LR_TRAIN)

        # graph building
        tower_grads_train = []
        logits_list = []
        loss_list = []
        attention_list = []
        for gpu_index in range(GPU_NUM):
            with tf.device('/gpu:%d' % gpu_index):
                visual = visual_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :, :, :]
                visual = tf.reshape(visual,shape=(BATCH_SIZE*SEQ_LEN,V_NUM,V_HEIGHT,V_WIDTH,V_CHANN))
                audio = audio_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :, :, :]
                audio = tf.reshape(audio,shape=(BATCH_SIZE*SEQ_LEN,A_NUM,A_HEIGHT,A_WIDTH,A_CHANN))
                labels = labels_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                scores = scores_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]

                # predict scores
                logits, atlist_one = score_pred(visual,audio,scores,weights,biases,audio_weights,audio_biases,
                                    None,None,dropout_holder,training_holder)
                logits_list.append(logits)
                attention_list += atlist_one  # 逐个拼接各个卡上的attention_list
                # calculate loss & gradients
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = tower_loss(loss_name_scope, logits, labels)
                varlist = tf.trainable_variables()  # 全部训练
                varlist = list(set(varlist) - set(varlist_visual) - set(varlist_audio))
                # varlist = varlist + list(biases.values()) + list(audio_biases.values())
                grads_train = opt_train.compute_gradients(loss, varlist)
                thresh = GRAD_THRESHOLD  # 梯度截断 防止爆炸
                grads_train_cap = [(tf.clip_by_value(grad, -thresh, thresh), var) for grad, var in grads_train]
                tower_grads_train.append(grads_train_cap)
                loss_list.append(loss)
        grads_t = average_gradients(tower_grads_train)
        train_op = opt_train.apply_gradients(grads_t, global_step=global_step)
        if test_mode == 1:
            train_op = tf.no_op()

        # session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)

        # load model
        saver_visual = tf.train.Saver(varlist_visual)
        saver_audio = tf.train.Saver(varlist_audio)
        saver_visual.restore(sess, visual_model_path)
        saver_audio.restore(sess, audio_model_path)

        saver_overall = tf.train.Saver(max_to_keep=100)
        if load_ckpt_model:
            # logging.info(' Ckpt Model Restoring: ' + model_path)
            saver_overall.restore(sess, model_path)
            # logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        max_test_step = math.ceil(len(data_test_concat['visual_concat']) / BATCH_SIZE / GPU_NUM)
        # 固定正负样本比例
        # train_scheme = train_scheme_build_v2(data_train, SEQ_LEN, SEQ_INTERVAL)
        # epoch_step = math.ceil(len(train_scheme[0]) / (BATCH_SIZE * GPU_NUM - 1))
        # 不区分正负样本
        train_scheme = train_scheme_build(data_train, SEQ_LEN, SEQ_INTERVAL)
        epoch_step = math.ceil(len(train_scheme) / (BATCH_SIZE * GPU_NUM))

        # Beging training
        ob_loss = []
        timepoint = time.time()
        for step in range(MAXSTEPS):
            visual_b, audio_b, score_b, label_b = get_batch_train(data_train, train_scheme, step,GPU_NUM,BATCH_SIZE,SEQ_LEN)
            observe = sess.run([tf.no_op()] + loss_list + logits_list + attention_list + [global_step, lr],
                               feed_dict={visual_holder: visual_b,
                                          audio_holder: audio_b,
                                          scores_holder: score_b,
                                          labels_holder: label_b,
                                          dropout_holder: DROP_OUT,
                                          training_holder: True})

            loss_batch = np.array(observe[1:1+GPU_NUM])
            ob_loss.append(loss_batch)  # 卡0和卡1返回的是来自同一个batch的两部分loss，求平均

            # save checkpoint &  evaluate
            epoch = step / epoch_step
            if step % epoch_step == 0 or (step + 1) == MAXSTEPS:
                if step == 0 and test_mode == 0:
                    continue
                duration = time.time() - timepoint
                timepoint = time.time()
                loss_array = np.array(ob_loss)
                ob_loss.clear()
                logging.info(' Step %d: %.3f sec' % (step, duration))
                logging.info(' Evaluate: '+str(step)+' Epoch: '+str(epoch))
                logging.info(' Average Loss: '+str(np.mean(loss_array))+' Min Loss: '+str(np.min(loss_array))+' Max Loss: '+str(np.max(loss_array)))

                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                pred_scores = []  # 每个batch输出的预测得分
                for test_step in range(max_test_step):
                    visual_b, audio_b, score_b = get_batch_test(data_test_concat, test_step,
                                                                GPU_NUM, BATCH_SIZE, SEQ_LEN)
                    logits_temp_list = sess.run(logits_list, feed_dict={visual_holder: visual_b,
                                                                        audio_holder: audio_b,
                                                                        scores_holder: score_b,
                                                                        training_holder: False,
                                                                        dropout_holder: 0})
                    for preds in logits_temp_list:
                        pred_scores.append(preds.reshape((-1)))
                evaluation_simple(pred_scores, data_test, test_ids, SEQ_LEN)
                return
    return

def main(self):
    label_record, video_cat = load_label_info(LABEL_PATH,INFO_PATH)
    # data = load_data(label_record, FEATURE_BASE)
    # data_train, data_valid, data_test = split_data(video_cat,data)
    data_train, data_valid, data_test = load_data_old(label_record, FEATURE_BASE)
    logging.info('Data loaded !')

    data_test_actual = data_valid
    data_test_concat, test_ids = test_data_build(data_test_actual, SEQ_LEN)
    models_to_restore = model_search(model_save_dir)
    # models_to_restore = ['../../model_HL_v3/model_bilibili_SA_6l_2/MAXF1_0.286_0']
    for i in range(len(models_to_restore)):
        print('-' * 20, i, models_to_restore[i].split('/')[-1], '-' * 20)
        ckpt_model_path = models_to_restore[i]
        run_training(data_train, data_test_actual, data_test_concat, test_ids, ckpt_model_path, 1)  # for testing
    logging.info(LABEL_PATH)
if __name__ == "__main__":
    tf.app.run()