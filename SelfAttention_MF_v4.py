# 用于实验8：在Bilibili数据集上实现新的训练方法，使用每个序列的中间一位节点的输出作为这一节点的预测，
# 使用pairwise训练的方法，训练时选择正负样本的各自邻域节点分别组成一个序列输入attention，
# 测试时用滑动窗口的方法获取每个样本的邻域并计算样本的输出。

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
import Transformer_v2
from Transformer_v2 import self_attention

SERVER = 0

class Path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1',type=str)
    parser.add_argument('--num_heads',default=32,type=int)
    parser.add_argument('--num_blocks',default=5,type=int)
    parser.add_argument('--seq_len',default=15,type=int)
    parser.add_argument('--bc',default=4,type=int)
    parser.add_argument('--dropout',default='0.1',type=float)
    parser.add_argument('--gpu_num',default=1,type=int)
    if SERVER == 0:
        parser.add_argument('--msd', default='SelfAttention', type=str)
    else:
        parser.add_argument('--msd', default='model_bilibili_SA', type=str)
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
WARMUP_STEP = 4000
MIN_TRAIN_STEPS = 0
MAXSTEPS = 20000
PHASES_STEPS = [2000]
PHASES_LR = [1e-6,1e-7]
HIDDEN_SIZE = 128  # for lstm
DROP_OUT = hp.dropout

EVL_EPOCHS = 1  # epochs for evaluation
L2_LAMBDA = 0.005  # weightdecay loss
GRAD_THRESHOLD = 10.0  # gradient threshold
MAX_F1 = 0.21

GPU_NUM = hp.gpu_num
BATCH_SIZE = hp.bc
D_MODEL = Transformer_v2.D_MODEL
SEQ_LEN = hp.seq_len
NUM_BLOCKS = hp.num_blocks
NUM_HEADS = hp.num_heads

V_NUM = 2  # 3D卷积的最高一维
V_HEIGHT = 7
V_WIDTH = 7
V_CHANN = 512

A_NUM = 6  # 一个clip的A_NUM个spectro，运算时需要并入batch，保持2D卷积操作的3D输入张量
A_HEIGHT = 8
A_WIDTH = 8
A_CHANN = 128

load_ckpt_model = False

if SERVER == 0:
    # path for JD server
    LABEL_PATH = r'/public/data0/users/hulinkang/bilibili/label_record_zmn_24s.json'
    FEATURE_BASE = r'/public/data0/users/hulinkang/bilibili/feature/'
    visual_model_path = '../model_HL/pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../model_HL/pretrained/MINMSE_0.019'
    model_save_dir = r'/public/data0/users/hulinkang/model_HL/'+hp.msd+'/'
    ckpt_model_path = '../model_HL/SelfAttention_3/STEP_30000'
    # ckpt_model_path = '../model_HL/SelfAttention_1/MAXF1_0.286_0'

else:
    # path for USTC server
    LABEL_PATH = '//data//linkang//bilibili//label_record_zmn_24s.json'
    FEATURE_BASE = '//data//linkang//bilibili//feature//'
    visual_model_path = '../../model_HL/mosi_pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../../model_HL_v2/mosi_pretrained/MINMSE_0.019'
    model_save_dir = r'/data/linkang/model_HL_v3/'+hp.msd+'/'
    # ckpt_model_path = '../../model_HL_v3/model_bilibili_SA_2/STEP_9000'
    ckpt_model_path = '../../model_HL_v3/model_bilibili_SA_6l/STEP_27000'

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

def load_label(label_path):
    file = open(label_path,'r')
    label_record = json.load(file)
    file.close()
    return label_record

def load_data(label_record, feature_base):
    # 装载所有特征，划分测试集
    # 增加对所有正样本和负样本的索引，处理score使所有元素都大于零
    vids = list(label_record.keys())
    data_train = {}
    data_valid = {}
    data_test = {}
    for vid in vids:
        logging.info('-'*20+str(vid)+'-'*20)
        # load data & label
        # label顺序：train-valid-test
        visual_path = feature_base + vid + r'/features_visual_ovr.npy'
        audio_path = feature_base + vid + r'/features_audio_ovr.npy'
        visual = np.load(visual_path).reshape((-1, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
        audio = np.load(audio_path).reshape((-1, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
        labels = np.array(label_record[vid]['label'])
        scores = np.array(label_record[vid]['score'])
        scores = scores - np.min(scores) + 1e-6  # 将scores调整到最小值为1e-6，方差仍为1

        # split train & valid & test set
        valid_pos = round(len(labels) * 0.6)
        test_pos = round(len(labels) * 0.8)

        temp_train = {}
        temp_train['visual'] = visual[:valid_pos]
        temp_train['audio'] = audio[:valid_pos]
        temp_train['labels'] = labels[:valid_pos]
        temp_train['scores'] = scores[:valid_pos]
        temp_train['pos_index'] = np.where(temp_train['labels'] > 0)[0]  # 正样本索引
        temp_train['neg_index'] = np.where(temp_train['labels'] < 1)[0]  # 负样本索引
        data_train[vid] = temp_train

        temp_valid = {}
        temp_valid['visual'] = visual[valid_pos:test_pos]
        temp_valid['audio'] = audio[valid_pos:test_pos]
        temp_valid['labels'] = labels[valid_pos:test_pos]
        temp_valid['scores'] = scores[valid_pos:test_pos]
        temp_valid['pos_index'] = np.where(temp_valid['labels'] > 0)[0]
        temp_valid['neg_index'] = np.where(temp_valid['labels'] < 1)[0]
        data_valid[vid] = temp_valid

        temp_test = {}
        temp_test['visual'] = visual[test_pos:]
        temp_test['audio'] = audio[test_pos:]
        temp_test['labels'] = labels[test_pos:][:len(temp_test['visual'])]  # 截断
        temp_test['scores'] = scores[test_pos:][:len(temp_test['visual'])]  # 截断
        temp_test['pos_index'] = np.where(temp_test['labels'] > 0)[0]
        temp_test['neg_index'] = np.where(temp_test['labels'] < 1)[0]
        data_test[vid] = temp_test

        logging.info('Data(train, valid, test): '+str(temp_train['visual'].shape)+str(temp_valid['audio'].shape)+str(temp_test['labels'].shape))
        logging.info('Scores(train, valid, test): '+str(len(temp_train['scores']))+str(len(temp_valid['scores']))+str(len(temp_test['scores'])))

    return data_train, data_valid, data_test

def train_scheme_build_v3(data_train,seq_len):
    # 根据正负样本制定的train_scheme，取每个样本的左右领域与样本共同构成一个序列，分别得到正样本序列与负样本序列
    # 在getbatch时数据用零填充，score也用零填充，在attention计算时根据score将负无穷输入softmax，消除padding片段对有效片段的影响
    # 正负样本序列生成后随机化，直接根据step确定当前使用哪个序列，正负各取一个计算pairwise loss
    # train_scheme = [pos_list=(vid,seq_start,seq_end,sample_pos,sample_label),neg_list=()]

    pos_list = []
    neg_list = []
    for vid in data_train:
        label = data_train[vid]['labels']
        pos_index = data_train[vid]['pos_index']
        neg_index = data_train[vid]['neg_index']
        vlength = len(label)
        # 遍历正样本索引与负样本索引中的所有样本，计算其邻域索引范围，分别加入两个列表
        for sample_pos in pos_index:
            seq_start = sample_pos - int(seq_len / 2)
            seq_end = seq_start + seq_len
            seq_start = max(0,seq_start)  # 截断
            pos_list.append((vid,seq_start,seq_end,sample_pos,1))
        for sample_pos in neg_index:
            seq_start = sample_pos - int(seq_len / 2)
            seq_end = seq_start + seq_len
            seq_start = max(0, seq_start)  # 截断
            neg_list.append((vid,seq_start,seq_end,sample_pos,0))

    random.shuffle(pos_list)
    random.shuffle(neg_list)
    return (pos_list,neg_list)

def get_batch_train(data,train_scheme,step,gpu_num,bc,seq_len):
    # 按照train-scheme制作batch，每次选择gpu_num*bc个序列返回，要求每个bc中一半是正样本一半是负样本，交替排列
    # 每个序列构成一个sample，故共有gpu_num*bc个sample，每个gpu上计算bc个sample的loss
    # 返回gpu_num*bc个label，对应每个sample中一个片段的标签
    # 同时返回一个取样位置序列sample_pos，顺序记录每个sample中标签对应的片段在序列中的位置，模型输出后根据sample_pos计算loss
    # 根据step顺序读取pos_list与neg_list中的序列并组合为batch_index，再抽取对应的visual，audio，score与label
    pos_list,neg_list = train_scheme
    pos_num = len(pos_list)
    neg_num = len(neg_list)

    # 生成batch_index与sample_pos
    batch_index = []
    sample_poses = []
    batch_labels = []  # only for check
    for i in range(int(gpu_num * bc / 2)):  # gpu_num*bc应当为偶数
        pos_position = (step * int(gpu_num * bc / 2) + i) % pos_num  # 当前在pos_list中的起始位置
        neg_position = (step * int(gpu_num * bc / 2) + i) % neg_num  # 当前在neg_list中的起始位置
        # 读正样本
        vid,seq_start,seq_end,sample_pos,sample_label = pos_list[pos_position]
        batch_index.append((vid,seq_start,seq_end,sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)
        # 读负样本
        vid, seq_start, seq_end, sample_pos, sample_label = neg_list[neg_position]
        batch_index.append((vid, seq_start, seq_end, sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)

    # 根据索引读取数据，并做padding
    visuals = []
    audios = []
    scores = []
    labels = []
    for i in range(len(batch_index)):
        vid,seq_start,seq_end,sample_pos = batch_index[i]
        vlength = len(data[vid]['labels'])
        seq_end = min(vlength,seq_end)  # 截断
        padding_len = seq_len - (seq_end - seq_start)
        visual = data[vid]['visual'][seq_start:seq_end]
        audio = data[vid]['audio'][seq_start:seq_end]
        score = data[vid]['scores'][seq_start:seq_end]
        if padding_len > 0:
            visual_pad = np.zeros((padding_len, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
            audio_pad = np.zeros((padding_len, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
            score_pad = np.zeros((padding_len,))
            visual = np.vstack((visual,visual_pad))  # 统一在后侧padding
            audio = np.vstack((audio,audio_pad))
            score = np.hstack((score, score_pad))
        visuals.append(visual)
        audios.append(audio)
        scores.append(score)
        labels.append(data[vid]['labels'][sample_pos])
    visuals = np.array(visuals).reshape((gpu_num * bc, seq_len, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
    audios = np.array(audios).reshape((gpu_num * bc, seq_len, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
    scores = np.array(scores).reshape((gpu_num * bc, seq_len))
    labels = np.array(labels).reshape((gpu_num * bc,))
    sample_poses = np.array(sample_poses).reshape((gpu_num * bc,))

    # check
    if np.sum(labels - np.array(batch_labels)) != 0:
        logging.info('Label Mismatch: %d' % step)
    return visuals, audios, scores, labels, sample_poses

def test_scheme_build(data_test,seq_len):
    # 与train_schem_build一致，但是不区分正负样本，也不做随机化
    seq_list = []
    test_vids = []
    for vid in data_test:
        label = data_test[vid]['labels']
        vlength = len(label)
        # 顺序将每个片段的邻域加入列表中，记录片段在序列中的位置以及片段标签
        for sample_pos in range(vlength):
            seq_start = sample_pos - int(seq_len / 2)
            seq_end = seq_start + seq_len
            seq_start = max(0, seq_start)  # 截断
            seq_list.append((vid, seq_start, seq_end, sample_pos, label[sample_pos]))
        test_vids.append(vid)  # 记录vid顺序用于evaluation
    return seq_list, test_vids

def get_batch_test(data,test_scheme,step,gpu_num,bc,seq_len):
    # 与get_batch_test一致，每次选择gpu_num*bc个序列返回，但是保持原有顺序
    seq_list = test_scheme

    # 生成batch_index与sample_pos
    batch_index = []
    sample_poses = []  # 取样点在序列中的相对位置
    batch_labels = []  # only for check
    for i in range(gpu_num * bc):  # 每次预测gpu_num*bc个片段
        position = (step * gpu_num * bc + i) % len(seq_list)  # 当前起始位置，经过最后一个视频末尾后折返，多余的序列作为padding
        # 读取样本
        vid,seq_start,seq_end,sample_pos,sample_label = seq_list[position]
        batch_index.append((vid,seq_start,seq_end,sample_pos))
        sample_poses.append(sample_pos - seq_start)
        batch_labels.append(sample_label)

    # 根据索引读取数据，并做padding
    visuals = []
    audios = []
    scores = []
    labels = []
    for i in range(len(batch_index)):
        vid,seq_start,seq_end,sample_pos = batch_index[i]
        vlength = len(data[vid]['labels'])
        seq_end = min(vlength,seq_end)  # 截断
        padding_len = seq_len - (seq_end - seq_start)
        visual = data[vid]['visual'][seq_start:seq_end]
        audio = data[vid]['audio'][seq_start:seq_end]
        score = data[vid]['scores'][seq_start:seq_end]
        if padding_len > 0:
            visual_pad = np.zeros((padding_len, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
            audio_pad = np.zeros((padding_len, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
            score_pad = np.zeros((padding_len,))
            visual = np.vstack((visual,visual_pad))  # 统一在后侧padding
            audio = np.vstack((audio,audio_pad))
            score = np.hstack((score, score_pad))
        visuals.append(visual)
        audios.append(audio)
        scores.append(score)
        labels.append(data[vid]['labels'][sample_pos])
    visuals = np.array(visuals).reshape((gpu_num * bc, seq_len, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
    audios = np.array(audios).reshape((gpu_num * bc, seq_len, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
    scores = np.array(scores).reshape((gpu_num * bc, seq_len))
    labels = np.array(labels).reshape((gpu_num * bc,))
    sample_poses = np.array(sample_poses).reshape((gpu_num * bc,))

    # check
    if np.sum(labels - np.array(batch_labels)) != 0:
        logging.info('Label Mismatch: %d' % step)
    return visuals, audios, scores, labels, sample_poses

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

def score_pred(visual,audio,score,sample_poses,visual_weights,visual_biases,audio_weights,audio_biases,drop_out,training):
    # audio convolution
    audio_feat = tf.reshape(audio,shape=(-1,A_HEIGHT,A_WIDTH,A_CHANN))  # 6b*8*8*128
    audio_conv5 = tf.nn.conv2d(audio_feat, audio_weights['wc5'], [1, 1, 1, 1], padding='SAME')
    audio_conv5 = tf.nn.relu(tf.nn.bias_add(audio_conv5, audio_biases['bc5']))
    audio_out = tf.nn.max_pool(audio_conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # 6b*4*4*256

    # visual convolution
    visual_conv5 = conv3d('conv5b', visual, visual_weights['wc5b'], visual_biases['bc5b'])
    visual_conv5 = tf.nn.relu(visual_conv5, 'relu5b')
    visual_pool5 = tf.nn.max_pool3d(visual_conv5,ksize=[1,2,2,2,1],strides=[1,2,2,2,1],padding='SAME')
    visual_pool6 = tf.transpose(visual_pool5, perm=(0, 4, 2, 3, 1))
    visual_pool6 = tf.nn.max_pool3d(visual_pool6, ksize=[1, 4, 1, 1, 1], strides=[1, 4, 1, 1, 1], padding='SAME')
    visual_pool6 = tf.transpose(visual_pool6, perm=(0, 4, 2, 3, 1))  # b*1*4*4*128
    visual_out = tf.squeeze(visual_pool6, axis=1)  # b*4*4*128

    # bilinear pooling
    A = tf.transpose(audio_out,perm=[0,3,1,2])  # 6b*256*4*4
    shape_A = A.get_shape().as_list()
    A = tf.reshape(A,shape=[-1,A_NUM*shape_A[1],shape_A[2]*shape_A[3]])  # b*1536*16
    B = visual_out
    shape_B = B.get_shape().as_list()
    B = tf.reshape(B,shape=[-1,shape_B[1]*shape_B[2],shape_B[3]])  # b*16*128
    I = tf.matmul(A,B)  # b*1536*128
    shape_I = I.get_shape().as_list()
    x = tf.reshape(I,shape=(-1,shape_I[1]*shape_I[2]))  # b*196608
    y = tf.multiply(tf.sign(x), tf.sqrt(tf.abs(x)))  # b*196608
    z = tf.nn.l2_normalize(y, dim=1)  # b*196608

    # self-attention
    # z形式为bc*seq_len个clip
    # 对encoder来说每个gpu上输入bc*seq_len*d，即每次输入bc个序列，每个序列长seq_len，每个元素维度为d
    # 在encoder中将输入的序列映射到合适的维度
    seq_input = tf.reshape(z,shape=(BATCH_SIZE,SEQ_LEN,-1))  # bc*seq_len*196608
    logits, attention_list = self_attention(seq_input, score, SEQ_LEN, NUM_BLOCKS,
                                            NUM_HEADS, drop_out, training)  # bc*seq_len
    # logits = tf.clip_by_value(tf.reshape(tf.sigmoid(logits), [-1, 1]), 1e-6, 0.999999)  # (bc*seq_len,1)

    target = tf.one_hot(indices=sample_poses,depth=logits.get_shape().as_list()[-1],on_value=1,off_value=0)
    target = tf.cast(target,dtype=tf.float32)
    logits = tf.reduce_sum(logits * target, axis=1)  # 只保留取样位置的值
    logits = tf.reshape(logits, [-1,1])

    return logits, [target]

def _loss(sp,sn,delta):
    zeros = tf.constant(0,tf.float32,shape=[sp.get_shape().as_list()[0],1])
    delta_tensor = tf.constant(delta,tf.float32,shape=[sp.get_shape().as_list()[0],1])
    u = 1 - sp + sn
    lp = tf.maximum(zeros,u)
    condition = tf.less(u,delta_tensor)
    v = tf.square(lp)*0.5
    w = lp*delta-delta*delta*0.5
    loss = tf.where(condition,x=v,y=w)
    return tf.reduce_mean(loss)

def tower_loss_huber(name_scope,preds,labels):
    # 每一组相邻的分段计算一次loss，取平均
    cij_list = []
    for i in range(BATCH_SIZE - 1):
        condition = tf.greater(labels[i],labels[i+1])
        sp = tf.where(condition,preds[i],preds[i+1])
        sn = tf.where(condition,preds[i+1],preds[i])
        cij = _loss(sp,sn,3)
        cij_list.append(cij)
    cost = cij_list[0]
    for i in range(1,len(cij_list)):
        cost = cost + cij_list[i]
    cost = cost / len(cij_list)
    weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))
    total_loss = cost + weight_decay_loss

    return tf.reduce_mean(total_loss)

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

def evaluation(pred_scores, data_test, test_ids, seq_len):
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
        preds = preds_c[pos:pos + vlength]
        preds = np.array(preds).reshape((-1,))
        pos += vlength
        # predict
        hlnum = int(np.sum(labels))
        preds_list = list(preds)
        preds_list.sort(reverse=True)
        threshold = preds_list[hlnum]
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
    logging.info('APRF: %.3f,%.3f,%.3f,%.3f,%d,%d' % (
                a, p, r, f, np.sum(label_true_all), np.sum(label_pred_all)))
    return a,p,r,f

def run_training(data_train, data_test, test_mode):
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    max_f1 = MAX_F1

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
        labels_holder = tf.placeholder(tf.float32,shape=(BATCH_SIZE * GPU_NUM,))
        sample_poses_holder = tf.placeholder(tf.int32,shape=(BATCH_SIZE * GPU_NUM,))
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
        # with tf.variable_scope('var_name_fusion') as var_name_fusion:
        #     fusion_weights = {
        #         'wd1': _variable_with_weight_decay('wd1', [32768, 1024], L2_LAMBDA),
        #         'wd2': _variable_with_weight_decay('wd2', [1024, 512], L2_LAMBDA),
        #         'wd3': _variable_with_weight_decay('wd3', [512, 256], L2_LAMBDA),
        #         'wd4': _variable_with_weight_decay('wd4', [256, 64], L2_LAMBDA),
        #         'wout': _variable_with_weight_decay('wout', [64, 1], L2_LAMBDA),
        #     }
        #     fusion_biases = {
        #         'bd1': _variable_with_weight_decay('bd1', [1024], 0.0000),
        #         'bd2': _variable_with_weight_decay('bd2', [512], 0.0000),
        #         'bd3': _variable_with_weight_decay('bd3', [256], 0.0000),
        #         'bd4': _variable_with_weight_decay('bd4', [64], 0.0000),
        #         'bout': _variable_with_weight_decay('bout', [1], 0.0000),
        #     }

        varlist_visual = list(weights.values()) + list(biases.values())
        varlist_audio = list(audio_weights.values()) + list(audio_biases.values())
        # training operations
        # lr = noam_scheme(LR_TRAIN,global_step,WARMUP_STEP)
        lr = tf.train.piecewise_constant(global_step,PHASES_STEPS,PHASES_LR)
        opt_train = tf.train.AdamOptimizer(lr)

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
                labels = labels_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,]
                scores = scores_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                sample_poses = sample_poses_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,]

                # predict scores
                # logits, atlist_one = score_pred(visual,audio,scores,weights,biases,audio_weights,audio_biases,
                #                     fusion_weights,fusion_biases,dropout_holder,training_holder)
                logits, atlist_one = score_pred(visual, audio, scores, sample_poses, weights, biases, audio_weights, audio_biases,
                                                dropout_holder, training_holder)
                logits_list.append(logits)
                attention_list += atlist_one  # 逐个拼接各个卡上的attention_list
                # calculate loss & gradients
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = tower_loss_huber(loss_name_scope, logits, labels)
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
            logging.info(' Ckpt Model Restoring: '+ckpt_model_path)
            saver_overall.restore(sess, ckpt_model_path)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        train_scheme = train_scheme_build_v3(data_train, SEQ_LEN)
        epoch_step = math.ceil(len(train_scheme[1]) / (BATCH_SIZE * GPU_NUM))  # 所有负样本都计算过一次作为一个epoch
        test_scheme, test_vids = test_scheme_build(data_test,SEQ_LEN)
        max_test_step = math.ceil(len(test_scheme) / BATCH_SIZE / GPU_NUM)

        # Beging training
        ob_loss = []
        timepoint = time.time()
        for step in range(MAXSTEPS):
            visual_b, audio_b, score_b, label_b, sample_pose_b = get_batch_train(data_train, train_scheme,
                                                                                 step,GPU_NUM,BATCH_SIZE,SEQ_LEN)
            observe = sess.run([train_op] + loss_list + logits_list + attention_list + [global_step, lr],
                               feed_dict={visual_holder: visual_b,
                                          audio_holder: audio_b,
                                          scores_holder: score_b,
                                          labels_holder: label_b,
                                          sample_poses_holder: sample_pose_b,
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
                    visual_b, audio_b, score_b, label_b, sample_pose_b = get_batch_test(data_test, test_scheme,
                                                                                       test_step, GPU_NUM, BATCH_SIZE, SEQ_LEN)
                    logits_temp_list = sess.run(logits_list, feed_dict={visual_holder: visual_b,
                                                                        audio_holder: audio_b,
                                                                        scores_holder: score_b,
                                                                        sample_poses_holder: sample_pose_b,
                                                                        training_holder: False,
                                                                        dropout_holder: 0})
                    for preds in logits_temp_list:
                        pred_scores.append(preds.reshape((-1)))
                a, p, r, f = evaluation(pred_scores, data_test, test_vids, SEQ_LEN)
                # logging.info('Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (a, p, r, f))

                if test_mode == 1:
                    return
                # save model
                if step > MIN_TRAIN_STEPS - PRESTEPS and f >= max_f1:
                    if f > max_f1:
                        max_f1 = f
                    model_path = model_save_dir + 'S%d-E%d-L%.6f-F%.3f' % (step,epoch,np.mean(loss_array),f)
                    saver_overall.save(sess, model_path)
                    logging.info('Model Saved: '+model_path+'\n')

            if step % 1000 == 0 and step > 0:
                model_path = model_save_dir + 'S%d-E%d' % (step+PRESTEPS, epoch)
                saver_overall.save(sess, model_path)
                logging.info('Model Saved: '+str(step + PRESTEPS))

            # saving final model
        model_path = model_save_dir + 'S%d' % (MAXSTEPS + PRESTEPS)
        saver_overall.save(sess, model_path)
        logging.info('Model Saved: '+str(MAXSTEPS + PRESTEPS))

    return

def main(self):
    label_record = load_label(LABEL_PATH)
    data_train, data_valid, data_test = load_data(label_record, FEATURE_BASE)
    logging.info('Data loaded !')

    logging.info('*'*20+'Settings'+'*'*20)
    logging.info('Model Dir: '+model_save_dir)
    logging.info('Training Phases: ' + str(PHASES_STEPS))
    logging.info('LR: '+str(PHASES_LR))
    logging.info('Label: '+str(LABEL_PATH))
    logging.info('Dropout Rate: '+str(DROP_OUT))
    logging.info('Sequence Length: '+str(SEQ_LEN))
    logging.info('*' * 50+'\n')

    run_training(data_train, data_test, 0)  # for training
    # run_training(data_train, data_train, 1)  # for testing

if __name__ == "__main__":
    tf.app.run()
