# SelfAttention_MF_v3的check版，同时检验多个模型，检查ext_ratio

import os
import time
import numpy as np
import tensorflow as tf
import math
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import random
import logging
import copy
import Transformer
from Transformer import self_attention

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# global paras
PRESTEPS = 0
MAXSTEPS = 32000
MIN_TRAIN_STEPS = 0
WARMUP_STEP = 4000
LR_TRAIN = 2e-7
HIDDEN_SIZE = 128  # for lstm

EVL_EPOCHS = 1  # epochs for evaluation
L2_LAMBDA = 0.005  # weightdecay loss
GRAD_THRESHOLD = 10.0  # gradient threshold
MAX_F1 = 0.33

GPU_NUM = 1
BATCH_SIZE = 4
SEQ_INTERVAL = 1

D_MODEL = Transformer.D_MODEL
SEQ_LEN = Transformer.SEQ_LEN

V_NUM = 2  # 3D卷积的最高一维
V_HEIGHT = 7
V_WIDTH = 7
V_CHANN = 512

A_NUM = 6  # 一个clip的A_NUM个spectro，运算时需要并入batch，保持2D卷积操作的3D输入张量
A_HEIGHT = 8
A_WIDTH = 8
A_CHANN = 128

load_ckpt_model = True
SERVER = 0

if SERVER == 0:
    # path for JD server
    LABEL_PATH = r'/public/data0/users/hulinkang/bilibili/label_record_zmn_24s.json'
    FEATURE_BASE = r'/public/data0/users/hulinkang/bilibili/feature/'
    visual_model_path = '../model_HL/pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../model_HL/pretrained/MINMSE_0.019'
    model_save_dir = r'/public/data0/users/hulinkang/model_HL/SelfAttention_0/'
    ckpt_model_path = '../model_HL/SelfAttention_3/STEP_30000'
    # ckpt_model_path = '../model_HL/SelfAttention_1/MAXF1_0.286_0'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    # path for USTC server
    LABEL_PATH = '//data//linkang//bilibili//label_record_zmn_24s.json'
    FEATURE_BASE = '//data//linkang//bilibili//feature//'
    visual_model_path = '../../model_HL/mosi_pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../../model_HL_v2/mosi_pretrained/MINMSE_0.019'
    model_save_dir = '//data//linkang//model_HL_v3//model_bilibili_SA_3//'
    # ckpt_model_path = '../../model_HL_v3/model_bilibili_SA_2/STEP_9000'
    ckpt_model_path = '../../model_HL_v3/model_bilibili_SA_2/MAXF1_0.329_0'

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

        # split train & valid & test set
        valid_pos = round(len(labels) * 0.6)
        test_pos = round(len(labels) * 0.8)

        temp_train = {}
        temp_train['visual'] = visual[:valid_pos]
        temp_train['audio'] = audio[:valid_pos]
        temp_train['labels'] = labels[:valid_pos]
        temp_train['scores'] = scores[:valid_pos]
        data_train[vid] = temp_train

        temp_valid = {}
        temp_valid['visual'] = visual[valid_pos:test_pos]
        temp_valid['audio'] = audio[valid_pos:test_pos]
        temp_valid['labels'] = labels[valid_pos:test_pos]
        temp_valid['scores'] = scores[valid_pos:test_pos]
        data_valid[vid] = temp_valid

        temp_test = {}
        temp_test['visual'] = visual[test_pos:]
        temp_test['audio'] = audio[test_pos:]
        temp_test['labels'] = labels[test_pos:][:len(temp_test['visual'])]  # 截断
        temp_test['scores'] = scores[test_pos:][:len(temp_test['visual'])]  # 截断
        data_test[vid] = temp_test

        logging.info('Data(train, valid, test): '+str(temp_train['visual'].shape)+str(temp_valid['audio'].shape)+str(temp_test['labels'].shape))
        logging.info('Scores(train, valid, test): '+str(len(temp_train['scores']))+str(len(temp_valid['scores']))+str(len(temp_test['scores'])))

    return data_train, data_valid, data_test

def train_scheme_build(data_train,seq_len,interval):
    # 加强随机化的train_scheme，直接根据step确定当前使用哪个序列，找到对应的视频中的对应位置即可
    # train_scheme = [(vid,seq_start,seq_label)]
    PN_THRESH = 2  # 正负样本分界
    seq_list = []
    for vid in data_train:
        pos_list = []
        neg_list = []
        label = data_train[vid]['labels']
        vlength = len(label)
        # 对每个视频遍历所有将提取的序列，根据门限确定正负样本，归入相应的列表
        seq_start = 0
        while seq_start + seq_len <= vlength:
            seq_label = label[seq_start:seq_start+seq_len]
            if np.sum(seq_label) >= PN_THRESH:
                pos_list.append((vid,seq_start,seq_label))
            else:
                neg_list.append((vid,seq_start,seq_label))
            seq_start += interval

        # 合并pos_list & neg_list，穿插排列
        k = 0
        while k < min(len(pos_list),len(neg_list)):
            seq_list.append(pos_list[k])
            seq_list.append(neg_list[k])
            k += 1
        seq_list = seq_list + pos_list[k:] + neg_list[k:]

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

def test_record_build(data_test, seq_len, interval):
    # 采用与训练时相同的方法构建测试计划，同时对每个序列附加一个记录，用于记录对序列的预测结果
    # 需要padding，保证一个视频的最后几个片段在interval较大时也能加入序列中，根据序列标签的长度在getbatch时判断padding长度
    # test_record = [scheme,record]
    scheme = []
    record = []
    for vid in data_test:
        label = data_test[vid]['labels']
        vlength = len(label)
        seq_start = 0
        seq_num = math.ceil(vlength / interval)  # 视频中的序列数量
        padding_len = math.ceil((vlength-seq_len) / interval) * interval + seq_len - vlength # 保证末尾片段至少预测一次
        while seq_start + seq_len <= vlength + padding_len:
            seq_label = label[seq_start:seq_start + seq_len]  # 末尾可能会不足
            scheme.append((vid, seq_start, seq_label))
            record.append((vid, seq_start, len(seq_label) * [0]))
            seq_start += interval
    return [scheme, record]

def get_batch_test_v2(data, test_record, step, gpu_num, bc, seq_len):
    # 每次按照test_record读取gpu_num*bc个序列，根据seq_label的长度判断padding的长度
    scheme = test_record[0]
    seq_num = len(scheme)  # 序列总数
    # 每个step中提取gpu_num*bc个序列，顺序拼接后返回
    visual = []
    audio = []
    score = []
    for i in range(gpu_num * bc):
        position = step * gpu_num * bc + i
        if position < seq_num:
            vid, start, label_seq_origin = scheme[position]
            end = start + seq_len
            visual_seq = data[vid]['visual'][start:end]
            audio_seq = data[vid]['audio'][start:end]
            score_seq = data[vid]['scores'][start:end]
            label_seq = data[vid]['labels'][start:end]
            if len(label_seq) < seq_len:
                # 视频末尾，需要对序列做padding
                padding_len = seq_len - len(label_seq)
                visual_seq = np.vstack((visual_seq, np.zeros_like(visual_seq)[:padding_len]))
                audio_seq = np.vstack((audio_seq, np.zeros_like(audio_seq)[:padding_len]))
                score_seq = np.hstack((score_seq, np.zeros((padding_len))))
            if np.sum(label_seq - label_seq_origin) > 0:
                logging.info('\n\nPos Error!',step,position,vid,i,label_seq,label_seq_origin,'\n\n')
        else:
            # 序列不足，需要对batch做padding
            vid = scheme[0][0]
            visual_seq = np.zeros_like(data[vid]['visual'][0:seq_len])
            audio_seq = np.zeros_like(data[vid]['audio'][0:seq_len])
            score_seq = np.zeros_like(data[vid]['scores'][0:seq_len])
        visual.append(visual_seq)
        audio.append(audio_seq)
        score.append(score_seq)
    visual = np.array(visual).reshape((gpu_num * bc, seq_len, V_NUM, V_HEIGHT, V_WIDTH, V_CHANN))
    audio = np.array(audio).reshape((gpu_num * bc, seq_len, A_NUM, A_HEIGHT, A_WIDTH, A_CHANN))
    score = np.array(score).reshape((gpu_num * bc, seq_len))

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
               attention_weights,attention_biases,drop_out,training):
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
    logits = self_attention(seq_input, score, attention_weights, attention_biases, drop_out, training)  # bc*seq_len
    logits = tf.clip_by_value(tf.reshape(tf.sigmoid(logits), [-1, 1]), 1e-8, 0.99999999)  # (bc*seq_len,1)

    return logits

def tower_loss(name_scope,logits,labels):
    y = tf.reshape(labels,[-1,1])
    # ce = -y * (tf.log(logits)) * (1-logits) ** 2.0 *0.25 - (1 - y) * tf.log(1 - logits) * (logits) ** 2.0 * 0.75
    # loss = tf.reduce_sum(ce)
    ce = -y * (tf.log(logits)) - (1 - y) * tf.log(1 - logits)
    loss = tf.reduce_mean(ce)
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

def evaluation_v2(pred_scores, data, test_record, bc,seq_len):
    # 输入一个列表，每个元素是一个卡上一个batch的全部结果，填充test_record
    # 根据vid将record中的结果填充到对应视频的对应片段上，计算每个片段的预测值平均分，求结果
    # reshape
    predictions = np.zeros((0,seq_len))
    seq_num = len(test_record[0])
    for i in range(len(pred_scores)):
        predictions = np.vstack((predictions,pred_scores[i].reshape((bc,seq_len))))  # predictions:(seq_num,seq_len)
    # recording
    record = test_record[1]
    for i in range(len(record)):  # 舍弃predictions中剩余的padding序列
        vid,seq_start,seq_record = record[i]
        seq_record = list(predictions[i][0:len(seq_record)])  # 舍弃视频末尾的对序列的padding
        record[i] = (vid,seq_start,seq_record)
    # 记录每个视频每个片段的所有得分
    clip_scores = {}
    for vid in data:
        vlength = len(data[vid]['labels'])
        temp = []
        for i in range(vlength):
            temp.append([])  # 为每个分段生成一个空列表
        clip_scores[vid] = temp
    for seq_pred in record:
        vid,seq_start,seq_record = seq_pred
        for i in range(len(seq_record)):
            position = seq_start + i
            clip_scores[vid][position].append(seq_record[i])
    # 计算片段平均得分
    label_pred_all = np.array(())
    label_true_all = np.array(())
    for vid in clip_scores:
        # 计算均值
        for i in range(len(clip_scores[vid])):
            preds_list = clip_scores[vid][i]
            preds_avg = np.mean(np.array(preds_list))
            clip_scores[vid][i] = preds_avg
        labels = data[vid]['labels'].reshape((-1))
        # predict label
        hlnum = int(np.sum(labels))
        preds_sort = copy.deepcopy(clip_scores[vid])
        preds_sort.sort(reverse=True)
        threshold = preds_sort[hlnum]
        ext_ratio = 1.02
        if threshold * ext_ratio <= preds_sort[0]:
            threshold = threshold * ext_ratio
        labels_pred = (np.array(clip_scores[vid]) > threshold).astype(int)
        label_pred_all = np.concatenate((label_pred_all,labels_pred))
        label_true_all = np.concatenate((label_true_all,labels))
    label_pred_all = label_pred_all.reshape((-1))
    label_true_all = label_true_all.reshape((-1))
    # 求指标
    a = accuracy_score(label_true_all, label_pred_all)
    p = precision_score(label_true_all, label_pred_all)
    r = recall_score(label_true_all, label_pred_all)
    f = f1_score(label_true_all, label_pred_all)

    return a, p, r, f

def model_search(model_save_dir):
    # 找到要验证的模型名称
    model_to_restore = []
    for root,dirs,files in os.walk(model_save_dir):
        for file in files:
            if file.startswith('MINMSE'):
                model_name = 'MINMSE_0.' + file.split('.')[1]
                model_to_restore.append(os.path.join(root, model_name))
            if file.startswith('MAXF1'):
                model_name = 'MAXF1_0.' + file.split('.')[1]
                model_to_restore.append(os.path.join(root, model_name))
            if file.startswith('STEP'):
                model_name = file.split('.')[0]
                model_to_restore.append(os.path.join(root,model_name))
    model_to_restore = list(set(model_to_restore))
    model_to_restore.sort()
    return model_to_restore

def run_training(data_train, data_test, model_path, test_mode):
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
        opt_train = tf.train.AdamOptimizer(lr)

        # graph building
        tower_grads_train = []
        logits_list = []
        loss_list = []
        for gpu_index in range(GPU_NUM):
            with tf.device('/gpu:%d' % gpu_index):
                visual = visual_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :, :, :]
                visual = tf.reshape(visual,shape=(BATCH_SIZE*SEQ_LEN,V_NUM,V_HEIGHT,V_WIDTH,V_CHANN))
                audio = audio_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :, :, :, :, :]
                audio = tf.reshape(audio,shape=(BATCH_SIZE*SEQ_LEN,A_NUM,A_HEIGHT,A_WIDTH,A_CHANN))
                labels = labels_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]
                scores = scores_holder[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE, :]

                # predict scores
                logits = score_pred(visual,audio,scores,weights,biases,audio_weights,audio_biases,
                                    None,None,dropout_holder,training_holder)
                logits_list.append(logits)

                # calculate loss & gradients
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = tower_loss(loss_name_scope, logits, labels)
                varlist = tf.trainable_variables()  # 全部训练
                varlist = list(set(varlist) - set(varlist_visual) - set(varlist_audio))  # 只训练Self-attention参数
                # varlist = varlist + list(biases.values()) + list(audio_biases.values())  # 加上卷积层偏置
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
        saver_overall = tf.train.Saver()
        if load_ckpt_model:
            logging.info(' Ckpt Model Restoring: '+model_path)
            saver_overall.restore(sess, model_path)
            logging.info(' Ckpt Model Resrtored !')

        # train & test preparation
        test_record = test_record_build(data_test, SEQ_LEN, SEQ_INTERVAL)
        max_test_step = math.ceil(len(test_record[0]) / BATCH_SIZE / GPU_NUM)
        train_scheme = train_scheme_build(data_train,SEQ_LEN,SEQ_INTERVAL)
        epoch_step = math.ceil(len(train_scheme) / BATCH_SIZE / GPU_NUM)
        # epoch_step = math.ceil(len(train_scheme[0]) / (BATCH_SIZE*GPU_NUM-1))

        # Beging training
        ob_loss = []
        timepoint = time.time()
        for step in range(MAXSTEPS):
            visual_b, audio_b, score_b, label_b = get_batch_train(data_train, train_scheme, step,GPU_NUM,BATCH_SIZE,SEQ_LEN)
            observe = sess.run([train_op] + loss_list + logits_list + [global_step, lr],
                               feed_dict={visual_holder: visual_b,
                                          audio_holder: audio_b,
                                          scores_holder: score_b,
                                          labels_holder: label_b,
                                          dropout_holder: 0.1,
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
                # logging.info(' Step %d: %.3f sec' % (step, duration))
                # logging.info(' Evaluate: '+str(step)+' Epoch: '+str(epoch))
                # logging.info(' Average Loss: '+str(np.mean(loss_array))+' Min Loss: '+str(np.min(loss_array))+' Max Loss: '+str(np.max(loss_array)))

                # 按顺序预测测试集中每个视频的每个分段，全部预测后在每个视频内部排序，计算指标
                pred_scores = []  # 每个batch输出的预测得分
                for test_step in range(max_test_step):
                    visual_b, audio_b, score_b = get_batch_test_v2(data_test, test_record, test_step,
                                                                   GPU_NUM, BATCH_SIZE, SEQ_LEN)
                    logits_temp_list = sess.run(logits_list, feed_dict={visual_holder: visual_b,
                                                                        audio_holder: audio_b,
                                                                        scores_holder: score_b,
                                                                        training_holder: False,
                                                                        dropout_holder: 0})
                    for preds in logits_temp_list:
                        pred_scores.append(preds.reshape((-1)))  # 每个添加gpu_num个bc*seq_len长度的序列
                a, p, r, f = evaluation_v2(pred_scores, data_test, test_record, BATCH_SIZE, SEQ_LEN)
                logging.info('Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (a, p, r, f))
                return
    return

def main(self):
    label_record = load_label(LABEL_PATH)
    data_train, data_valid, data_test = load_data(label_record, FEATURE_BASE)
    print('Data loaded !')

    models_to_restore = model_search(model_save_dir)
    for i in range(len(models_to_restore)):
        print('-' * 20, i, models_to_restore[i].split('/')[-1], '-' * 20)
        ckpt_model_path = models_to_restore[i]
        run_training(data_train, ckpt_model_path,1)  # for testing

if __name__ == "__main__":
    tf.app.run()


