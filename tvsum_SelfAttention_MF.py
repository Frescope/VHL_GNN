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
    parser.add_argument('--gpu', default='',type=str)
    parser.add_argument('--dropout',default='0.1',type=float)
    if SERVER == 0:
        parser.add_argument('--msd', default='TVSum_SelfAttention', type=str)
    else:
        parser.add_argument('--msd', default='model_tvsum_SA', type=str)
hparams = Path()
parser = hparams.parser
hp = parser.parse_args()

if SERVER == 0:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu

# global paras
PRESTEPS = 0
MAXSTEPS = 36000
MIN_TRAIN_STEPS = 0
WARMUP_STEP = 4000
LR_TRAIN = 1e-7
HIDDEN_SIZE = 128  # for lstm
DROP_OUT = hp.dropout

EVL_EPOCHS = 1  # epochs for evaluation
L2_LAMBDA = 0.005  # weightdecay loss
GRAD_THRESHOLD = 10.0  # gradient threshold
MAX_F1 = 0.26

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

load_ckpt_model = False

if SERVER == 0:
    # path for JD server
    LABEL_PATH = r'/public/data0/users/hulinkang/tvsum/label_record.json'
    INFO_PATH = r'/public/data0/users/hulinkang/tvsum/video_info.json'
    FEATURE_BASE = r'/public/data0/users/hulinkang/tvsum/feature_intermid/'
    visual_model_path = '../model_HL/pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../model_HL/pretrained/MINMSE_0.019'
    model_save_dir = r'/public/data0/users/hulinkang/model_HL/'+hp.msd+'/'
    ckpt_model_path = '../model_HL/TVSum_SelfAttention_0/STEP_30000'
    # ckpt_model_path = '../model_HL/SelfAttention_1/MAXF1_0.286_0'

else:
    # path for USTC server
    LABEL_PATH = '//data//linkang//tvsum50//label_record.json'
    INFO_PATH = '//data/linkang//tvsum50//video_info.json'
    FEATURE_BASE = '//data//linkang//tvsum50//feature_intermid//'
    visual_model_path = '../../model_HL/mosi_pretrained/sports1m_finetuning_ucf101.model'
    audio_model_path = '../../model_HL_v2/mosi_pretrained/MINMSE_0.019'
    model_save_dir = r'/data/linkang/model_HL_v3/'+hp.msd+'/'
    # ckpt_model_path = '../../model_HL_v3/model_bilibili_SA_2/STEP_9000'
    ckpt_model_path = '../../model_HL_v3/model_tvsum_SA_0/STEP_27000'

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
    return label_record, video_info

def load_data(label_record,feature_base):
    # 只加载数据



def split_data(label_record,video_info,data)
