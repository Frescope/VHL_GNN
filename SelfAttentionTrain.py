# 相对于Transformer模型调整计算图构建方式即可，数据集构建等其他模块相同

import os
import time
import numpy as np
import tensorflow as tf
import math
import json
import logging
from tqdm import tqdm
from hparams import Hparams_selfattn
from SelfAttentionModel import Self_attention
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

logging.basicConfig(level=logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

hparams = Hparams_selfattn()
parser = hparams.parser
hp = parser.parse_args()

def load_label():
    file = open(hp.label_path,'r')
    label_record = json.load(file)
    file.close()
    return label_record

def load_data(label_record):
    # 装载所有特征，划分数据
    vids = list(label_record.keys())
    data_train = {}
    data_valid = {}
    data_test = {}
    for i in range(len(vids)):
        vid = vids[i]
        print('-'*20,i,vid,'-'*20)
        feature_path = hp.feature_path + vid + r'/features_ovr.npy'
        feature_ovr = np.load(feature_path).reshape((-1,hp.d_model))
        labels = np.array(label_record[vid]['label']).reshape((-1))
        scores = np.array(label_record[vid]['score']).reshape((-1))

        valid_pos = round(len(labels) * 0.6)
        test_pos = round(len(labels) * 0.8)

        temp_train = {}
        temp_train['features'] = feature_ovr[:valid_pos]
        temp_train['labels'] = labels[:valid_pos]
        temp_train['scores'] = scores[:valid_pos]
        data_train[vid] = temp_train
        temp_valid = {}
        temp_valid['features'] = feature_ovr[valid_pos:test_pos]
        temp_valid['labels'] = labels[valid_pos:test_pos]
        temp_valid['scores'] = scores[valid_pos:test_pos]
        data_valid[vid] = temp_valid
        temp_test = {}
        temp_test['features'] = feature_ovr[test_pos:]
        temp_test['labels'] = labels[test_pos:]
        temp_test['scores'] = scores[test_pos:]
        data_test[vid] = temp_test

        print('Data(train, valid, test): ',temp_train['features'].shape,
              temp_valid['features'].shape, temp_test['features'].shape)
        print('Scores(train, valid, test): ', len(temp_train['scores']),
              len(temp_valid['scores']),len(temp_test['scores']))

    return data_train,data_valid,data_test

def generator_fn(features,labels,scores):
    # yields: (x),(decoder_input,y)
    # 考虑到二元分类问题与原本的seq2seq问题不同，使用score作为decoder的输入，进行后续的嵌入
    sample_num = len(features)
    for i in range(sample_num):
        x = features[i]
        decoder_input = scores[i]
        y = labels[i]
        yield (x), (decoder_input, y)

def input_fn(features,labels,scores,shuffle=False):
    # input：feature(?,SEQ_LEN,512),label(?,SEQ_LEN),score(?,SEQ_LEN)
    # output: xs( x(?,SEQ_LEN,512) ),
    #         ys( decoder_input(?,SEQ_LEN),y(?,SEQ_LEN) )
    shapes = (([None,None]),
              ([None],[None]))
    types = ((tf.float32),
             (tf.float32,tf.float32))
    paddings = ((0.0),
                (0.0,0.0))
    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(features,labels,scores))
    if shuffle:
        dataset = dataset.shuffle(16*hp.batch_size)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(hp.batch_size,shapes,paddings).prefetch(1)

    return dataset

def get_batch_train(data_train):
    # 用于构建训练集，包含重叠的片段序列
    # 读入的数据与标签首先按照vid组织，从每个视频的分段特征中选择若干个连续的片段序列作为这一视频的若干样例，样例之间可以重叠
    # 即，模型输入的各个batch是在每次模型运行之前就构建好的，只是输入模型的顺序可能有所不同，构建过程中注意保持标签和特征的对应关系
    # 生成的训练集中将来自各个视频的序列都拼接在一次，只保持同一序列中的片段来自同一个视频即可，使用时用生成器提取batch即可

    # 每个视频的片段抽取若干组连续的分段作为样例，从第一个片段开始，以固定长度、固定步长提取
    vids = list(data_train.keys())
    samples_features = []
    samples_labels = []
    samples_scores = []
    for i in range(len(vids)):
        vid = vids[i]
        
        # for gradient check
        # vid = vids[0]

        features = data_train[vid]['features']  # (?,512)
        labels = data_train[vid]['labels']  # (?,)
        scores = data_train[vid]['scores']  # (?,)
        clip_num = len(features)
        pos = 0
        cnt = 0
        while pos + hp.seq_len <= clip_num:
            samples_features.append(features[pos:pos+hp.seq_len])
            samples_labels.append(labels[pos:pos+hp.seq_len])
            samples_scores.append(scores[pos:pos+hp.seq_len])
            
            # for gradient check
            # samples_features.append(features[0:hp.seq_len])
            # samples_labels.append(labels[0:hp.seq_len])
            # samples_scores.append(scores[0:hp.seq_len])

            cnt += 1
            pos += hp.seq_step
        print('Video: ',vid,'Sample Number: ',cnt)
    samples_features = np.array(samples_features)
    samples_labels = np.array(samples_labels)
    samples_scores = np.array(samples_scores)
    print('Training data(Feature, Label & Score): ',samples_features.shape,samples_labels.shape,samples_scores.shape)

    # 从训练集中获取batch
    batches = input_fn(samples_features,samples_labels,samples_scores,True)
    batch_num = len(samples_features) // hp.batch_size + int(len(samples_features) % hp.batch_size != 0)
    return batches, batch_num, len(samples_features)

def get_batch_eval(data_eval):
    # 用于构建评估数据集，可以是训练、验证或测试集
    # 将属于同一个视频的分段划分为多个batch，不足的部分填充，保证同一序列中分段来自同一视频；然后将不同视频的batch拼接
    # 依次对每个分段进行预测，预测结果按照label长度拆分并还原，计算准确率即可（在后处理中完成）
    vids = list(data_eval.keys())
    samples_features = []
    samples_labels = []
    samples_scores = []
    print('-'*30,'Evaluation Data','-'*30)
    for i in range(len(vids)):
        vid = vids[i]
        # vid = vids[0]
        features = data_eval[vid]['features']  # (?,512)
        labels = data_eval[vid]['labels']  # (?,)
        scores = data_eval[vid]['scores']  # (?,)
        clip_num = len(features)
        seq_num = clip_num // hp.seq_len + int(clip_num % hp.seq_len != 0)
        # padding for sequence
        padding_len = seq_num * hp.seq_len - clip_num
        features_pad = np.zeros((padding_len, hp.d_model))  # (padlen,512)
        labels_pad = np.zeros((padding_len))  # (padlen,)
        scores_pad = np.zeros((padding_len))  # (padlen,)
        features= np.vstack((features,features_pad))
        labels = np.hstack((labels,labels_pad))
        scores = np.hstack((scores,scores_pad))
        # split in squence
        pos = 0
        for j in range(seq_num):
            samples_features.append(features[pos:pos + hp.seq_len])
            samples_labels.append(labels[pos:pos + hp.seq_len])
            samples_scores.append(scores[pos:pos + hp.seq_len])
            pos += hp.seq_len
        print('Video: ',vid,'Clip Number: ',clip_num, 'Sequence Number: ',seq_num)
    samples_features = np.array(samples_features)
    samples_labels = np.array(samples_labels)
    samples_scores = np.array(samples_scores)
    print('Evaluation Data(Feature, Label & Score): ',samples_features.shape,samples_labels.shape,samples_scores.shape)

    # 获取batch
    batches = input_fn(samples_features, samples_labels, samples_scores, False)
    batch_num = len(samples_features) // hp.batch_size + int(len(samples_features) % hp.batch_size != 0)
    return batches, batch_num, len(samples_features), vids

def evaluation(preds_list, data_eval, eval_ids):
    # 输入所有分段的logits以及标签，首先根据标签长度划分到不同的视频，然后在视频内部筛选highlight，计算结果
    pos = 0
    label_pred_all = np.array(())
    label_true_all = np.array(())
    for i in range(len(eval_ids)):
        # get preds for this video
        vid = eval_ids[i]
        # vid = eval_ids[0]
        labels_video = data_eval[vid]['labels']
        clip_num = len(labels_video)
        seq_num = clip_num // hp.seq_len + int(clip_num % hp.seq_len != 0)  # padding for sequence

        preds_video = np.array(preds_list[pos:pos+clip_num])
        # print(vid, pos, clip_num, seq_num)
        pos += seq_num * hp.seq_len

        # pred highlight
        hlnum = int(np.sum(labels_video))
        preds_video_list = list(preds_video)
        preds_video_list.sort(reverse=True)
        threshold = preds_video_list[hlnum]*1.02
        labels_pred = (preds_video > threshold).astype(int)
        label_true_all = np.concatenate((label_true_all,labels_video))
        label_pred_all = np.concatenate((label_pred_all,labels_pred))
        # print('Labels: ',labels_video)
        # print('Preds:  ',labels_pred)
        # print(len(labels_video),len(preds_video),len(label_true_all),len(label_pred_all))

    a = accuracy_score(label_true_all, label_pred_all)
    p = precision_score(label_true_all, label_pred_all)
    r = recall_score(label_true_all, label_pred_all)
    f = f1_score(label_true_all, label_pred_all)
    return a,p,r,f

logging.info("# Prepare training batches")

# load data
label_record = load_label()
data_train, data_valid, data_test = load_data(label_record)
data_eval = data_train
train_batches, num_train_batches, num_train_samples = get_batch_train(data_train)
eval_batches, num_eval_batches, num_eval_samples, eval_ids = get_batch_eval(data_eval)

print('Train Batches: ',train_batches)
print('Number of Train Batches: ',num_train_batches,'Number of Train Samples: ',num_train_samples)
print('Evaluation Batches: ',eval_batches)
print('Number of Evaluation Batches: ',num_eval_batches,'Number of Evaluation Samples: ',num_eval_samples)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

# build the graph
logging.info("# Load model")
m = Self_attention(hp)
train_vars, train_grads, train_logits, loss, train_op, global_step = m.train(xs,ys)
eval_logits = m.eval(xs,ys)

# config session
logging.info("# Session")
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
saver = tf.train.Saver(max_to_keep=hp.ckpt_num)
with tf.Session(config=config) as sess:
    # load checkpoint:
    ckpt = tf.train.latest_checkpoint(hp.model_save_dir)
    # ckpt = os.path.join(hp.model_save_dir, "E1033L0.057F10.238-136356")
    if ckpt is None or hp.load_ckpt == False:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        # save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)
        logging.info("Ckpt Loaded: {}".format(ckpt))

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)

    # var & gradient check
    # variable_names = [v.name for v in tf.trainable_variables()]
    # graph = tf.get_default_graph()
    v_names = ['Q_w', 'Q_b', 'K_w', 'K_b', 'V_w', 'V_b', 'ln_beta', 'ln_gamma', 
            'ff_0_w', 'ff_0_b','ff_1_w','ff_1_b','ff_ln_beta','ff_ln_gamma',
            'mlp_0_w', 'mlp_0_b', 'mlp_1_w', 'mlp_1_b', 'mlp_2_w', 'mlp_2_b']
    # grads = []
    # for i in range(len(variable_names)):
    #     var = graph.get_tensor_by_name(variable_names[i])
    #     grad = tf.gradients(loss,var)
    #     grads.append(grad) 

    epoch_loss = []
    for i in tqdm(range(_gs, total_steps+1)):
        vars_observe, grads_observe, batch_loss, _, _gs = sess.run([train_vars, train_grads, loss, train_op, global_step])
        epoch_loss.append(batch_loss)

        epoch = math.ceil(_gs / num_train_batches)

        # # gradient check
        # logging.info("\nStep: {} Loss: {}".format(_gs, batch_loss))
        # # logging.info(len(vars_observe),len(grads_observe))
        # for i in range(len(grads_observe)):
            
        #     # grad = [g for (g,v) in grads_observe[i]]
        #     grad = grads_observe[i]
        #     var = vars_observe[i]
        #     logging.info("{} (Mean, Min, Max):\t{:.6f}\t{:.6f}\t{:.6f}".format(
        #         v_names[i],np.mean(var),np.min(var),np.max(var)))
        #     logging.info("Gradient (Mean, Min, Max):\t{:.6f}\t{:.6f}\t{:.6f}\n".format(
        #         np.mean(grad),np.min(grad),np.max(grad)))

        if _gs and _gs % num_train_batches == 0:
            # evaluation
            _ = sess.run(eval_init_op)
            preds_list = []
            for eval_step in range(num_eval_batches):
                preds = sess.run(eval_logits)  # (bc,seq_len)
                preds_list.extend(preds.reshape((-1)).tolist())

            # temp = np.array(preds_list)
            # logging.info('Preds(Mean, Min, Max): {:.10f} {:.10f} {:.10f}'.format(
            #     np.mean(temp),np.min(temp),np.max(temp)))
            a,p,r,f = evaluation(preds_list, data_eval, eval_ids)
            logging.info("\nAPRF: %.3f  %.3f  %.3f  %.3f"%(a,p,r,f))

            logging.info('Last Batch Loss: %.3f' % batch_loss)
            logging.info('Epoch Loss: %.3f' % np.mean(np.array(epoch_loss)))

            model_output = "E%04dL%.3fF1%.3f" % (epoch, np.mean(np.array(epoch_loss)), f)
            epoch_loss.clear()

            # logging.info("# save models")
            ckpt_name = os.path.join(hp.model_save_dir, model_output)
            if epoch > hp.ckpt_epoch:
                saver.save(sess, ckpt_name, global_step=_gs)
                logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
            # os._exit(0)

logging.info("Done")

