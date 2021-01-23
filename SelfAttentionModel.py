# Self-Attention模型，在提取特征之后使用Encoder模块代替MLP，省去Decoder部分
#

import tensorflow as tf
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def positional_encoding(inputs, hp, scope='positional_encoding'):
    E = hp.d_model
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(hp.seq_len)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        # 没有padding，不设mask
        return tf.to_float(outputs)

def ln(inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type="key")

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type="future")

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"
    """
    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1])  # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    elif type in ("f", "future", "right"):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(queries, d_model, use_bias=True, name='Q_dense')  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True, name='K_dense')  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True, name='V_dense')  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = ln(outputs)

    return outputs

def ff(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu, name='Inner_dense')

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1], name='Outer_dense')

        # Residual connection
        outputs += inputs

        # Normalize
        # outputs = ln(outputs)

    return outputs

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

class Self_attention:
    def __init__(self,hp):
        self.hp = hp

    def encode(self, xs, training=True):
        # return: memory(?,seq_len,d_model)

        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            x = xs  # (bc,seq_len,d_model)

            # src_masks
            src_masks = tf.math.equal(tf.reduce_sum(x, axis=2), 0)  # 先降到2维，然后判断是否是padding
            # src_masks = tf.constant(True,dtype=tf.bool,shape=(self.hp.batch_size,self.hp.seq_len))

            # embedding，encoder中直接使用嵌入的向量，没有嵌入步骤
            enc = x
            enc *= self.hp.d_model ** 0.5  # scale
            feat_ob1 = enc
            enc += positional_encoding(enc, self.hp)
            feat_ob2 = enc
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)
            feat_ob3 = enc

            # blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])

        memory = enc
        feat_ob4 = enc
        return memory, [feat_ob1, feat_ob2, feat_ob3, feat_ob4]

    def mlp(self,memory, scope="final_mlp"):
        # input: memroy(?,seq_len,d_model)
        # output: logits(?,seq_len)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(memory, self.hp.d_ff, activation=tf.nn.relu, name='Inner_dense')
            feat_ob1 = logits
            logits = tf.layers.dense(logits, self.hp.d_model, activation=tf.nn.relu, name='Hidden_dense')
            feat_ob2 = logits
            logits = tf.layers.dense(logits, 1, activation=tf.nn.relu, name='Outer_dense')
            feat_ob3 = logits
            logits = tf.squeeze(logits)
        return logits, [feat_ob1, feat_ob2, feat_ob3]

    def train(self, xs, ys):
        # input: xs: x(bc,seq_len,d_model)
        #        ys: scores(bc,seq_len), labels(bc,seq_len)

        memory, enc_feat_obs = self.encode(xs)
        # memory=  xs
        # memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], 512])
        logits, mlp_feat_obs = self.mlp(memory)
        _,y = ys
        logits = tf.clip_by_value(tf.reshape(tf.sigmoid(logits),[-1,1]),5e-8,0.99999995)
        y = tf.reshape(y, [-1,1])
        # ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y)
        ce = -y * (tf.log(logits)) - (1-y)*tf.log(1-logits)
        loss = tf.reduce_mean(ce)

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(self.hp.lr)
        
        # train_op = optimizer.minimize(loss, global_step=global_step)
        
        varlist = tf.trainable_variables()
        gradient = optimizer.compute_gradients(loss, varlist)
        # gradient_clip = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gradient]
        # train_op = optimizer.apply_gradients(gradient, global_step=global_step)
        train_op = optimizer.minimize(loss,global_step=global_step)
        
        return enc_feat_obs, mlp_feat_obs, varlist, gradient, logits, loss, train_op, global_step

    def eval(self, xs, ys):
        # input: xs: x(bc,seq_len,d_model)
        #        ys: scores(bc,seq_len), labels(bc,seq_len)
        memory, enc_feat_obs = self.encode(xs)
        # memory=  xs
        # memory = tf.reshape(memory, [tf.shape(memory)[0], tf.shape(memory)[1], 512])
        logits, mlp_feat_obs = self.mlp(memory)
        logits = tf.clip_by_value(tf.reshape(tf.sigmoid(logits),[-1,1]),1e-8,0.99999999)
        return logits