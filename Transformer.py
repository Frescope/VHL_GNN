# transformer组件
import tensorflow as tf
import numpy as np

D_MODEL = 512
D_FF = 2048
SEQ_LEN = 20
INTERVAL = 1
NUM_BLOCKS = 2
NUM_HEADS = 32

def positional_encoding(inputs, scope='positional_encoding'):
    E = D_MODEL
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(SEQ_LEN)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        # if masking:
        #     outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)
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
        # outputs = tf.zeros_like(outputs)  # attention权重全部相等
        outputs = tf.nn.softmax(outputs)
        # outputs = tf.zeros_like(outputs)  # 没有attention输出
        attention = outputs

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs, attention

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
        Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs, attention = scaled_dot_product_attention(Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs, attention

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
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs

def self_attention(seq_input, score, attention_weights, attention_biases, drop_out, training=True):
    # input: seq_input(bc*seq_len*d) score(bc*seq_len)
    # return: logits(bc,seq_len)
    with tf.variable_scope('self-attetion', reuse=tf.AUTO_REUSE):
        src_masks = tf.math.equal(score, 0)  # socre=0判断是padding的部分
        enc = tf.layers.dense(seq_input, D_MODEL)
        enc *= D_MODEL ** 0.5  # scale
        enc += positional_encoding(enc)

        enc = tf.layers.dropout(enc, drop_out, training=training)

        # blocks
        attention_list = []
        for i in range(NUM_BLOCKS):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # self-attention
                enc, attention = multihead_attention(queries=enc,
                                          keys=enc,
                                          values=enc,
                                          key_masks=src_masks,
                                          num_heads=NUM_HEADS,
                                          dropout_rate=drop_out,
                                          training=training,
                                          causality=False)
                attention_list.append(attention)
                # feed forward
                enc = ff(enc, num_units=[D_FF, D_MODEL])

        logits = tf.squeeze(tf.layers.dense(enc,1))  # bc*seq_len

    return logits, attention_list