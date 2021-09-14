import warnings
warnings.filterwarnings('ignore')

import os
import random
import keras
from tqdm import *
from utils import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras import activations, constraints, initializers, regularizers
import time
import gc
import multiprocessing

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""
    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


class NR_GraphAttention(Layer):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 depth=1,
                 use_w=False,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.3,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias
        self.use_w = use_w
        self.depth = depth

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.biases = []
        self.attn_kernels = []
        self.gat_kernels = []
        self.interfaces = []
        self.gate_kernels = []

        super(NR_GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        node_F = input_shape[0][-1]
        # rel_F = input_shape[1][-1]
        self.ent_F = node_F
        # ent_F = self.ent_F

        for l in range(self.depth):  # depths
            self.attn_kernels.append([])
            for head in range(self.attn_heads):
                attn_kernel = self.add_weight(shape=(3 * node_F, 1),
                                              initializer=self.attn_kernel_initializer,
                                              regularizer=self.attn_kernel_regularizer,
                                              constraint=self.attn_kernel_constraint,
                                              name='attn_kernel_self_{}'.format(head))

                self.attn_kernels[l].append(attn_kernel)
        self.built = True

    def call(self, inputs):
        outputs = []
        features = inputs[0]  # entity entity/relation embeddings
        rel_emb = inputs[1]  # relation embedding
        adj = tf.SparseTensor(K.cast(K.squeeze(inputs[2], axis=0), dtype="int64"),
                              K.ones_like(inputs[2][0, :, 0]), (self.node_size, self.node_size))
        sparse_indices = tf.squeeze(inputs[3], axis=0)
        sparse_val = tf.squeeze(inputs[4], axis=0)

        features = self.activation(features)
        outputs.append(features)  # the initial embeddings

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]

                rels_sum = tf.SparseTensor(indices=sparse_indices, values=sparse_val,
                                           dense_shape=(self.triple_size, self.rel_size))

                rels_sum = tf.sparse_tensor_dense_matmul(rels_sum, rel_emb)  # 针对每一个(h,t)来说，其对应的r的embedding表示

                neighs = K.gather(features, adj.indices[:, 1])  # no self-loops... 和前面rels_sum的(h,t) key是对应的
                selfs = K.gather(features, adj.indices[:, 0])

                rels_sum = tf.nn.l2_normalize(rels_sum, 1)
                bias = tf.reduce_sum(neighs * rels_sum, 1, keepdims=True) * rels_sum
                neighs = neighs - 2 * bias

                att = K.squeeze(K.dot(K.concatenate([selfs, neighs, rels_sum]), attention_kernel), axis=-1)
                att = tf.SparseTensor(indices=adj.indices, values=att, dense_shape=adj.dense_shape)
                att = tf.sparse_softmax(att)

                new_features = tf.segment_sum(neighs * K.expand_dims(att.values, axis=-1), adj.indices[:, 0])
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = K.concatenate(features_list)  # (N x KF')
            else:
                features = K.mean(K.stack(features_list), axis=0)

            features = self.activation(features)
            outputs.append(features)

        outputs = K.concatenate(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        node_shape = self.node_size, (input_shape[0][-1]) * (self.depth + 1)
        return node_shape

def get_embedding():
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    return get_emb.predict_on_batch(inputs)


def test(wrank=None):
    vec = get_embedding()
    np.save(args.data_dir + '/vec.npy', vec)
    return get_hits(vec, dev_pair, wrank=wrank)

def get_train_set(batch_size):
    negative_ratio = batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set)
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set


def get_trgat(node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0, gamma=3, lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input) # do not care about the input, just obtain embeddings..
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64") #数据类型转换
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]

    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])

    encoder = NR_GraphAttention(node_size, activation="relu",
                                rel_size=rel_size,
                                depth=depth,
                                attn_heads=n_attn_heads,
                                triple_size=triple_size,
                                attn_heads_reduction='average',
                                dropout_rate=dropout_rate)

    out_feature = Concatenate(-1)([encoder([ent_feature] + opt), encoder([rel_feature] + opt)])
    out_feature = Dropout(dropout_rate)(out_feature)

    alignment_input = Input(shape=(None, 4))
    find = Lambda(lambda x: K.gather(reference=x[0], indices=K.cast(K.squeeze(x[1], axis=0), 'int32')))(
        [out_feature, alignment_input])

    def align_loss(tensor):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_

        def l1(ll, rr):
            return K.sum(K.abs(ll - rr), axis=-1, keepdims=True)

        def l2(ll, rr):
            return K.sum(K.square(ll - rr), axis=-1, keepdims=True)

        l, r, fl, fr = [tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], tensor[:, 3, :]]
        loss = K.relu(gamma + l1(l, r) - l1(l, fr)) + K.relu(gamma + l1(l, r) - l1(fl, r))
        return tf.reduce_sum(loss, keep_dims=True) / (batch_size)

    loss = Lambda(align_loss)(find)

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))

    feature_model = keras.Model(inputs=inputs, outputs=out_feature)
    return train_model, feature_model

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/en_de", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--ratio", type=float, default=0.3, help="the ratio for training")  # 0.2
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    lang = args.data_dir
    train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features = load_data('./%s/'%lang, train_ratio=args.ratio)

    adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1) # convert the sparse matrix to the connection matrix


    rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data # convert the sparse matrix to the connection matrix with rel values..
    ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data

    node_size = adj_features.shape[0]
    rel_size = rel_features.shape[1]
    triple_size = len(adj_matrix)
    batch_size = node_size # num of triples + self_loops!!!

    model,get_emb = get_trgat(dropout_rate=0.30,node_size=node_size,rel_size=rel_size,n_attn_heads = 1,depth=2,gamma =3,node_hidden=100,rel_hidden = 100,triple_size = triple_size)
    model.summary(); initial_weights = model.get_weights()

    rest_set_1 = [e1 for e1, e2 in dev_pair]
    rest_set_2 = [e2 for e1, e2 in dev_pair]
    np.random.shuffle(rest_set_1)
    np.random.shuffle(rest_set_2)

    epoch = 1200
    for i in trange(epoch):
        train_set = get_train_set(batch_size)
        inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        model.train_on_batch(inputs, np.zeros((1, 1)))
        if i % 300 == 299:
            test()