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

def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]
    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        k = ls_len % n
        ls_return = []
        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

def cal_csls_sim(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values

def CSLS_sim(sim_mat1, k, nums_threads):
    tasks = div_list(np.array(range(sim_mat1.shape[0])), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_csls_sim, (sim_mat1[task, :], k)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values

def sim_handler(embed1, embed2, k, nums_threads):
    sim_mat = np.matmul(embed1, embed2.T)
    if k <= 0:
        print("k = 0")
        return sim_mat
    csls1 = CSLS_sim(sim_mat, k, nums_threads)
    csls2 = CSLS_sim(sim_mat.T, k, nums_threads)
    csls_sim_mat = 2 * sim_mat.T - csls1
    csls_sim_mat = csls_sim_mat.T - csls2
    del sim_mat
    gc.collect()
    return csls_sim_mat

def cal_rank_by_sim_mat(task, sim, top_k, accurate):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    prec_set = set()
    for i in range(len(task)):
        ref = task[i]
        if accurate:
            rank = (-sim[i, :]).argsort()
        else:
            rank = np.argpartition(-sim[i, :], np.array(top_k) - 1)
        prec_set.add((ref, rank[0]))
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set

def eval_alignment_by_sim_mat(embed1, embed2, top_k, nums_threads, csls=0, accurate=False,output = True):
    t = time.time()
    sim_mat = sim_handler(embed1, embed2, csls, nums_threads)
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    t_prec_set = set()
    tasks = div_list(np.array(range(ref_num)), nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank_by_sim_mat, (task, sim_mat[task, :], top_k, accurate)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)
        t_prec_set |= prec_set
    assert len(t_prec_set) == ref_num
    acc = np.array(t_num) / ref_num * 100
    for i in range(len(acc)):
        acc[i] = round(acc[i], 2)
    t_mean /= ref_num
    t_mrr /= ref_num
    if output:
        if accurate:
            print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                                       t_mrr,
                                                                                                       time.time() - t))
        else:
            print("hits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    return t_prec_set, hits1

def get_embedding():
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    return get_emb.predict_on_batch(inputs)


def test(wrank=None):
    vec = get_embedding()
    return get_hits(vec, dev_pair, wrank=wrank)


def CSLS_test(thread_number=16, csls=10, accurate=True):
    vec = get_embedding()
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    np.save(args.data_dir + '/vec.npy', vec)
    return None

def recip(thread_number=16, csls=10, accurate=True):
    vec = get_embedding()
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
    np.save(args.data_dir + '/vec.npy', vec)
    return None


def get_train_set(batch_size):
    negative_ratio = batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set)
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set


def get_trgat(node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0, gamma=3,
              lr=0.005, depth=2):
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

def make_print_to_file(fileName, path='./'):
    import sys
    import os
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/en_de", required=False, help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--ratio", type=float, default=0.3, help="the ratio for training")  # 0.2
    args = parser.parse_args()

    # make_print_to_file(args.data_dir.split('/')[-1]  + '_' + str(args.ratio), path='./logs/')
    # print(args)

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
            CSLS_test()