# %%

import warnings

warnings.filterwarnings('ignore')

import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras import activations, constraints, initializers, regularizers
from utils import *
import os
import numpy as np
import scipy.sparse as sp
import multiprocessing
import gc
import time


tf.compat.v1.disable_v2_behavior()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_WARNINGS"] = 'off'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def get_matrix(triples, ent_size, rel_size):
    print(ent_size, rel_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    for i in range(ent_size):
        adj_features[i, i] = 1

    for h, r, t in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1
        adj_features[t, h] = 1
        radj.append([h, t, r])
        radj.append([t, h, r + rel_size])
        rel_out[h][r] += 1
        rel_in[t][r] += 1

    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []
    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))
    return adj_matrix, r_index, r_val, adj_features, rel_features

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
        if ref in rank:
            rank_index = np.where(rank == ref)[0][0]
        else:
            rank_index = 15000
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    return mean, mrr, num, prec_set

def eval_alignment_by_sim_mat(embed1, embed2, top_k, nums_threads, csls=0, accurate=False, output=True):
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
            print(
                "\n hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, t_mean,
                                                                                      t_mrr,
                                                                                      time.time() - t))
        else:
            print("\nhits@{} = {}, time = {:.3f} s ".format(top_k, acc, time.time() - t))
    hits1 = acc[0]
    del sim_mat
    gc.collect()
    return t_prec_set, hits1


def get_hits(vec, test_pair, wrank=None, top_k=(1, 5, 10)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])

    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i, sim[i, j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank, -1), np.expand_dims(wrank, -1)], -1), axis=-1)
        sim = rank.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_lr / Lvec.shape[0]))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))

############################3NR_GraphAttention

class NR_GraphAttention(Layer):

    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 depth=1,
                 use_w=False,
                 attn_heads=2,
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
        rel_F = input_shape[1][-1]
        self.ent_F = node_F
        ent_F = self.ent_F
        # print('depth is {0}, head is {1}'.format(self.depth, self.attn_heads))
        for l in range(self.depth):

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
        features = inputs[0]
        rel_emb = inputs[1]
        adj = tf.SparseTensor(K.cast(K.squeeze(inputs[2], axis=0), dtype="int64"),
                              K.ones_like(inputs[2][0, :, 0]), (self.node_size, self.node_size))
        sparse_indices = tf.squeeze(inputs[3], axis=0)
        sparse_val = tf.squeeze(inputs[4], axis=0)

        features = self.activation(features)
        outputs.append(features)

        def p(**kwargs):
            pass
            # print("----------------")
            # for k, v in kwargs.items():
            #     print("{0}={1}".format(k, v))
            # print("----------------")

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                # print('layer {0}, head {1}'.format(l, head))
                attention_kernel = self.attn_kernels[l][head]
                rels_sum = tf.SparseTensor(indices=sparse_indices, values=sparse_val,
                                           dense_shape=(self.triple_size, self.rel_size))
                p(rel_emb=rel_emb, attention_kernel=attention_kernel, rels_sum=rels_sum, features=features)
                rels_sum = tf.compat.v1.sparse_tensor_dense_matmul(rels_sum, rel_emb)
                neighs = K.gather(features, adj.indices[:, 1])
                selfs = K.gather(features, adj.indices[:, 0])
                p(rels_sum=rels_sum, neighs=neighs, selfs=selfs)
                rels_sum = tf.nn.l2_normalize(rels_sum, 1)
                bias = tf.reduce_sum(neighs * rels_sum, 1, keepdims=True) * rels_sum
                neighs = neighs - 2 * bias
                p(rels_sum=rels_sum, bias=bias, neighs=neighs)

                att = K.squeeze(K.dot(K.concatenate([selfs, neighs, rels_sum]), attention_kernel), axis=-1)
                p(att=att)
                att = tf.SparseTensor(indices=adj.indices, values=att, dense_shape=adj.dense_shape)
                att = tf.compat.v1.sparse_softmax(att)
                new_features = tf.compat.v1.segment_sum(neighs * K.expand_dims(att.values, axis=-1), adj.indices[:, 0])

                p(att=att, new_features=new_features)
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = K.concatenate(features_list)  # (N x KF')
                p(features=features)
            else:
                features = K.mean(K.stack(features_list), axis=0)
            # print(self.attn_heads_reduction)
            features = self.activation(features)
            p(features=features)
            outputs.append(features)

        # print(outputs[0].get_shape())
        # print(outputs[1].get_shape())
        # print(outputs[2].get_shape())
        # if outputs[0].get_shape()[0] == outputs[1].get_shape()[0] == outputs[2].get_shape()[0]:
        outputs = K.concatenate(outputs)
        # else:
        #     outputs = K.concatenate([outputs[0], outputs[0], outputs[0]])
        p(outputs=outputs)
        # print(outputs.device)
        return outputs

    def compute_output_shape(self, input_shape):
        node_shape = self.node_size, (input_shape[0][-1]) * (self.depth + 1)
        return node_shape


class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""
    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim
    def compute_mask(self, inputs, mask=None):
        return None
    def call(self, inputs):
        return self.embeddings


class TFModelWrapper(object):
    def __init__(self, name='rrea', lang='en_fr', **kwargs):
        self.ent_sizes = kwargs['ent_sizes']
        self.sess = sess
        self.lang = lang
        self.construct_adj(**kwargs)
        self.load_pair(**kwargs)
        self.train_pair = []
        self.node_size = self.adj_features.shape[0]
        self.rel_size = self.rel_features.shape[1]
        self.triple_size = len(self.adj_matrix)
        self.batch_size = self.node_size
        default_params = dict(
            dropout_rate=0.30,
            node_size=self.node_size,
            rel_size=self.rel_size,
            n_attn_heads=1,
            depth=2,
            gamma=3,
            node_hidden=100,
            rel_hidden=100,
            triple_size=self.triple_size,
            batch_size=self.batch_size
        )
        get_model = dict(rrea=self.get_trgat)
        self.model, self.get_emb = get_model[name](**default_params)
        # self.model.summary()
        self.initial_weights = self.model.get_weights()

    def load_pair(self, **kwargs):
        self.update_devset(kwargs['link'].cpu().numpy())

    def construct_adj(self, triples=None, ent_sizes=None, rel_sizes=None, **kwargs):
        entsz = ent_sizes[0] + ent_sizes[1]
        relsz = rel_sizes[0] + rel_sizes[1]
        adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples, entsz, relsz)

        # return
        self.adj_matrix, self.r_index, self.r_val, self.adj_features, self.rel_features = \
            adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features
        self.adj_matrix = np.stack(self.adj_matrix.nonzero(), axis=1)

        # ## in case final ents have no connection, adds here?
        maxind0 = max(self.adj_matrix[:, 0])
        if maxind0 +1 < entsz:
            newlyadded = []
            print("RREA fault!!!!!!!!!!!!!!!!!!!!!!!!")
            for item in range(maxind0 +1, entsz):
                newlyadded.append([item, item])
            self.adj_matrix = np.concatenate((self.adj_matrix, np.array(newlyadded)))

        self.rel_matrix, self.rel_val = np.stack(self.rel_features.nonzero(), axis=1), self.rel_features.data
        self.ent_matrix, self.ent_val = np.stack(self.adj_features.nonzero(), axis=1), self.adj_features.data

    def convert_ent_id(self, ent_ids, which=0):

        return ent_ids
        # if which == 1:
        #     # return [val + self.ent_sizes[0] for val in ent_ids]
        #     return ent_ids + self.ent_sizes[0]
        # else:
        #     return ent_ids

    def convert_rel_id(self, rel_ids, which=0):
        raise NotImplementedError()

    def append_pairs(self, old_pair, new_pair):
        if len(old_pair) == 0:
            return new_pair
        px, py = set(), set()
        for e1, e2 in old_pair:
            px.add(e1)
            py.add(e2)
        filtered = []
        for e1, e2 in new_pair:
            if e1 not in px and e2 not in py:
                filtered.append([e1, e2])
        if len(filtered) == 0:
            return old_pair
        filtered = np.array(filtered)
        return np.concatenate([filtered, old_pair], axis=0)

    def update_trainset(self, pairs, append=False):
        trainset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]

        curr_pair = np.array(trainset).T
        if append:
            if append == 'REVERSE':
                self.train_pair = self.append_pairs(curr_pair, self.train_pair)
            else:
                self.train_pair = self.append_pairs(self.train_pair, curr_pair)
        else:
            self.train_pair = curr_pair
        # print('srs-iteration-update-train-pair')

    def update_devset(self, pairs):
        # pairs = [pairs[0], pairs[1]]
        devset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]
        self.dev_pair = np.array(devset).T

    def get_curr_embeddings(self, device=None):
        vec = self.get_embedding()
        vec = np.array(vec)
        sep = self.ent_sizes[0]
        vecs = vec[:sep], vec[sep:]
        vecs = apply(torch.from_numpy, *vecs)
        tf.compat.v1.reset_default_graph()
        return vecs if device is None else apply(lambda x: x.to(device), *vecs)

    def train1step(self, epoch=75):
        # self.test_train_pair_acc()
        # print("iteration %d start." % turn)
        verbose = 20
        if epoch > 100:
            verbose = epoch // 5
        for i in range(epoch):
            train_set = self.get_train_set()
            inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, train_set]
            inputs = [np.expand_dims(item, axis=0) for item in inputs]
            self.model.train_on_batch(inputs, np.zeros((1, 1)))
            # if (i + 1) % verbose == 0:
            #     self.CSLS_test()
        # self.CSLS_test()

    def test_train_pair_acc(self):
        pred = set((int(k), int(v)) for k, v in self.train_pair)
        actual = set((int(k), int(v)) for k, v in self.dev_pair)
        print('train pair={0}, dev pair={1}'.format(len(self.train_pair), len(self.dev_pair)))

        tp = len(pred.intersection(actual))
        fp = len(pred.difference(actual))
        fn = len(actual.difference(pred))
        # 😀
        print("tp={0}, fp={1}, fn={2}".format(tp, fp, fn))
        prec = float(tp) / float(tp + fp)
        recall = float(tp) / float(tp + fn)
        f1 = 2 * prec * recall / (prec + recall)
        print('prec={0}, recall={1}, f1={2}'.format(prec, recall, f1))
        return {
            'precision': prec,
            'recall': recall,
            'f1-score': f1,
            'confusion_matrix': (tp, fp, fn)
        }

    def get_embedding(self):
        inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        return self.get_emb.predict_on_batch(inputs)

    def test(self, wrank=None):
        vec = self.get_embedding()
        return get_hits(vec, self.dev_pair, wrank=wrank)

    def CSLS_test(self, thread_number=16, csls=10, accurate=True):
        if len(self.dev_pair) == 0:
            print('EVAL--No dev')
            return
        vec = self.get_embedding()
        Lvec = np.array([vec[e1] for e1, e2 in self.dev_pair])
        Rvec = np.array([vec[e2] for e1, e2 in self.dev_pair])
        Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
        return None

    def get_train_set(self, batch_size=None):
        if not batch_size:
            batch_size = self.batch_size
        negative_ratio = batch_size // len(self.train_pair) + 1
        train_set = np.reshape(np.repeat(np.expand_dims(self.train_pair, axis=0), axis=0, repeats=negative_ratio),
                               newshape=(-1, 2))
        np.random.shuffle(train_set)
        train_set = train_set[:batch_size]
        train_set = np.concatenate([train_set, np.random.randint(0, self.node_size, train_set.shape)], axis=-1)
        return train_set

    def get_trgat(self, node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0.,
                  gamma=3, lr=0.005, depth=2, **kwargs):
        adj_input = Input(shape=(None, 2))
        index_input = Input(shape=(None, 2), dtype='int64')
        val_input = Input(shape=(None,))
        rel_adj = Input(shape=(None, 2))
        ent_adj = Input(shape=(None, 2))

        ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
        rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

        def avg(tensor, size):
            adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
            adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                                  dense_shape=(node_size, size))
            adj = tf.compat.v1.sparse_softmax(adj)
            return tf.compat.v1.sparse_tensor_dense_matmul(adj, tensor[1])

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
            return tf.compat.v1.reduce_sum(loss, keep_dims=True) / self.batch_size

        loss = Lambda(align_loss)(find)
        inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
        train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
        train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.RMSprop(lr))

        feature_model = keras.Model(inputs=inputs, outputs=out_feature)
        return train_model, feature_model
