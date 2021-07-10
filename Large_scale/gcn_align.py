from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp
import math


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name + ":0")
    except:
        return tf.placeholder(tf.int32, name=name)

def align_loss(outlayer, ILL, gamma, k, t):
    left = ILL[:, 0]
    right = ILL[:, 1]
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = get_placeholder_by_name("neg_left")  # tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = get_placeholder_by_name("neg_right")  # tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    neg_left = get_placeholder_by_name("neg2_left")  # tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = get_placeholder_by_name("neg2_right")  # tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def trunc_normal(shape, name=None, normalize=True):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(shape[0])))
    if not normalize: return initial
    return tf.nn.l2_normalize(initial, 1)

_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class GraphConvolution(Layer):
    """Graph convolution layer. (featureless=True and transform=False) is not supported for now."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, transform=True, init=glorot, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.transform = transform

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                if input_dim == output_dim and not self.transform and not featureless: continue
                self.vars['weights_' + str(i)] = init([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.dropout:
            if self.sparse_inputs:
                x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            else:
                x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if 'weights_'+str(i) in self.vars:
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
            else:
                pre_sup = x
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)




class Model(object):
    def __init__(self, **kwargs):
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN_Align(Model):
    def __init__(self, placeholders, input_dim, output_dim, sparse_inputs=False, featureless=True, **kwargs):
        super(GCN_Align, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ILL = placeholders['ill']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=kwargs.get('lr', 0.005))
        self.k = kwargs.get('k', 5)
        self.t = placeholders['t']
        self.gamma = kwargs.get('gamma', 3.0)
        self.build()

    def _loss(self):
        self.loss += align_loss(self.outputs, self.ILL, self.gamma, self.k, self.t)

    def _accuracy(self):
        pass

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            transform=False,
                                            init=trunc_normal,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=False,
                                            transform=False,
                                            logging=self.logging))



def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if

def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))

def construct_feed_dict(features, support, placeholders, train=[]):
    """Construct feed dictionary for GCN-Align."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['ill']: train})
    feed_dict.update({placeholders['t']: float(len(train))})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict

class GCNAlignWrapper:
    def __init__(self, **kwargs):
        # Set random seed
        seed = 12306
        np.random.seed(seed)
        tf.set_random_seed(seed)
        tf.disable_eager_execution()

        self.init_inputs(**kwargs)
        self.k = 5
        self.dim = 200
        self.dropout = 0.
        self.gamma = 3.0
        self.lr = 20
        # Some preprocessing
        self.support = [preprocess_adj(self.adj)]
        num_supports = 1
        model_func = GCN_Align
        k = self.k

        self.ph_se = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder_with_default(0, shape=()),
            'ill': tf.placeholder(tf.int32, shape=(None, 2)),
            't': tf.placeholder(tf.float32),
        }

        # Create model
        # model_ae = model_func(ph_ae, input_dim=ae_input[2][1], output_dim=self.ae_dim, ILL=train, sparse_inputs=True,
        #                       featureless=False, logging=True)
        self.model_se = GCN_Align(self.ph_se, input_dim=self.ent_total, output_dim=self.dim, ILL=None,
                                  sparse_inputs=False,
                                  featureless=True,
                                  logging=True,
                                  gamma=self.gamma,
                                  k=self.k,
                                  lr=self.lr)
        # Initialize session
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        self.sess = sess
        pass

    def init_inputs(self, **kwargs):

        self.ent_nums = kwargs.get('ent_sizes', [15000, 15000])
        self.rel_nums = kwargs.get('rel_sizes', [500, 500])
        self.ent_total = sum(self.ent_nums)
        self.rel_total = sum(self.rel_nums)
        self.ent_split = self.ent_nums[0]
        self.rel_split = self.rel_nums[0]

        triples = np.array(kwargs['triples'])
        self.adj = get_weighted_adj(self.ent_total, triples)
        self.update_devset(kwargs['link'].cpu().numpy())

    def convert_ent_id(self, ent_ids, which=0):
        # if which == 1:
        #     # return [val + self.ent_sizes[0] for val in ent_ids]
        #     return ent_ids + self.ent_split
        # else:
        return ent_ids

    def convert_rel_id(self, rel_ids, which=0):
        # if which == 1:
        #     # return [val + self.ent_sizes[0] for val in ent_ids]
        #     return rel_ids + self.ent_split
        # else:
        return rel_ids

    def append_pairs(self, old_pair, new_pair):
        px, py = set(), set()
        for e1, e2 in old_pair:
            px.add(e1)
            py.add(e2)
        filtered = []
        for e1, e2 in new_pair:
            if e1 not in px and e2 not in py:
                filtered.append([e1, e2])
        return np.concatenate([np.array(filtered), old_pair], axis=0)

    def update_trainset(self, pairs, append=False):
        trainset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]

        curr_pair = np.array(trainset).T
        self.train_pair = self.append_pairs(self.train_pair, curr_pair) if append else curr_pair
        # print('srs-iteration-update-train-pair')

    def update_devset(self, pairs):
        devset = [self.convert_ent_id(p, i) for i, p in enumerate(pairs)]
        self.dev_pair = np.array(devset).T

    def get_curr_embeddings(self, device=None):
        import torch
        from utils import apply
        feed_dict_se = construct_feed_dict(1.0, self.support, self.ph_se, train=self.train_pair)

        vec_se = self.sess.run(self.model_se.outputs, feed_dict=feed_dict_se)
        vec = np.array(vec_se)
        sep = self.ent_split
        vecs = vec[:sep], vec[sep:]
        vecs = apply(torch.from_numpy, *vecs)

        # tf.compat.v1.disable_eager_execution()
        return vecs if device is None else apply(lambda x: x.to(device), *vecs)

    def train1step(self, epoch):

        cost_val = []
        train = self.train_pair
        k = self.k
        e = self.ent_total
        sess = self.sess
        ph_se = self.ph_se
        support = self.support
        model_se = self.model_se
        t = len(train)
        L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
        neg_left = L.reshape((t * k,))
        L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
        neg2_right = L.reshape((t * k,))

        # Train model
        for epoch in range(epoch):
            if epoch % 10 == 0:
                neg2_left = np.random.choice(e, t * k)
                neg_right = np.random.choice(e, t * k)
            # Construct feed dictionary
            # feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
            # feed_dict_ae.update({ph_ae['dropout']: self.dropout})
            # feed_dict_ae.update(
            #     {'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left,
            #      'neg2_right:0': neg2_right})
            feed_dict_se = construct_feed_dict(1.0, support, ph_se, train=train)
            feed_dict_se.update({self.ph_se['dropout']: self.dropout})
            feed_dict_se.update(
                {'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left,
                 'neg2_right:0': neg2_right})
            # Training step
            # outs_ae = sess.run([model_ae.opt_op, model_ae.loss], feed_dict=feed_dict_ae)
            outs_se = sess.run([model_se.opt_op, model_se.loss], feed_dict=feed_dict_se)
            # cost_val.append((outs_ae[1], outs_se[1]))

            # Print results
            # print("Epoch:", '%04d' % (epoch + 1), "SE_train_loss=",
            #       "{:.5f}".format(outs_se[1]))

        # print("Optimization Finished!")
        pass
