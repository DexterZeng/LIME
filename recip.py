import warnings

warnings.filterwarnings('ignore')

import os
import random
import keras
from tqdm import *
import numpy as np
from utils import *
from CSLS import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from layer import NR_GraphAttention
from keras import activations, constraints, initializers, regularizers
from scipy.stats import rankdata
from collections import defaultdict
class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings


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

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(
        val_input)  # do not care about the input, just obtain embeddings..
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")  # 数据类型转换
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]

    ent_feature = Lambda(avg, arguments={'size': node_size})(
        [ent_adj, ent_emb])  # 初始化为相邻节点和自己的embeddings的平均值 对于每一个ent, 其周围ent信息
    rel_feature = Lambda(avg, arguments={'size': rel_size})(
        [rel_adj, rel_emb])  # 初始化为相邻节点和自己的embeddings的平均值 对于每一个ent，其周围rel信息

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
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60, '*'))


def gen_adMtrx(x, y, aep_fuse):
    adMtrx = dict()
    for i in range(len(x)):
        x_ele = x[i]
        y_ele = y[i] + aep_fuse.shape[0]
        if x_ele not in adMtrx:
            ents = []
        else:
            ents = adMtrx[x_ele]
        ents.append(y_ele)
        adMtrx[x_ele] = ents
        if y_ele not in adMtrx:
            ents = []
        else:
            ents = adMtrx[y_ele]
        ents.append(x_ele)
        adMtrx[y_ele] = ents
    return adMtrx

def gen_adMtrx_more(x, y, rows, columns, aep_fuse):
    adMtrx1 = dict()
    for i in range(len(x)):
        x_ele = rows[x[i]]
        y_ele = columns[y[i]] + aep_fuse.shape[0]
        if x_ele not in adMtrx1:
            ents = []
        else:
            ents = adMtrx1[x_ele]
        ents.append(y_ele)

        adMtrx1[x_ele] = ents

        if y_ele not in adMtrx1:
            ents = []
        else:
            ents = adMtrx1[y_ele]
        ents.append(x_ele)
        adMtrx1[y_ele] = ents
    return adMtrx1

def BFS(graph, vertex):
    # 使用列表作为队列
    queue = []
    # 将首个节点添加到队列中
    queue.append(vertex)
    # 使用集合来存放已访问过的节点
    looked = set()
    # 将首个节点添加到集合中表示已访问
    looked.add(vertex)
    # 当队列不为空时进行遍历
    while (len(queue) > 0):
        # 从队列头部取出一个节点并查询该节点的相邻节点
        temp = queue.pop(0)
        nodes = graph[temp]
        # 遍历该节点的所有相邻节点
        for w in nodes:
            # 判断节点是否存在于已访问集合中,即是否已被访问过
            if w not in looked:
                # 若未被访问,则添加到队列中,同时添加到已访问集合中,表示已被访问
                queue.append(w)
                looked.add(w)
        #print(temp, end=' ')
    # print(len(looked))
    return looked

def get_blocks(adMtrx, allents):
    count1 = 0
    Graph = adMtrx
    leftents = allents
    blocks = []
    lenghs = []
    while len(leftents) > 0:
        vertex = list(leftents)[0]
        if vertex in Graph:
            matched = BFS(Graph, vertex)
        else:
            matched = {vertex}
        leftents = leftents.difference(matched)
        blocks.append(matched)
        lenghs.append(len(matched))
        if len(matched) == 1:
            count1 += 1
        # print()
    # print(blocks)
    print('Total blocks: ' + str(len(blocks)))
    # print(lenghs)
    print('Total blocks with length 1: ' + str(count1))
    # print(count1)
    # print(lenghs[0])
    return blocks

def reciprocal(sim_mat):
    sim_mat_r = sim_mat.T
    sim_mat = csls_sim_as(sim_mat)
    # h1, h10, mrr = get_hits_ma(sim_mat)
    sim_mat_r = csls_sim_as(sim_mat_r)
    ranks = rankdata(-sim_mat, axis=1)
    ranks_r = rankdata(-sim_mat_r, axis=1)
    # ARITHMETIC MEAN
    rankfused = (ranks + ranks_r.T) / 2
    return rankfused

def ana_blocks(blocks, aep_fuse, maxtruth, correct_coun, recip_flag):
    all1s = []
    refined_blocks = 0
    lens = []
    for block in blocks:
        if len(block) > 1:
            lens.append(len(block))
            refined_blocks += 1
            rows = []
            columns = []

            for item in block:
                if item < aep_fuse.shape[0]:
                    rows.append(item)
                    if item + aep_fuse.shape[0] in block:
                        maxtruth += 1
                else:
                    columns.append(item - aep_fuse.shape[0])

            tempM = aep_fuse[rows][:, columns]

            if recip_flag is False:
                for i in range(tempM.shape[0]):
                    rank = (-tempM[i, :]).argsort()
                    if rows[i] == columns[rank[0]]:
                        correct_coun += 1
            else:
                rankfused = reciprocal(tempM)
                for i in range(rankfused.shape[0]):
                    rank = (rankfused[i, :]).argsort()
                    if rows[i] == columns[rank[0]]:
                        correct_coun += 1
        else:
            all1s.extend(block)
    print('Max length: ' + str(np.max(np.array(lens))))
    print('Total blocks after refinement: ' + str(refined_blocks+1))
    return all1s, maxtruth, correct_coun

def dirtect_process(maxtruth, correct_coun, all1s, aep_fuse, flag, recip_flag):
    rows = []
    columns = []

    for item in all1s:
        if item < aep_fuse.shape[0] and item + aep_fuse.shape[0] in all1s:
            maxtruth += 1
        if item < aep_fuse.shape[0]:
            rows.append(item)
        if item > aep_fuse.shape[0]:
            columns.append(item-aep_fuse.shape[0])

    tempM = aep_fuse[rows][:, columns]

    if flag is True:
        if len(rows)>1 and len(columns)>1:
            if recip_flag is False:
                for i in range(tempM.shape[0]):
                    rank = (-tempM[i, :]).argsort()
                    if rows[i] == columns[rank[0]]:
                        correct_coun += 1
            else:
                # # reciprocal
                rankfused = reciprocal(tempM)
                for i in range(rankfused.shape[ 0]):
                    rank = (rankfused[i, :]).argsort()
                    if rows[i] == columns[rank[0]]:
                        correct_coun += 1

        print()
        print('Max truth: ' + str(maxtruth))
        print('Total correct: ' + str(correct_coun))

        print()
        print("Hits@1: " + str(correct_coun*1.0/aep_fuse.shape[0]))
    return rows, columns, tempM


def male_without_match(matches, males):
    for male in males:
        if male not in matches:
            return male

def deferred_acceptance(male_prefs, female_prefs):
    female_queue = defaultdict(int)
    males = list(male_prefs.keys())
    matches = {}
    while True:
        male = male_without_match(matches, males)
        # print(male)
        if male is None:
            break
        female_index = female_queue[male]
        female_queue[male] += 1
        # print(female_index)

        try:
            female = male_prefs[male][female_index]
        except IndexError:
            matches[male] = male
            continue
        # print('Trying %s with %s... ' % (male, female), end='')
        prev_male = matches.get(female, None)
        if not prev_male:
            matches[male] = female
            matches[female] = male
            # print('auto')
        elif female_prefs[female].index(male) < \
             female_prefs[female].index(prev_male):
            matches[male] = female
            matches[female] = male
            del matches[prev_male]
            # print('matched')
        # else:
            # print('rejected')
    return {male: matches[male] for male in male_prefs.keys()}

def eva_sm(sim_mat):
    t = time.time()
    print('stable matching...')
    thr = 10500
    thr2 = 10500

    scale = sim_mat.shape[0]
    # store preferences
    MALE_PREFS = {}
    FEMALE_PREFS = {}
    pref = np.argsort(-sim_mat[:scale, :scale], axis=1)
    print("Generate the preference scores time elapsed: {:.4f} s".format(time.time() - t))

    for i in range(scale):
        lis = (pref[i] + thr2).tolist()
        MALE_PREFS[i] = lis
    print("Forming the preference scores time 1 elapsed: {:.4f} s".format(time.time() - t))
    del pref

    pref_col = np.argsort(-sim_mat[:scale, :scale], axis=0)
    print("Generate the preference scores time elapsed: {:.4f} s".format(time.time() - t))

    for i in range(scale):
        FEMALE_PREFS[i + thr2] = pref_col[:, i].tolist()
    print("Forming the preference scores time 2 elapsed: {:.4f} s".format(time.time() - t))
    del pref_col

    matches = deferred_acceptance(MALE_PREFS, FEMALE_PREFS)
    del MALE_PREFS
    del FEMALE_PREFS
    print("Deferred acceptance time elapsed: {:.4f} s".format(time.time() - t))

    # print(matches)
    trueC = 0
    for match in matches:
        if match + thr2 == matches[match]:
            trueC += 1
    print('accuracy： ' + str(float(trueC) / thr))
    print("total time elapsed: {:.4f} s".format(time.time() - t))


def eva_sm_1(sim_mat, sim_mat1):
    t = time.time()
    print('stable matching...')
    thr = 10500
    thr2 = 10500

    scale = sim_mat.shape[0]
    # store preferences
    MALE_PREFS = {}
    FEMALE_PREFS = {}
    pref = np.argsort(-sim_mat[:scale, :scale], axis=1)
    print("Generate the preference scores time elapsed: {:.4f} s".format(time.time() - t))

    for i in range(scale):
        lis = (pref[i] + thr2).tolist()
        MALE_PREFS[i] = lis
    print("Forming the preference scores time 1 elapsed: {:.4f} s".format(time.time() - t))
    del pref

    pref_col = np.argsort(-sim_mat1[:scale, :scale], axis=1)
    print("Generate the preference scores time elapsed: {:.4f} s".format(time.time() - t))

    for i in range(scale):
        FEMALE_PREFS[i + thr2] = pref_col[i].tolist()
    print("Forming the preference scores time 2 elapsed: {:.4f} s".format(time.time() - t))
    del pref_col

    matches = deferred_acceptance(MALE_PREFS, FEMALE_PREFS)
    del MALE_PREFS
    del FEMALE_PREFS
    print("Deferred acceptance time elapsed: {:.4f} s".format(time.time() - t))

    # print(matches)
    trueC = 0
    for match in matches:
        if match + thr2 == matches[match]:
            trueC += 1
    print('accuracy： ' + str(float(trueC) / thr))
    print("total time elapsed: {:.4f} s".format(time.time() - t))

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dbp_yg zh_en ja_en fr_en en_fr en_de dbp_wd dbp_yg
    parser.add_argument("--data_dir", type=str, default="data/en_fr", required=False,
                        help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--ratio", type=float, default=0.3, help="the ratio for training")  # 0.2
    args = parser.parse_args()

    make_print_to_file(args.data_dir.split('/')[-1] + '_' + str(args.ratio), path='./logs/')

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    lang = args.data_dir
    train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data('./%s/' % lang,
                                                                                             train_ratio=args.ratio)

    vec = np.load(args.data_dir + '/vec.npy')
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])


    # Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    # Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    # thread_number = 16; csls = 10; accurate = True
    # eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)

    def eva(sim_mat, use_min = False):
        if use_min is True:
            predicted = np.argmin(sim_mat, axis=1)
        else:
            predicted = np.argmax(sim_mat, axis=1)
        cor = predicted == np.array(range(sim_mat.shape[0]))
        cor_num = np.sum(cor)
        print("Acc: " + str(cor_num) + ' / ' + str(len(cor)) + ' = ' + str(cor_num*1.0/len(cor)))

    def csls_sim__(sim_mat, k):
        nearest_values1 = calculate_nearest_k(sim_mat, k)
        nearest_values2 = calculate_nearest_k(sim_mat.T, k)
        csls_sim_mat = 2 * sim_mat.T - nearest_values1
        csls_sim_mat = csls_sim_mat.T - nearest_values2
        return csls_sim_mat

    def calculate_nearest_k(sim_mat, k):
        sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
        nearest_k = sorted_mat[:, 0:k]
        return np.mean(nearest_k, axis=1)

    def csls_sim_as(sim_mat):
        nearest_values2 = np.max(sim_mat.T, axis=1)
        csls_sim_mat = sim_mat - nearest_values2
        return csls_sim_mat

    import time

    t_total = time.time()
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    sim_mat = np.matmul(Lvec, Rvec.T)
    # eva(sim_mat)
    # eva_sm(sim_mat)


    # csls_sim_mat = csls_sim__(sim_mat, 10)
    # eva(csls_sim_mat)
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    #
    # csls_sim_mat = csls_sim__(sim_mat, 5)
    # eva(csls_sim_mat)
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    #
    # csls_sim_mat = csls_sim__(sim_mat, 1)
    # eva(csls_sim_mat)
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    # str_sim = np.load(args.data_dir + '/string_mat_train.npy')
    # str_sim = str_sim[:10500, :10500]
    # # eva(str_sim)
    # # get_hits_ma(str_sim, test)
    # aep_n = np.load(args.data_dir + '/name_mat_train.npy')
    # # eva(aep_n)
    # if 'fr_en' in args.data_dir or 'en_fr' in args.data_dir or 'en_de' in args.data_dir:
    #     weight_stru = 0.33
    #     weight_text = 0.33
    #     weight_string = 0.33
    # else:
    #     weight_stru = 0.7
    #     weight_text = 0.3
    #     weight_string = 0
    # sim_mat = (sim_mat * weight_stru + aep_n * weight_text + str_sim * weight_string)

    sim_mat_r = sim_mat.T

    eva(sim_mat)
    print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    sim_mat = csls_sim_as(sim_mat)
    sim_mat_r = csls_sim_as(sim_mat_r)

    recip_sim = (sim_mat + sim_mat_r.T) / 2.0
    eva(recip_sim)
    print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    eva_sm_1(sim_mat, sim_mat_r)


    # ranks = rankdata(-sim_mat, method="average", axis=1) # ordinal dense min max average
    # ranks_r = rankdata(-sim_mat_r, method="average", axis=1)
    # rankfused = (ranks + ranks_r.T) / 2
    # eva(rankfused, True)
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    # eva_sm(-rankfused)

    # #progressive blocking
    # recip_flag = True
    # t = time.time()
    # aep_fuse = sim_mat  # should be the similarity score
    # row_max = np.max(aep_fuse, axis=1)
    # Thres = [np.percentile(row_max, 40), np.percentile(row_max, 20), np.percentile(row_max, 1)]
    # print(Thres)
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    #
    # thres = Thres[0]
    # x, y = np.where(aep_fuse > thres)
    # adMtrx = gen_adMtrx(x, y, aep_fuse)
    #
    # allents = set()
    # for i in range(aep_fuse.shape[0] + aep_fuse.shape[1]):
    #     allents.add(i)
    # blocks = get_blocks(adMtrx, allents)
    # del adMtrx
    # del allents
    # del x, y
    # # evaluation!!!!
    # maxtruth = 0
    # correct_coun = 0
    #
    # all1s, maxtruth, correct_coun = ana_blocks(blocks, aep_fuse, maxtruth, correct_coun, recip_flag)
    #
    # rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s, aep_fuse, False,
    #                                        recip_flag)
    # print('Lagrest block... (all 1s): ' + str(len(all1s)))
    # del all1s
    #
    # print('\n*********************************************************')
    # thres2 = Thres[1]
    # x, y = np.where(tempM > thres2)
    # adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)
    #
    # allents = []
    # allents.extend(rows)
    # for item in columns:
    #     allents.append(item + aep_fuse.shape[0])
    # allents = set(allents)
    # newblocks = get_blocks(adMtrx1, allents)
    # del adMtrx1
    # del allents
    #
    # all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag)
    #
    # rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, False, recip_flag)
    # print('Lagrest block... (all 1s): ' + str(len(all1s_new)))
    # del all1s_new
    #
    # print('\n**********************************************************')
    # thres2 = Thres[2]
    # x, y = np.where(tempM > thres2)
    # adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)
    #
    # allents = []
    # allents.extend(rows)
    # for item in columns:
    #     allents.append(item + aep_fuse.shape[0])
    # allents = set(allents)
    #
    # newblocks = get_blocks(adMtrx1, allents)
    # del adMtrx1
    # del allents
    #
    # all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag)
    #
    # rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, False, recip_flag)
    #
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    #
    # print('\n**********************************************************')
    # thres2 = 0.5
    # x, y = np.where(tempM > thres2)
    # adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)
    #
    # allents = []
    # allents.extend(rows)
    # for item in columns:
    #     allents.append(item + aep_fuse.shape[0])
    # allents = set(allents)
    #
    # newblocks = get_blocks(adMtrx1, allents)
    # del adMtrx1
    # del allents
    #
    # all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag)
    #
    # rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, False, recip_flag)
    #
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    #
    # print('\n**********************************************************')
    # thres2 = 0.4
    # x, y = np.where(tempM > thres2)
    # adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)
    #
    # allents = []
    # allents.extend(rows)
    # for item in columns:
    #     allents.append(item + aep_fuse.shape[0])
    # allents = set(allents)
    #
    # newblocks = get_blocks(adMtrx1, allents)
    # del adMtrx1
    # del allents
    #
    # all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag)
    #
    # rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, True, recip_flag)
    #
    # print("total time elapsed: {:.4f} s".format(time.time() - t_total))

