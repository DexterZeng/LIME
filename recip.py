import warnings

warnings.filterwarnings('ignore')

import os
import random
import keras
from tqdm import *
import numpy as np
from utils import *
import tensorflow as tf
from scipy.stats import rankdata
from collections import defaultdict


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
    queue = []
    queue.append(vertex)
    looked = set()
    looked.add(vertex)
    while (len(queue) > 0):
        temp = queue.pop(0)
        nodes = graph[temp]
        for w in nodes:
            if w not in looked:
                queue.append(w)
                looked.add(w)
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
    print('Total blocks: ' + str(len(blocks)))
    print('Total blocks with length 1: ' + str(count1))
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

import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dbp_yg zh_en ja_en fr_en en_fr en_de dbp_wd dbp_yg
    parser.add_argument("--data_dir", type=str, default="data/en_fr", required=False,
                        help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--method", type=str, default="ralign", help="inference strategies, ralign, ralign-wr, ralign-pb, dalign")  # 0.2
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    lang = args.data_dir
    train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data('./%s/' % lang)

    vec = np.load(args.data_dir + '/vec.npy')
    Lvec = np.array([vec[e1] for e1, e2 in dev_pair])
    Rvec = np.array([vec[e2] for e1, e2 in dev_pair])

    t_total = time.time()
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    sim_mat = np.matmul(Lvec, Rvec.T)

    if args.method == "dalign":
        eva(sim_mat)
    elif args.method == "ralign-wr":
        sim_mat_r = sim_mat.T

        sim_mat = csls_sim_as(sim_mat)
        sim_mat_r = csls_sim_as(sim_mat_r)

        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        recip_sim = (sim_mat + sim_mat_r.T) / 2.0
        eva(recip_sim)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    elif args.method == "ralign":
        sim_mat_r = sim_mat.T

        sim_mat = csls_sim_as(sim_mat)
        sim_mat_r = csls_sim_as(sim_mat_r)

        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        recip_sim = (sim_mat + sim_mat_r.T) / 2.0
        eva(recip_sim)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        ranks = rankdata(-sim_mat, method="average", axis=1) # ordinal dense min max average
        ranks_r = rankdata(-sim_mat_r, method="average", axis=1)
        rankfused = (ranks + ranks_r.T) / 2
        eva(rankfused, True)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))
    elif args.method == "ralign-pb":
        #progressive blocking
        recip_flag = True
        t = time.time()
        aep_fuse = sim_mat  # should be the similarity score
        row_max = np.max(aep_fuse, axis=1)
        Thres = [np.percentile(row_max, 50), np.percentile(row_max, 25), np.percentile(row_max, 1)]
        print(Thres)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        thres = Thres[0]
        x, y = np.where(aep_fuse > thres)
        adMtrx = gen_adMtrx(x, y, aep_fuse)

        allents = set()
        for i in range(aep_fuse.shape[0] + aep_fuse.shape[1]):
            allents.add(i)
        blocks = get_blocks(adMtrx, allents)
        del adMtrx
        del allents
        del x, y
        # evaluation!!!!
        maxtruth = 0
        correct_coun = 0

        all1s, maxtruth, correct_coun = ana_blocks(blocks, aep_fuse, maxtruth, correct_coun, recip_flag)

        rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s, aep_fuse, False,
                                               recip_flag)
        print('Lagrest block... (all 1s): ' + str(len(all1s)))
        del all1s

        print('\n*********************************************************')
        thres2 = Thres[1]
        x, y = np.where(tempM > thres2)
        adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)

        allents = []
        allents.extend(rows)
        for item in columns:
            allents.append(item + aep_fuse.shape[0])
        allents = set(allents)
        newblocks = get_blocks(adMtrx1, allents)
        del adMtrx1
        del allents

        all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag)

        rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, False, recip_flag)
        print('Lagrest block... (all 1s): ' + str(len(all1s_new)))
        del all1s_new

        print('\n**********************************************************')
        thres2 = Thres[2]
        x, y = np.where(tempM > thres2)
        adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)

        allents = []
        allents.extend(rows)
        for item in columns:
            allents.append(item + aep_fuse.shape[0])
        allents = set(allents)

        newblocks = get_blocks(adMtrx1, allents)
        del adMtrx1
        del allents

        all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag)

        rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, True, recip_flag)

        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

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







