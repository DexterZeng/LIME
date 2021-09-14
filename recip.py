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

def reciprocal(sim_mat):
    sim_mat_r = sim_mat.T
    pref = sim_mat - np.max(sim_mat.T, axis=1) + 1
    pref_r = sim_mat_r - np.max(sim_mat_r.T, axis=1) + 1

    ranks = rankdata(-pref, axis=1)
    ranks_r = rankdata(-pref_r, axis=1)
    # ARITHMETIC MEAN
    rankfused = (ranks + ranks_r.T) / 2
    return rankfused


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

def eva(sim_mat, use_min = False):
    if use_min is True:
        predicted = np.argmin(sim_mat, axis=1)
    else:
        predicted = np.argmax(sim_mat, axis=1)
    cor = predicted == np.array(range(sim_mat.shape[0]))
    cor_num = np.sum(cor)
    print("Acc: " + str(cor_num) + ' / ' + str(len(cor)) + ' = ' + str(cor_num*1.0/len(cor)))

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

        pref = sim_mat - np.max(sim_mat.T, axis=1) +1
        pref_r = sim_mat_r - np.max(sim_mat_r.T, axis=1) +1

        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        recip_sim = (pref + pref_r.T) / 2.0
        eva(recip_sim)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    elif args.method == "ralign":
        sim_mat_r = sim_mat.T

        pref = sim_mat - np.max(sim_mat.T, axis=1) + 1
        pref_r = sim_mat_r - np.max(sim_mat_r.T, axis=1) + 1

        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        # recip_sim = (pref + pref_r.T) / 2.0
        # eva(recip_sim)
        # print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        ranks = rankdata(-pref, method="average", axis=1) # ordinal dense min max average
        ranks_r = rankdata(-pref_r, method="average", axis=1)
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

        rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s, aep_fuse, False, recip_flag)
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







