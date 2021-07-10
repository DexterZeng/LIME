import time
from partition import Partition
from collections import defaultdict
import torch
from torch import Tensor
import numpy as np
from typing import *
from utils import add_cnt_for, get_batch_sim, ind2sparse
from model import ModelWrapper


class AlignmentBatch:
    def __init__(self, triple1, triple2, src_nodes, trg_nodes, train_pairs, test_pairs, real_test_pairs, backbone='gcn-align'):
        self.backbone = backbone
        self.merge = True
        # print("Batch info: ", '\n\t'.join(map(lambda x, y: '='.join(map(str, [x, y])),
        print("\nBatch info: ", ' '.join(map(lambda x, y: '='.join(map(str, [x, y])),
                                              ['triple1', 'triple2', 'srcNodes', 'trgNodes',
                                               'trainPairs', 'testPairs'],
                                              map(len,
                                                  [triple1, triple2, src_nodes, trg_nodes, train_pairs, test_pairs]))))
        # try:
        self.ent_maps, self.rel_maps, ent_ids, rel_ids, \
        [t1, t2, train_ill, test_ill] = rearrange_ids([src_nodes, trg_nodes], self.merge,
                                                      triple1, triple2, train_pairs, test_pairs)
        # dev_split = int(dev_ratio * len(train_ill))
        # train_ill, dev_ill = train_ill[dev_split:], train_ill[:dev_split]
        self.test_ill = test_ill
        self.train_ill = train_ill
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.len_src, self.len_trg = len(src_nodes), len(trg_nodes)
        self.shift = len(src_nodes)
        self.src_nodes, self.trg_nodes = src_nodes, trg_nodes
        self.assoc = make_assoc(self.ent_maps, self.len_src, self.len_trg, self.merge)

        # train_test_source = set([item[0] for item in train_pairs])
        # train_test_target = set([item[1] for item in train_pairs])
        #
        # self.test_source = [item[0] for item in test_pairs if item[0] not in train_test_source]
        # self.test_target = [item[1] for item in test_pairs if item[1] not in train_test_target]

        self.test_source = [item[0] for item in real_test_pairs]
        self.test_target = [item[1] for item in real_test_pairs]

        self.test_source_matrixid = [self.ent_maps[0][item] for item in self.test_source]
        self.test_target_matrixid = [self.ent_maps[1][item] - self.shift for item in self.test_target]

        if self.backbone == 'rrea' or self.backbone == 'gcn-align':
            self.model = ModelWrapper(self.backbone,
                                      triples=t1 + t2,
                                      link=torch.tensor(test_ill).t(),
                                      ent_sizes=[len(ids) for ids in ent_ids],
                                      rel_sizes=[x for x in map(len, rel_ids)],
                                      device='cuda',
                                      dim=200,
                                      )

            self.model.update_trainset(np.array(self.train_ill).T)
        else:
            raise NotImplementedError


    @staticmethod
    def get_ei(triple):
        return torch.tensor([[t[0], t[-1]] for t in triple]).t()

    @staticmethod
    def get_et(triple):
        return torch.tensor([t[1] for t in triple])

    @property
    def test_set(self):
        return torch.tensor(self.test_pairs).t()

    @property
    def train_set(self):
        return torch.tensor(self.train_pairs).t()

    @torch.no_grad()
    def get_sim_mat(self, all_embeds, size):
        # print(self.shift)
        if isinstance(all_embeds, tuple):
            embeds = all_embeds
        else:
            embeds = [all_embeds[:self.shift], all_embeds[self.shift:]]

        embeds = list(embeds)
        # print(embeds[0].size(), embeds[1].size())
        embeds[0] = embeds[0][self.test_source_matrixid]
        embeds[1] = embeds[1][self.test_target_matrixid]
        print(embeds[0].size(), embeds[1].size())

        ind, val = get_batch_sim(embeds)

        self.test_source_matrixid = torch.tensor(self.test_source_matrixid)
        self.test_target_matrixid = torch.tensor(self.test_target_matrixid)

        ind_1 = torch.stack(
            [self.test_source_matrixid[ind[0]],
             self.test_target_matrixid[ind[1]]]
        )

        ind_new = torch.stack(
            [self.assoc[ind_1[0]],
             self.assoc[ind_1[1] + self.shift]]
        )
        # return ind2sparse(ind, size, values=val)
        return ind2sparse(ind_new, size, values=val), ind2sparse(ind, (len(embeds[0]), len(embeds[1])), values=val)#, self.assoc


def rearrange_ids(nodes, merge: bool, *to_map):
    ent_mappings = [{}, {}]
    rel_mappings = [{}, {}]
    ent_ids = [[], []]
    shift = 0
    for w, node_set in enumerate(nodes):
        for n in node_set:
            ent_mappings[w], nn, shift = add_cnt_for(ent_mappings[w], n, shift)
            ent_ids[w].append(nn)
        shift = len(ent_ids[w]) if merge else 0
    mapped = []
    shift = 0
    curr = 0
    for i, need in enumerate(to_map):
        now = []
        if len(need) == 0:
            mapped.append([])
            continue
        is_triple = len(need[0]) == 3
        for tu in need:
            if is_triple:
                h, t = ent_mappings[curr][tu[0]], ent_mappings[curr][tu[-1]]
                rel_mappings[curr], r, shift = add_cnt_for(rel_mappings[curr], tu[1], shift)
                now.append((h, r, t))
            else:
                now.append((ent_mappings[0][tu[0]], ent_mappings[1][tu[-1]]))
        mapped.append(now)
        curr += is_triple
        if not merge:
            shift = 0
    rel_ids = [list(rm.values()) for rm in rel_mappings]

    return ent_mappings, rel_mappings, ent_ids, rel_ids, mapped

def make_assoc(maps, src_len, trg_len, merge):
    assoc = np.empty(src_len + trg_len, dtype=np.int)
    shift = 0 if merge else 1
    shift = shift * src_len
    for idx, ent_mp in enumerate(maps):
        for k, v in ent_mp.items():
            assoc[v + idx * shift] = k
    return torch.tensor(assoc)

def overlaps(src: List[set], trg: List[set]):
    return np.array([[float(len(s.intersection(t))) / (float(len(s)) + 0.01) for t in trg] for s in src])

def place_triplets(triplets, nodes_batch): # after divide the nodes, place the triples!!
    batch = defaultdict(list)
    node2batch = {}
    for i, nodes in enumerate(nodes_batch):
        for n in nodes:
            node2batch[n] = i
    removed = 0
    for h, r, t in triplets:
        h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
        if h_batch == t_batch and h_batch >= 0:
            batch[h_batch].append((h, r, t))
        else:
            removed += 1
    print('split triplets complete, total {} triplets removed'.format(removed))

    return batch, removed

def make_pairs(src, trg, mp):
    return list(filter(lambda p: p[1] in trg, [(e, mp[e]) for e in set(filter(lambda x: x in mp, src))]))

def gen_partition(corr_ind_1, src_nodes_1, trg_nodes_1, src_train_1, trg_train_1, corr_val_1, mapping_1, triple1_batch_1, triple2_batch_1):
    train_pair_cnt = 0
    test_pair_cnt = 0

    IDs_s_1 = []
    IDs_t_1 = []
    Trains_s_1 = []
    Trains_t_1 = []
    Triples_s_1 = []
    Triples_t_1 = []
    for src_id, src_corr in enumerate(corr_ind_1):
        ids1_1, train1_1 = src_nodes_1[src_id], src_train_1[src_id]
        train2_1, ids2_1, triple2_1 = [], [], []
        corr_rate = 0.
        for trg_rank, trg_id in enumerate(src_corr):
            train2_1 += trg_train_1[trg_id]
            ids2_1 += trg_nodes_1[trg_id]
            triple2_1 += triple2_batch_1[trg_id]
            corr_rate += corr_val_1[src_id][trg_rank]
        ids1_1, ids2_1, train1_1, train2_1 = map(set, [ids1_1, ids2_1, train1_1, train2_1])

        IDs_s_1.append(ids1_1)
        IDs_t_1.append(ids2_1)
        Trains_s_1.append(train1_1)
        Trains_t_1.append(train2_1)
        Triples_s_1.append(set(triple1_batch_1[src_id]))
        Triples_t_1.append(set(triple2_1))

        print('Train corr=', corr_rate)

        train_pairs = make_pairs(train1_1, train2_1, mapping_1)
        train_pair_cnt += len(train_pairs)
        test_pairs = make_pairs(ids1_1, ids2_1, mapping_1)
        test_pair_cnt += len(test_pairs)

    print("*************************************************************")
    print("Total trainig pairs: " + str(train_pair_cnt))
    print("Total testing pairs: " + str(test_pair_cnt - train_pair_cnt))
    print("Total links: " + str(test_pair_cnt))
    print("*************************************************************")

    return IDs_s_1, IDs_t_1, Trains_s_1, Trains_t_1, Triples_s_1, Triples_t_1

def batch_sampler_consistency(data, args, semi_round, src_split=30, trg_split=100, top_k_corr=5, which=0, share_triples=True, backbone='rrea', random=False, **kwargs):

    time_now = time.time()
    metis = Partition(data)

    print("\n*************************************************************")
    print("Partition left 2 right: ")
    print("*************************************************************")

    src_nodes_1, trg_nodes_1, src_train_1, trg_train_1 = metis.random_partition(which, src_split, trg_split, share_triples) \
        if random else metis.partition(which, src_split, trg_split, share_triples)

    triple1_batch_1, removed1_1 = place_triplets(data.triples[which], src_nodes_1)
    triple2_batch_1, removed2_1 = place_triplets(data.triples[1 - which], trg_nodes_1)

    corr_1 = torch.from_numpy(overlaps(
        [set(metis.train_map[which][i] for i in s) for s in src_train_1],
        [set(s) for s in trg_train_1]
    ))

    mapping_1= metis.train_map[which]
    corr_val_1, corr_ind_1= map(lambda x: x.numpy(), corr_1.topk(top_k_corr))

    # corr_ind = corr_ind.numpy()
    print('partition complete, time=', time.time() - time_now)

    IDs_s_1, IDs_t_1, Trains_s_1, Trains_t_1, Triples_s_1, Triples_t_1 = gen_partition(corr_ind_1, src_nodes_1, trg_nodes_1, src_train_1, trg_train_1, corr_val_1, mapping_1, triple1_batch_1, triple2_batch_1)

    print("\n*************************************************************")
    print("Partition right 2 left: ")
    print("*************************************************************")

    trg_nodes_2, src_nodes_2, trg_train_2, src_train_2 = metis.random_partition(1 - which, src_split, trg_split, share_triples) \
        if random else metis.partition(1 - which, src_split, trg_split, share_triples)

    triple1_batch_2, removed1_2 = place_triplets(data.triples[which], src_nodes_2)
    triple2_batch_2, removed2_2 = place_triplets(data.triples[1 - which], trg_nodes_2)

    corr2 = torch.from_numpy(overlaps(
        [set(metis.train_map[which][i] for i in s) for s in src_train_2], # no change here
        [set(s) for s in trg_train_2]
    ))

    mapping_2 = metis.train_map[which] # converted for corr2, so here might still use this mapping from source 2 target
    corr_val_2, corr_ind_2 = map(lambda x: x.numpy(), corr2.topk(top_k_corr))
    print('partition complete, time=', time.time() - time_now)

    IDs_s_2, IDs_t_2, Trains_s_2, Trains_t_2, Triples_s_2, Triples_t_2 = gen_partition(corr_ind_2, src_nodes_2, trg_nodes_2, src_train_2,
                                                             trg_train_2, corr_val_2, mapping_2, triple1_batch_2, triple2_batch_2)


    print("\n*************************************************************")
    print("Combination: ")
    print("*************************************************************")

    corr_3 = torch.from_numpy(overlaps(
        [set(s) for s in src_train_1],
        [set(s) for s in src_train_2]
    ))

    corr_val_3, corr_ind_3 = map(lambda x: x.numpy(), corr_3.topk(top_k_corr))

    train_pair_cnt = 0
    test_pair_cnt = 0
    real_test_pairs_cnt = 0
    real_train_pairs_cnt = 0
    train_pair_unq = []
    test_pair_unq = []

    real_test_sourceids = set([item[0] for item in data.test])

    for src1_id, src1_corr in enumerate(corr_ind_3):
        ids_s_1, trains_s_1, triples_s_1 = IDs_s_1[src1_id], Trains_s_1[src1_id], Triples_s_1[src1_id]
        ids_t_1, trains_t_1, triples_t_1 = IDs_t_1[src1_id], Trains_t_1[src1_id], Triples_t_1[src1_id]

        corr_rate = 0.
        for src2_rank, src2_id in enumerate(src1_corr):
            ids_s_2, trains_s_2, triples_s_2 = IDs_s_2[src2_id], Trains_s_2[src2_id], Triples_s_2[src2_id]
            ids_t_2, trains_t_2, triples_t_2 = IDs_t_2[src2_id], Trains_t_2[src2_id], Triples_t_2[src2_id]
            corr_rate += corr_val_3[src1_id][src2_rank]

            ids_s_1 = ids_s_1.union(ids_s_2)
            ids_t_1 = ids_t_1.union(ids_t_2)
            trains_s_1 = trains_s_1.union(trains_s_2)
            trains_t_1 = trains_t_1.union(trains_t_2)
            triples_s_1 = triples_s_1.union(triples_s_2)
            triples_t_1 = triples_t_1.union(triples_t_2)


        # print('Train corr=', corr_rate)

        train_pairs = make_pairs(trains_s_1, trains_t_1, mapping_1)
        train_pair_cnt += len(train_pairs)
        test_pairs = make_pairs(ids_s_1, ids_t_1, mapping_1)
        test_pair_cnt += len(test_pairs)


        real_train_pairs = [item for item in train_pairs if item[0] not in real_test_sourceids]
        real_train_pairs_cnt += len(real_train_pairs)

        real_test_pairs = [item for item in test_pairs if item[0] in real_test_sourceids]
        real_test_pairs_cnt += len(real_test_pairs)

        train_pair_unq.extend(real_train_pairs)
        test_pair_unq.extend(real_test_pairs)

        if (args.offline is True and (args.use_semi is False or (args.use_semi is True and semi_round ==0))) or (args.use_semi is True and semi_round > 0):
            yield [list(triples_s_1), list(triples_t_1), ids_s_1, ids_t_1, train_pairs, test_pairs, real_test_pairs, backbone]
        else:
            yield AlignmentBatch(list(triples_s_1), list(triples_t_1), ids_s_1, ids_t_1, train_pairs, test_pairs, real_test_pairs, backbone=backbone)

        # print(str(len(ids_s_1)) + '\t' + str(len(ids_t_1)))
        # print(len(train_pairs))
        # print(len(test_pairs) - len(train_pairs))
        # ids1_2, ids2_2, train1_2, train2_2 = map(set, [ids1_2, ids2_2, train1_2, train2_2])

    print("\n*************************************************************")
    print("Total trainig pairs: " + str(train_pair_cnt))
    print("Real trainig pairs: " + str(real_train_pairs_cnt))
    print("Real testing pairs: " + str(real_test_pairs_cnt))
    print("Total links: " + str(test_pair_cnt))
    train_pair_unq = set(train_pair_unq)
    test_pair_unq = set(test_pair_unq)
    print("Real trainig pairs uniq: " + str(len(train_pair_unq)))
    print("Real testing pairs uniq: " + str(len(test_pair_unq)))
    print("Total links uniq: " + str(len(test_pair_unq) + len(train_pair_unq)))
    print("*************************************************************\n")



def batch_sampler(data, args, semi_round, src_split=30, trg_split=100, top_k_corr=5, which=0, share_triples=True,
                  backbone='rrea', random=False, **kwargs):

    # which = 1- which

    time_now = time.time()
    metis = Partition(data)

    src_nodes, trg_nodes, src_train, trg_train = metis.random_partition(which, src_split, trg_split, share_triples) \
        if random else metis.partition(which, src_split, trg_split, share_triples)

    triple1_batch, removed1 = place_triplets(data.triples[which], src_nodes)
    triple2_batch, removed2 = place_triplets(data.triples[1 - which], trg_nodes)

    corr = torch.from_numpy(overlaps(
        [set(metis.train_map[which][i] for i in s) for s in src_train],
        [set(s) for s in trg_train]
    ))

    mapping = metis.train_map[which]
    corr_val, corr_ind = map(lambda x: x.numpy(), corr.topk(top_k_corr)) # generate the corresponding blocks!!!

    # corr_ind = corr_ind.numpy()
    print('partition complete, time=', time.time() - time_now)

    train_pair_cnt = 0
    test_pair_cnt = 0
    real_test_pairs_cnt = 0
    real_train_pairs_cnt = 0

    # test
    real_test_sourceids = set([item[0] for item in data.test])

    for src_id, src_corr in enumerate(corr_ind):
        ids1, train1 = src_nodes[src_id], src_train[src_id]
        train2, ids2, triple2 = [], [], []
        corr_rate = 0.
        for trg_rank, trg_id in enumerate(src_corr):
            train2 += trg_train[trg_id]
            ids2 += trg_nodes[trg_id]
            triple2 += triple2_batch[trg_id]
            corr_rate += corr_val[src_id][trg_rank]
        ids1, ids2, train1, train2 = map(set, [ids1, ids2, train1, train2])
        print('Train corr=', corr_rate)

        train_pairs = make_pairs(train1, train2, mapping)
        train_pair_cnt += len(train_pairs)
        test_pairs = make_pairs(ids1, ids2, mapping)
        test_pair_cnt += len(test_pairs)

        # real_test_pairs: those in the test pairs that are not original trianing!! not necesarritly 700000

        real_train_pairs = [item for item in train_pairs if item[0] not in real_test_sourceids]
        real_train_pairs_cnt += len(real_train_pairs)

        real_test_pairs = [item for item in test_pairs if item[0] in real_test_sourceids]
        real_test_pairs_cnt += len(real_test_pairs)


        if args.offline is True and (args.use_semi is False or (args.use_semi is True and semi_round ==0)):
            yield [triple1_batch[src_id], triple2, ids1, ids2, train_pairs, test_pairs, real_test_pairs, backbone]
        else:
            yield AlignmentBatch(triple1_batch[src_id], triple2, ids1, ids2, train_pairs, test_pairs, real_test_pairs, backbone=backbone)

    print("\n*************************************************************")
    print("Total trainig pairs: " + str(train_pair_cnt))
    print("Real trainig pairs: " + str(real_train_pairs_cnt))
    print("Real testing pairs: " + str(real_test_pairs_cnt))
    print("Total links: " + str(test_pair_cnt))
    print("*************************************************************\n")