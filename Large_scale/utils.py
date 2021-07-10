import numpy as np
from collections import defaultdict
import torch
from torch import Tensor
from torch_scatter import scatter, scatter_max, scatter_min, scatter_add
import time
import gc
import faiss


def argprint(**kwargs):
    return '\n'.join([str(k) + "=" + str(v) for k, v in kwargs.items()])

def prepare_block_fisrtR(sim_block, batch, theta, shift_value):
    ind_0_ori = sim_block._indices()[0].cpu().detach().numpy()
    ind_1_ori = sim_block._indices()[1].cpu().detach().numpy()
    values = sim_block._values().cpu().detach().numpy()

    ind_0 = batch.assoc[batch.test_source_matrixid[sim_block._indices()[0]]].cpu().detach().numpy()
    ind_1 = (batch.assoc[batch.test_target_matrixid[sim_block._indices()[1]] + batch.shift] + shift_value).cpu().detach().numpy()

    adMtrx = defaultdict(set)
    now2id = dict()
    assert len(ind_0) == len(ind_1)
    for i in range(len(ind_0)):
        if values[i] >= theta:
            adMtrx[ind_0[i]].add(ind_1[i])
            adMtrx[ind_1[i]].add(ind_0[i])

    assoc = batch.assoc.cpu().detach().numpy()
    test_source_matrixid = batch.test_source_matrixid.cpu().detach().numpy()
    test_target_matrixid = batch.test_target_matrixid.cpu().detach().numpy()
    ind_0_ori = list(set(ind_0_ori))
    for i in range(len(ind_0_ori)):
        now2id[assoc[test_source_matrixid[ind_0_ori[i]]]] = ind_0_ori[i]

    ind_1_ori = list(set(ind_1_ori))
    for i in range(len(ind_1_ori)):
        now2id[assoc[test_target_matrixid[ind_1_ori[i]] + batch.shift] + shift_value] = ind_1_ori[i]

    # NOTE THAT SOME TARGET ENTITIES ARE ISOLATED.. AND THEY ARE NOT even IN THE indexes of the SIMILARITY MATRIX?
    # allents = set(batch.src_nodes).union(set(x + shift_value for x in batch.trg_nodes))
    allents = set(batch.test_source).union(set(x + shift_value for x in batch.test_target))

    return adMtrx, allents, now2id


def prepare_followingR(block, sim_block, now2id, theta, assoc, shift, test_source_matrixid, test_target_matrixid, shift_value):
    # block include all the entities... even those not in now2id
    # row to record the matrix indexing ids of the source entities
    # column to record the matrix indexing ids of the target entities
    rows = []
    columns = []
    # print(block)
    for item in block:
        if item < 10000000:
            try:
                rows.append(now2id[item])
            except:
                continue
        else:
            try:
                columns.append(now2id[item])
            except:
                continue

    # print(len(rows), len(columns))

    sim_b = sim_block[rows][:, columns]

    rows = np.array(rows)
    columns = np.array(columns)

    a, b = torch.where(sim_b> theta)
    a = a.cpu().detach().numpy()
    a_ = assoc[test_source_matrixid[rows[a]]]
    b = b.cpu().detach().numpy()
    # print(columns[b])
    b_ = assoc[test_target_matrixid[columns[b]] + shift] + shift_value
    adMtrx = defaultdict(set)
    for i in range(len(a_)):
        adMtrx[a_[i]].add(b_[i])
        adMtrx[b_[i]].add(a_[i])
    return adMtrx


def block_recip(block, sim_block, now2id, assoc, shift, test_source_matrixid, test_target_matrixid,  linking_dic, predicted_result, redun = False, sem = False):
    # block represent the group of entites, including the source and target ones, their original IDs!
    # sim_block represent the dense similarity matrix among the entities in this block
    # now2id represent the mapping from the original ids (not exactly... that adds 1000000 in the target side)
    # to the new ids for indexing the sim_block matrix
    # assoc represents the mapping of ids for indexing the sim_block matrix to the original ids
    # shift is used collectively with assoc, with marks the mapping for the target entities
    # test_source_matrixid, test_target_matrixid, we only use the test ids! convert the testids to the bactch wise ids
    # linking_dic records the ground truth mappings of test entities, which could be used to filter test entities from the block entities..
    # predicted_result records the prediction results (of the test entities)...
    rows = []
    columns = []
    for item in block:
        if item < 10000000:
            try:
                rows.append(now2id[item])
            except:
                continue
        else:
            try:
                columns.append(now2id[item])
            except:
                continue
    del block
    del now2id
    # if rows and columns both contain one or more entities... otherwise it would be meaningless
    # print("In this block, source ents are " + str(len(rows)) + " and the target entities are " + str(len(columns)))
    if len(rows)>=1 and len(columns)>=1:
        # print(len(rows))
        # print(len(columns))
        # now2ori is charizized in rows and columns
        # sim_b = torch.index_select(torch.index_select(sim_block, 0, torch.tensor(rows).to(device)), 1, torch.tensor(columns).to(device))
        sim_b = sim_block[rows][:, columns]
        del sim_block
        # print(sim_b.shape)

        max_value = torch.max(sim_b, dim=0)[0]
        max_value[max_value == 0.0] = 1.0
        a = sim_b - max_value + 1
        max_value = torch.max(sim_b, dim=1)[0]
        b = (torch.transpose(sim_b, 0, 1) - max_value) + 1
        del sim_b
        del max_value
        # print("convert to ranking matrix")
        from scipy.stats import rankdata
        a_rank = rankdata(-a.cpu().detach().numpy(), axis=1)
        del a
        b_rank = rankdata(-b.cpu().detach().numpy(), axis=1)
        del b
        recip_sim = (torch.from_numpy(a_rank) + torch.transpose(torch.from_numpy(b_rank), 0, 1)) / 2.0
        del a_rank
        del b_rank
        # print("start to predict...")
        recip_pred = matrix_argmin(recip_sim).view(-1).cpu().detach().numpy()

        if redun is True:
            recip_pred_value = torch.min(recip_sim, dim=1)[0].cpu().detach().numpy()

            if sem is True:
                predicted_result_ = predicted_result[1]
                predicted_result = predicted_result[0]
                recip_pred_r2l = matrix_argmin(recip_sim, dim=0).view(-1).cpu().detach().numpy()
                recip_pred_r2l_value = torch.min(recip_sim, dim=0)[0].cpu().detach().numpy()
            del recip_sim

            for i in range(len(rows)):
                rowid_ori = assoc[test_source_matrixid[rows[i]]]  # note that the rowid include both training and testing dataset
                # if rowid_ori in linking_dic:
                predicted = assoc[test_target_matrixid[columns[recip_pred[i]]] + shift]  # for the target side
                predicted_result[rowid_ori][predicted] = recip_pred_value[i]
                # else:
                #     print("fxx")
            if sem is True:
                for i in range(len(columns)):
                    columnid_ori = assoc[test_target_matrixid[columns[i]] + shift]
                    predicted = assoc[test_source_matrixid[rows[recip_pred_r2l[i]]]]
                    predicted_result_[columnid_ori][predicted] = recip_pred_r2l_value[i]

        else:
            if sem is True:
                predicted_result_ = predicted_result[1]
                predicted_result = predicted_result[0]
                recip_pred_r2l = matrix_argmin(recip_sim, dim=0).view(-1).cpu().detach().numpy()

            del recip_sim

            for i in range(len(rows)):
                rowid_ori = assoc[test_source_matrixid[rows[i]]]  # note that the rowid include both training and testing dataset
                # if rowid_ori in linking_dic:
                predicted = assoc[test_target_matrixid[columns[recip_pred[i]]] + shift]  # for the target side
                predicted_result[rowid_ori].append(predicted)
                # else:
                #     print("fxx")
            if sem is True:
                for i in range(len(columns)):
                    columnid_ori = assoc[test_target_matrixid[columns[i]] + shift]
                    predicted = assoc[test_source_matrixid[rows[recip_pred_r2l[i]]]]
                    predicted_result_[columnid_ori].append(predicted)

    if sem is True:
        return [predicted_result, predicted_result_]
    else:
        return predicted_result


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

def get_blocks(adMtrx, allents, args):
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
    if args.use_semi is False:
        print('Total blocks: ' + str(len(blocks)))
        # print(lenghs)
        print('Total blocks with length 1: ' + str(count1))
        print('Maxlen: ' + str(np.max(np.array(lenghs))))
        # print(count1)
        # print(lenghs[0])
    return blocks

def sparse_min(tensor: Tensor, dim=-1):
    tensor = tensor.coalesce()
    return scatter_min(tensor._values(), tensor._indices()[dim], dim_size=tensor.size(dim))

def sparse_max(tensor: Tensor, dim=-1):
    tensor = tensor.coalesce()
    return scatter_max(tensor._values(), tensor._indices()[dim], dim_size=tensor.size(dim))

def sparse_argmin(tensor, scatter_dim, dim=0):
    tensor = tensor.coalesce()
    return tensor._indices()[scatter_dim][sparse_min(tensor, dim)[1]]

def sparse_argmax(tensor, scatter_dim, dim=0):
    tensor = tensor.coalesce()
    argmax = sparse_max(tensor, dim)[1]
    argmax[argmax == tensor._indices().size(1)] = 0
    return tensor._indices()[scatter_dim][argmax]

def matrix_argmin(tensor: Tensor, dim=1):
    assert tensor.dim() == 2
    if tensor.is_sparse:
        return sparse_argmin(tensor, dim, 1 - dim)
    else:
        return torch.argmin(tensor, dim)

def matrix_argmax(tensor: Tensor, dim=1):
    assert tensor.dim() == 2
    if tensor.is_sparse:
        return sparse_argmax(tensor, dim, 1 - dim)
    else:
        return torch.argmax(tensor, dim)


def topk2spmat(val0, ind0, size, dim=0, device: torch.device = 'cuda', split=False):
    if isinstance(val0, np.ndarray):
        val0, ind0 = torch.from_numpy(val0).to(device), \
                     torch.from_numpy(ind0).to(device)

    if split:
        return val0, ind0, size

    ind_x = torch.arange(size[dim]).to(device)
    ind_x = ind_x.view(-1, 1).expand_as(ind0).reshape(-1)
    ind_y = ind0.reshape(-1)
    ind = torch.stack([ind_x, ind_y])
    val0 = val0.reshape(-1)
    filter_invalid = torch.logical_and(ind[0] >= 0, ind[1] >= 0)
    ind = ind[:, filter_invalid]
    val0 = val0[filter_invalid]
    return ind2sparse(ind, list(size), values=val0).coalesce()

def ind2sparse(indices: Tensor, size, size2=None, dtype=torch.float, values=None):
    device = indices.device
    if isinstance(size, int):
        size = (size, size if size2 is None else size2)

    assert indices.dim() == 2 and len(size) == indices.size(0)
    if values is None:
        values = torch.ones([indices.size(1)], device=device, dtype=dtype)
    else:
        assert values.dim() == 1 and values.size(0) == indices.size(1)
    return torch.sparse_coo_tensor(indices, values, size)

def set_seed(seed):
    if seed:
        import random
        import numpy
        import torch
        numpy.random.seed(seed)
        random.seed(seed)
        torch.random.manual_seed(seed)

def apply(func, *args):
    if func is None:
        func = lambda x: x
    lst = []
    for arg in args:
        lst.append(func(arg))
    return tuple(lst)

def norm_process(embed: torch.Tensor, eps=1e-5) -> torch.Tensor:
    n = embed.norm(dim=1, p=2, keepdim=True)
    embed = embed / (n + eps)
    return embed

def add_cnt_for(mp, val, begin=None):
    if begin is None:
        if val not in mp:
            mp[val] = len(mp)
        return mp, mp[val]
    else:
        if val not in mp:
            mp[val] = begin
            begin += 1
        return mp, mp[val], begin



def faiss_search_impl(emb_q, emb_id, emb_size, shift, k=50, search_batch_sz=50000, gpu=True):
    index = faiss.IndexFlat(emb_size)
    if gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(emb_id)
    # print('Total index =', index.ntotal)
    vals, inds = [], []
    # for i_batch in tqdm(range(0, len(emb_q), search_batch_sz)):
    for i_batch in range(0, len(emb_q), search_batch_sz):
        val, ind = index.search(emb_q[i_batch:min(i_batch + search_batch_sz, len(emb_q))], k)
        val = torch.from_numpy(val)
        val = 1 - val
        vals.append(val)
        inds.append(torch.from_numpy(ind) + shift)
        # print(vals[-1].size())
        # print(inds[-1].size())
    del index, emb_id, emb_q
    vals, inds = torch.cat(vals), torch.cat(inds)
    return vals, inds

@torch.no_grad()
def global_level_semantic_sim(embs, k=50, search_batch_sz=50000, index_batch_sz=500000
                              , split=False, norm=True, gpu=True):
    # print('FAISS number of GPUs=', faiss.get_num_gpus())
    size = [embs[0].size(0), embs[1].size(0)]
    emb_size = embs[0].size(1)
    if norm:
        embs = apply(norm_process, *embs)
    emb_q, emb_id = apply(lambda x: x.cpu().numpy(), *embs)
    del embs
    gc.collect()
    vals, inds = [], []
    total_size = emb_id.shape[0]
    for i_batch in range(0, total_size, index_batch_sz):
        i_end = min(total_size, i_batch + index_batch_sz)
        val, ind = faiss_search_impl(emb_q, emb_id[i_batch:i_end], emb_size, i_batch, k, search_batch_sz, gpu)
        vals.append(val)
        inds.append(ind)

    vals, inds = torch.cat(vals, dim=1), torch.cat(inds, dim=1)
    # print(vals.size(), inds.size())

    return topk2spmat(vals, inds, size, 0, torch.device('cpu'), split)

def get_batch_sim(embed, topk=50, split=True):
    spmat = global_level_semantic_sim(embed, k=topk, gpu=False).to(embed[0].device)
    if split:
        return spmat._indices(), spmat._values()
    else:
        return spmat

def sparse_acc_direct(sp_sim: Tensor, link: Tensor, t_total, use_semi, device='cpu'):
    # approx version 1, which still oprerates on the sprs matrix
    print('Total link is', link.size(1))
    sp_sim, link = apply(lambda x: x.to(device), sp_sim, link) # unify to the same machine
    print(sp_sim.size(), sp_sim._indices().size(), sp_sim._values().size())

    pred = matrix_argmax(sp_sim).view(-1)
    acc: Tensor = pred[link[0]] == link[1]
    print('DIRECT acc is ' + str(acc.sum().item()) + ' / ' + str(link.size(1)) + ' = ' + str((acc.sum() / acc.numel()).item()))
    print("DIRECT time elapsed: {:.4f} s".format(time.time() - t_total))

    if use_semi is False:
        return None
    else:
        print('\nright2left::')
        pred_r2l = matrix_argmax(sp_sim, dim=0).view(-1)
        acc_r2l: Tensor = pred_r2l[link[1]] == link[0]
        print('acc is ' + str(acc_r2l.sum().item()) + ' / ' + str(link.size(1)) + ' = ' + str((acc_r2l.sum() / acc_r2l.numel()).item()))

        acc_confi: Tensor = pred_r2l[pred[link[0]]] == link[0]  ## note it is not entirely correct
        acc_confi_correct: Tensor = (acc_confi * acc) == 1

        # print(acc_confi.shape)
        lefts = link[0][acc_confi]
        rights = pred[lefts]
        assert len(lefts) == len(rights)
        new_semi_pairs = np.array([[lefts[i], rights[i]] for i in range(len(lefts))])

        print('acc confi is ' + str(acc_confi_correct.sum().item()) + ' / ' + str(acc_confi.sum().item()) + ' = ' + str(
            (acc_confi_correct.sum() / acc_confi.sum()).item()))

        return new_semi_pairs

@torch.no_grad()
def sparse_acc_recip_approx1(sp_sim: Tensor, link: Tensor, t_total, use_semi, device='cpu'):
    # approx version 1, which still oprerates on the sprs matrix
    print('Total link is', link.size(1))
    sp_sim, link = apply(lambda x: x.to(device), sp_sim, link) # unify to the same machine
    print(sp_sim.size(), sp_sim._indices().size(), sp_sim._values().size())

    pred = matrix_argmax(sp_sim).view(-1)
    acc: Tensor = pred[link[0]] == link[1]
    print('DIRECT acc is ' + str(acc.sum().item()) + ' / ' + str(link.size(1)) + ' = ' + str((acc.sum() / acc.numel()).item()))
    print("DIRECT time elapsed: {:.4f} s".format(time.time() - t_total))

    ## recoprocal modeling
    # t_total = time.time()
    max_value = sparse_max(sp_sim, dim=0)[0]
    max_value_sp = torch.sparse_coo_tensor(sp_sim._indices(), -max_value[sp_sim._indices()[0]] + 1, sp_sim.size())
    a = sp_sim + max_value_sp
    del max_value
    del max_value_sp

    max_value_1 = sparse_max(sp_sim, dim=1)[0]
    max_value_sp_1 = torch.sparse_coo_tensor(sp_sim._indices(), -max_value_1[sp_sim._indices()[1]] + 1, sp_sim.size())
    # print(max_value_sp_1)
    # generate preference matrix
    a_1 = sp_sim + max_value_sp_1
    del max_value_1
    del max_value_sp_1
    # print(a_1)

    # print("For dividing wn norm and norm.... Time elapsed: {:.4f} s".format(time.time() - t_total))
    recip_sim = (a + a_1)/2.0
    del a
    del a_1
    recip_pred = matrix_argmax(recip_sim).view(-1)
    # del recip_sim
    acc: Tensor = recip_pred[link[0]] == link[1]
    print('Recip (worank) acc is ' + str(acc.sum().item()) + ' / ' + str(link.size(1)) + ' = '+ str((acc.sum() / acc.numel()).item()))

    if use_semi is False:
        return None
    else:
        print('\nright2left::')
        recip_pred_r2l = matrix_argmax(recip_sim, dim=0).view(-1)
        acc_r2l: Tensor = recip_pred_r2l[link[1]] == link[0]
        print('acc is ' + str(acc_r2l.sum().item()) + ' / ' + str(link.size(1)) + ' = ' + str((acc_r2l.sum() / acc_r2l.numel()).item()))

        acc_confi: Tensor = recip_pred_r2l[recip_pred[link[0]]] == link[0]  ## note it is not entirely correct
        acc_confi_correct: Tensor = (acc_confi * acc) == 1

        # print(acc_confi.shape)
        lefts = link[0][acc_confi]
        rights = recip_pred[lefts]
        assert len(lefts) == len(rights)
        new_semi_pairs = [[lefts[i], rights[i]] for i in range(len(lefts))]

        print('\nacc confi is ' + str(acc_confi_correct.sum().item()) + ' / ' + str(acc_confi.sum().item()) + ' = ' + str((acc_confi_correct.sum() / acc_confi.sum()).item()))
        return np.array(new_semi_pairs)


@torch.no_grad()
def sparse_acc_recip(sp_sim: Tensor, link: Tensor, use_semi, device='cpu'):
    # The full version! including the ranking process
    print('Total link is', link.size(1))
    sp_sim, link = apply(lambda x: x.to(device), sp_sim, link) # unify to the same machine
    # print(sp_sim)
    # evaluate as dense matrix

    sim = sp_sim.to_dense()
    print(sim.shape)

    sim = sim[link[0]][:,link[1]]
    print(sim.shape)
    del sp_sim
    # print(sim)

    recip_pred = matrix_argmax(sim).view(-1)
    acc: Tensor = recip_pred == torch.from_numpy(np.arange(len(recip_pred)))
    print('DIR acc is ' + str(acc.sum().item()) + ' / ' + str(link.size(1)) + ' = ' + str((acc.sum() / acc.numel()).item()))

    max_value = torch.max(sim, dim=0)[0]
    max_value[max_value==0.0] = 1.0
    a = sim - max_value + 1
    from scipy.stats import rankdata
    a_rank = rankdata(-a.cpu().detach().numpy(), axis=1)
    del a

    max_value = torch.max(sim, dim=1)[0]
    max_value[max_value == 0.0] = 1.0
    b = (torch.transpose(sim, 0, 1) - max_value) + 1
    b_rank = rankdata(-b.cpu().detach().numpy(), axis=1)

    del max_value
    del sim
    del b

    recip_sim = (torch.from_numpy(a_rank).to(device) + torch.transpose(torch.from_numpy(b_rank).to(device), 0, 1)) / 2.0
    del a_rank
    del b_rank

    recip_pred = matrix_argmin(recip_sim).view(-1)
    # acc: Tensor = recip_pred[link[0]] == link[1]
    acc: Tensor = recip_pred == torch.from_numpy(np.arange(len(recip_pred)))
    print('acc is ' + str(acc.sum().item()) + ' / ' + str(link.size(1)) + ' = ' + str((acc.sum() / acc.numel()).item()))

    if use_semi is False:
        return None
    else:
        print('\nright2left::')
        recip_pred_r2l = matrix_argmin(recip_sim, dim=0).view(-1)
        # acc_r2l: Tensor = recip_pred_r2l[link[1]] == link[0]
        acc_r2l: Tensor = recip_pred_r2l == torch.from_numpy(np.arange(len(recip_pred_r2l)))
        print('acc is ' + str(acc_r2l.sum().item()) + ' / ' + str(link.size(1)) + ' = ' + str((acc_r2l.sum() / acc_r2l.numel()).item()))

        # acc_confi: Tensor = recip_pred_r2l[recip_pred[link[0]]] == link[0]  ## note it is not entirely correct
        acc_confi: Tensor = recip_pred_r2l[recip_pred] == torch.from_numpy(np.arange(len(recip_pred)))
        acc_confi_correct: Tensor = (acc_confi * acc) == 1

        # print(acc_confi.shape)
        lefts = link[0][acc_confi]
        rights = link[1][recip_pred[acc_confi]]
        assert len(lefts) == len(rights)
        new_semi_pairs = [[lefts[i], rights[i]] for i in range(len(lefts))]

        print('acc confi is ' + str(acc_confi_correct.sum().item()) + ' / ' + str(acc_confi.sum().item()) + ' = ' + str((acc_confi_correct.sum() / acc_confi.sum()).item()))
        return np.array(new_semi_pairs)