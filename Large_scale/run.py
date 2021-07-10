# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'

# import text_sim as text
from dataset import *
import torch
from collections import defaultdict
from tqdm import tqdm
import argparse, logging, random, time
from sampler import batch_sampler_consistency, AlignmentBatch, batch_sampler
import time
import copy
from utils import prepare_block_fisrtR, get_blocks, block_recip, prepare_followingR, set_seed,sparse_acc_direct,sparse_acc_recip, sparse_acc_recip_approx1


import warnings
warnings.filterwarnings("ignore")

# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

def train(batch: AlignmentBatch, device: torch.device = 'cuda', **kwargs):
    args = kwargs['args']

    if hasattr(batch, 'skip'):
        return None
    elif hasattr(batch, 'model'):
        model = batch.model
        # try:
        ttt = time.time()
        for it in range(args.it_round):
            model.train1step(args.epoch)
            if it < args.it_round - 1:
                print("?????????????????????????????????????????????")
                model.mraea_iteration()
            print("structural training time elapsed: {:.4f} s".format(time.time() - ttt))
        return model.get_curr_embeddings('cpu')
    else:
        raise NotImplementedError


def run_each_batch(size, batch, device, args, curr_sim, linking_dic, predicted_result, redun=False):
    embed = train(batch, device, args=copy.deepcopy(args))
    if embed is None:
        print('batch skipped')
    sim, sim_block = batch.get_sim_mat(embed, size)

    if args.approx2 == 0:
        curr_sim = sim if curr_sim is None else curr_sim + sim
    else:
        del sim
        THETA = [0.8, 0.4, 0.2, 0.1]
        shift_value = 10000000  # differenciate the values in the source and target
        t_pblock = time.time()
        # Note that blocking is only for test entities!!! change
        # print('Construct graph for blocking...')
        adMtrx, allents, now2id = prepare_block_fisrtR(sim_block, batch, THETA[0], shift_value)
        assoc = batch.assoc.cpu().detach().numpy()
        test_source_matrixid = batch.test_source_matrixid.cpu().detach().numpy()
        test_target_matrixid = batch.test_target_matrixid.cpu().detach().numpy()
        shift = batch.shift
        del batch

        print("Finish preparing for blocks: {:.4f} s".format(time.time() - t_pblock))
        blocks = get_blocks(adMtrx, allents, args)
        del adMtrx
        print("Finish blocking: {:.4f} s".format(time.time() - t_pblock))
        sim_block = sim_block.to_dense()  # .cpu().detach().numpy()

        all1s = []
        for block in blocks:
            if len(block) > 1:
                predicted_result = block_recip(block, sim_block, now2id, assoc, shift, test_source_matrixid, test_target_matrixid, linking_dic, predicted_result, redun, args.use_semi)
            else:
                all1s.append(list(block)[0])
        print("first round costs: {:.4f} s".format(time.time() - t_pblock))
        del blocks

        if args.approx2 == 1:
            predicted_result = block_recip(all1s, sim_block, now2id, assoc, shift, test_source_matrixid, test_target_matrixid, linking_dic, predicted_result, redun, args.use_semi)
            print("final round costs: {:.4f} s".format(time.time() - t_pblock))

        else:
            for i in range(1, args.approx2):
                print("$$$$$$$$$$$$$$$$$$$ Round " + str(i + 1) + " $$$$$$$$$$$$$$$$$$$$$")
                adMtrx = prepare_followingR(all1s, sim_block, now2id, THETA[i], assoc, shift, test_source_matrixid, test_target_matrixid, shift_value)
                blocks = get_blocks(adMtrx, set(all1s), args)
                del adMtrx
                all1s = []
                for block in blocks:
                    if len(block) > 1:
                        predicted_result = block_recip(block, sim_block, now2id, assoc, shift, test_source_matrixid, test_target_matrixid, linking_dic, predicted_result, redun, args.use_semi)
                    else:
                        all1s.append(list(block)[0])
                print("new round costs: {:.4f} s".format(time.time() - t_pblock))

            predicted_result = block_recip(all1s, sim_block, now2id, assoc, shift, test_source_matrixid, test_target_matrixid, linking_dic, predicted_result, redun, args.use_semi)
            print("final round costs: {:.4f} s".format(time.time() - t_pblock))

    return curr_sim, predicted_result

def overall_eval(args, curr_sim, predicted_result, linking_dic, redun=False):
    if args.approx2 == 0:
        return curr_sim
    else:
        total_correct_cnt = 0

        if args.use_semi is False:
            if redun is True:
                for item in predicted_result:
                    predicted = predicted_result[item]
                    if len(predicted) > 1:
                        predicted = min(predicted, key=predicted.get)
                    else:
                        predicted = list(predicted.keys())[0]
                    if linking_dic[item] == predicted:
                        total_correct_cnt += 1
                print("Accuracy: " + str(total_correct_cnt) + ' / ' + str(len(predicted_result)) + ' = ' + str(total_correct_cnt * 1.0 / len(predicted_result)))
            else:
                for item in predicted_result:
                    predicted = predicted_result[item]
                    predicted = predicted[0]
                    if linking_dic[item] == predicted:
                        total_correct_cnt += 1
                print("Accuracy: " + str(total_correct_cnt) + ' / ' + str(len(predicted_result)) + ' = ' + str(total_correct_cnt * 1.0 / len(predicted_result)))
            return None
        else:
            semi = []
            predicted_result_ = predicted_result[1]
            predicted_result = predicted_result[0]

            if redun is True:
                for item in predicted_result:
                    predicted = predicted_result[item]
                    if len(predicted) > 1:
                        predicted = min(predicted, key=predicted.get)
                    else:
                        predicted = list(predicted.keys())[0]

                    # reverse
                    predicted_ = predicted_result_[predicted]
                    if len(predicted_) > 1:
                        predicted_ = min(predicted_, key=predicted_.get)
                    else:
                        predicted_ = list(predicted_.keys())[0]

                    if predicted_ == item:
                        semi.append([item, predicted])

                    if linking_dic[item] == predicted:
                        total_correct_cnt += 1
                print("Accuracy: " + str(total_correct_cnt) + ' / ' + str(len(predicted_result)) + ' = ' + str(total_correct_cnt * 1.0 / len(predicted_result)))
            else:
                for item in predicted_result:
                    predicted = predicted_result[item]
                    predicted = predicted[0]
                    # reverse
                    predicted_ = predicted_result_[predicted][0]
                    if predicted_ == item:
                        semi.append([item, predicted])

                    if linking_dic[item] == predicted:
                        total_correct_cnt += 1
                print("\nAccuracy: " + str(total_correct_cnt) + ' / ' + str(len(predicted_result)) + ' = ' + str(total_correct_cnt * 1.0 / len(predicted_result)))

            semi_correct_cnt = 0
            for item in semi:
                if linking_dic[item[0]] == item[1]:
                    semi_correct_cnt += 1
            print("Semi confidence: " + str(semi_correct_cnt) + ' / ' + str(len(semi)) + ' = ' + str(semi_correct_cnt * 1.0 / len(semi)))
            return np.array(semi)


def run_batched_ea(data: EAData, src_split, trg_split, semi_round, topk, args):
    # data = LargeScaleEAData('raw_data', 'de')
    # data.save('dedata')
    # dataset = LargeScaleEAData.load('dedata')
    print('read data complete')
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    curr_sim = None

    # batch_sampler_consistency(data, src_split, trg_split, topk, random=args.random_split, backbone=args.model, args=copy.deepcopy(args))
    # batch_sampler

    links = d.test
    linking_dic = dict()
    for item in links:
        linking_dic[item[0]] = item[1]

    if args.with_redundancy == "Redundancy":
        if args.use_semi is True:
            predicted_result = [defaultdict(dict), defaultdict(dict)]
        else:
            predicted_result = defaultdict(dict)

        if args.use_semi is True and semi_round > 0:
            cnt = 0
            for item in tqdm(batch_sampler_consistency(data, copy.deepcopy(args), semi_round, src_split, trg_split, topk,
                                                       random=args.random_split, backbone=args.model)):
                np.save(args.path + args.semi_store + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '_round_' + str(semi_round),
                        np.array(item[0]))
                np.save(args.path + args.semi_store + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split) + '_round_' + str(semi_round),
                        np.array(item[1]))
                np.save(args.path + args.semi_store + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split) + '_round_' + str(semi_round),
                        np.array(item[2]))
                np.save(args.path + args.semi_store + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split) + '_round_' + str(semi_round),
                        np.array(item[3]))
                np.save(
                    args.path + args.semi_store + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round),
                    np.array(item[4]))
                np.save(args.path + args.semi_store + str(cnt) + '_test_pairs_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round),
                        np.array(item[5]))
                np.save(args.path + args.semi_store + str(cnt) + '_real_test_pairs_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round), np.array(item[6]))
                cnt += 1

            for cnt in range(src_split):
                # print(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy')
                triple1 =  np.load(args.path + args.semi_store + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round) + '.npy', allow_pickle=True).tolist()
                triple2 =  np.load(args.path + args.semi_store + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round) + '.npy', allow_pickle=True).tolist()
                ids1 =  set(np.load(args.path + args.semi_store + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round) + '.npy', allow_pickle=True).tolist())
                ids2 =  set(np.load(args.path + args.semi_store + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round) + '.npy', allow_pickle=True).tolist())
                train_pairs =  np.load(args.path + args.semi_store + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round) + '.npy', allow_pickle=True).tolist()
                test_pairs =  np.load(args.path + args.semi_store + str(cnt) + '_test_pairs_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round) + '.npy', allow_pickle=True).tolist()
                real_test_pairs =  np.load(args.path + args.semi_store + str(cnt) + '_real_test_pairs_' + args.with_redundancy + '_' + str(args.src_split)+ '_round_' + str(semi_round) + '.npy', allow_pickle=True).tolist()
                batch = AlignmentBatch(triple1, triple2, ids1, ids2, train_pairs,test_pairs, real_test_pairs, backbone=args.model)
                curr_sim, predicted_result = run_each_batch(data.size(), batch, device, copy.deepcopy(args), curr_sim, linking_dic, predicted_result, redun=True)
            curr_sim = overall_eval(args, curr_sim, predicted_result, linking_dic, redun=True)
        else:
            if args.offline is True and (args.use_semi is False or (args.use_semi is True and semi_round ==0)):
                try:
                    print("OFFLINE DIRECT LOADING......")
                    for cnt in range(src_split):
                        # print(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy')
                        triple1 =  np.load(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        triple2 =  np.load(args.path + 'store/' + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        ids1 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                        ids2 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                        train_pairs =  np.load(args.path + 'store/' + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        real_test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_real_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        batch = AlignmentBatch(triple1, triple2, ids1, ids2, train_pairs,test_pairs, real_test_pairs, backbone=args.model)
                        curr_sim, predicted_result = run_each_batch(data.size(), batch, device, copy.deepcopy(args), curr_sim, linking_dic, predicted_result, redun=True)
                    # only for the blocking process to produce the results
                    # for this, curr_sim is actually semi pairs
                    curr_sim = overall_eval(args, curr_sim, predicted_result, linking_dic, redun=True)
                except:
                    print("OFFLINE FIRST TIME......")
                    cnt = 0
                    for item in tqdm(batch_sampler_consistency(data, copy.deepcopy(args),semi_round, src_split, trg_split, topk, random=args.random_split, backbone=args.model)):
                        np.save(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[0]))
                        np.save(args.path + 'store/' + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[1]))
                        np.save(args.path + 'store/' + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[2]))
                        np.save(args.path + 'store/' + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[3]))
                        np.save(args.path + 'store/' + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[4]))
                        np.save(args.path + 'store/' + str(cnt) + '_test_pairs_'  + args.with_redundancy + '_' + str(args.src_split), np.array(item[5]))
                        np.save(args.path + 'store/' + str(cnt) + '_real_test_pairs_'  + args.with_redundancy + '_' + str(args.src_split), np.array(item[6]))
                        cnt+=1
                    for cnt in range(src_split):
                        # print(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy')
                        triple1 =  np.load(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        triple2 =  np.load(args.path + 'store/' + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        ids1 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                        ids2 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                        train_pairs =  np.load(args.path + 'store/' + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        real_test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_real_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                        batch = AlignmentBatch(triple1, triple2, ids1, ids2, train_pairs,test_pairs, real_test_pairs, backbone=args.model)
                        curr_sim, predicted_result = run_each_batch(data.size(), batch, device, copy.deepcopy(args), curr_sim, linking_dic, predicted_result, redun=True)
                    curr_sim = overall_eval(args, curr_sim, predicted_result, linking_dic, redun=True)
            else:
                print("ONLINE MODE......")
                for batch in tqdm(batch_sampler_consistency(data, copy.deepcopy(args),semi_round, src_split, trg_split, topk, random=args.random_split, backbone=args.model)):
                    curr_sim, predicted_result = run_each_batch(data.size(), batch, device, copy.deepcopy(args), curr_sim, linking_dic, predicted_result, redun=True)
                curr_sim = overall_eval(args, curr_sim, predicted_result, linking_dic, redun=True)
    else:
        if args.use_semi is True:
            predicted_result = [defaultdict(list), defaultdict(list)]
        else:
            predicted_result = defaultdict(list)

        if args.offline is True and (args.use_semi is False or (args.use_semi is True and semi_round ==0)):
            try:
                print("OFFLINE DIRECT LOADING......")
                for cnt in range(src_split):
                    triple1 =  np.load(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    triple2 =  np.load(args.path + 'store/' + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    ids1 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                    ids2 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                    train_pairs =  np.load(args.path + 'store/' + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    real_test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_real_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    batch = AlignmentBatch(triple1, triple2, ids1, ids2, train_pairs,test_pairs, real_test_pairs, backbone=args.model)
                    curr_sim, predicted_result = run_each_batch(data.size(), batch, device, args, curr_sim, linking_dic, predicted_result)
                curr_sim = overall_eval(args, curr_sim, predicted_result, linking_dic)
            except:
                print("OFFLINE FIRST TIME......")
                cnt = 0
                for item in tqdm(batch_sampler(data, copy.deepcopy(args), semi_round, src_split, trg_split, topk, random=args.random_split, backbone=args.model)):
                    np.save(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[0]))
                    np.save(args.path + 'store/' + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[1]))
                    np.save(args.path + 'store/' + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[2]))
                    np.save(args.path + 'store/' + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[3]))
                    np.save(args.path + 'store/' + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[4]))
                    np.save(args.path + 'store/' + str(cnt) + '_test_pairs_'  + args.with_redundancy + '_' + str(args.src_split), np.array(item[5]))
                    np.save(args.path + 'store/' + str(cnt) + '_real_test_pairs_' + args.with_redundancy + '_' + str(args.src_split), np.array(item[6]))
                    cnt+=1
                for cnt in range(src_split):
                    triple1 =  np.load(args.path + 'store/' + str(cnt) + '_triple1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    triple2 =  np.load(args.path + 'store/' + str(cnt) + '_triple2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    ids1 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids1_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                    ids2 =  set(np.load(args.path + 'store/' + str(cnt) + '_ids2_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist())
                    train_pairs =  np.load(args.path + 'store/' + str(cnt) + '_train_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    real_test_pairs =  np.load(args.path + 'store/' + str(cnt) + '_real_test_pairs_' + args.with_redundancy + '_' + str(args.src_split) + '.npy', allow_pickle=True).tolist()
                    batch = AlignmentBatch(triple1, triple2, ids1, ids2, train_pairs,test_pairs, real_test_pairs, backbone=args.model)
                    curr_sim, predicted_result = run_each_batch(data.size(), batch, device, args, curr_sim, linking_dic, predicted_result)
                curr_sim = overall_eval(args, curr_sim, predicted_result, linking_dic)
        else:
            print("ONLINE MODE......")
            for batch in tqdm(batch_sampler(data, copy.deepcopy(args), semi_round, src_split, trg_split, topk, random=args.random_split, backbone=args.model)):
                curr_sim, predicted_result = run_each_batch(data.size(), batch, device, args, curr_sim, linking_dic, predicted_result)
            curr_sim = overall_eval(args, curr_sim, predicted_result, linking_dic)

    return curr_sim

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rate", type=float, default=0.3, help="training set rate")
    parser.add_argument("--val", type=float, default=0.0, help="valid set rate")
    parser.add_argument("--save", default="", help="the output dictionary of the model and embedding")
    parser.add_argument("--pre", default="", help="pre-train embedding dir (only use in transr)")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--log", type=str, default="tensorboard_log", nargs="?", help="where to save the log")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--check", type=int, default=5, help="check point")
    parser.add_argument("--update", type=int, default=5, help="number of epoch for updating negtive samples")
    parser.add_argument("--train_batch_size", type=int, default=-1, help="train batch_size (-1 means all in)")
    parser.add_argument("--early", action="store_true", default=False,
                        help="whether to use early stop")  # Early stop when the Hits@1 score begins to drop on the validation sets, checked every 10 epochs.
    parser.add_argument("--share", action="store_true", default=False, help="whether to share ill emb")
    parser.add_argument("--swap", action="store_true", default=False, help="whether to swap ill in triple")

    parser.add_argument("--bootstrap", action="store_true", default=False, help="whether to use bootstrap")
    parser.add_argument("--start_bp", type=int, default=9, help="epoch of starting bootstrapping")
    parser.add_argument("--threshold", type=float, default=0.75, help="threshold of bootstrap alignment")

    parser.add_argument("--encoder", type=str, default="GCN-Align", nargs="?", help="which encoder to use: . max = 1")
    parser.add_argument("--hiddens", type=str, default="100,100,100",
                        help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
    parser.add_argument("--heads", type=str, default="1,1", help="heads in each gat layer, splitted with comma")
    parser.add_argument("--attn_drop", type=float, default=0, help="dropout rate for gat layers")

    parser.add_argument("--decoder", type=str, default="Align", nargs="?", help="which decoder to use: . min = 1")
    parser.add_argument("--sampling", type=str, default="N", help="negtive sampling method for each decoder")
    parser.add_argument("--k", type=str, default="25", help="negtive sampling number for each decoder")
    parser.add_argument("--margin", type=str, default="1",
                        help="margin for each margin based ranking loss (or params for other loss function)")
    parser.add_argument("--alpha", type=str, default="1", help="weight for each margin based ranking loss")
    parser.add_argument("--feat_drop", type=float, default=0, help="dropout rate for layers")

    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--dr", type=float, default=0, help="decay rate of lr")

    parser.add_argument("--train_dist", type=str, default="euclidean",
                        help="distance function used in train (inner, cosine, euclidean, manhattan)")
    parser.add_argument("--test_dist", type=str, default="euclidean",
                        help="distance function used in test (inner, cosine, euclidean, manhattan)")

    parser.add_argument("--csls", type=int, default=0, help="whether to use csls in test (0 means not using)")
    parser.add_argument("--rerank", action="store_true", default=False, help="whether to use rerank in test")

    # My arguments
    parser.add_argument('--dataset', type=str, default='dbp_wd')  # en_fr fb_dbp_2m dbp_wd zh_en
    parser.add_argument('--path', type=str, default='DWY100K/dbp_wd/')  # DWY100K/dbp_wd/ DBP15K/zh_en/
    parser.add_argument('--src_split', type=int, default=5)
    parser.add_argument('--trg_split', type=int, default=5)
    parser.add_argument('--topk_corr', type=int, default=1)
    parser.add_argument('--it_round', type=int, default=1)
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs to train")
    parser.add_argument('--model', type=str, default='gcn-align')  # gcn-align rrea
    parser.add_argument('--openea', action='store_true', default=False)
    parser.add_argument('--eval_which', type=str, default='sgtnf')
    parser.add_argument("--random_split", action="store_true", default=False, help="whether to use random split")
    parser.add_argument("--save_folder", type=str, default='tmp4')

    parser.add_argument('--use_semi', action='store_true', default=True)
    parser.add_argument("--semi_store", type=str, default='approx2/')  # Redundancy NoRedundancy
    parser.add_argument("--with_redundancy", type=str, default='Redundancy')  # Redundancy NoRedundancy
    parser.add_argument("--approx2", type=int, default=0)  # 0 represents not using 1 represents perform for one round
    parser.add_argument("--infer", type=str, default="recip_approx1")  # direct recip recip_approx1
    parser.add_argument('--offline', action='store_true', default=False) # only for the large dataset! change to True
    parser.add_argument('--skip_training', action='store_true', default=False)
    return parser.parse_args()

default_folder = 'tmp4/'


def save_sim(sim, args):
    path = args.path + args.model + '_sim_' + args.with_redundancy + '_' + str(args.src_split)
    torch.save(sim, path)


if __name__ == '__main__':
    begin = time.time()
    args = get_args()

    if args.epoch < 0:
        args.epoch = {'rrea': 100, 'gcn-align': 300}.get(args.model, 100)

    # load the data, pre-stored the large scale ones
    try:
        d = EAData.load(default_folder + 'dataset_{0}_{1}'.format(args.dataset, str(args.rate)))
    except:
        if args.dataset in ['zh_en', 'ja_en', 'fr_en']:
            d = EAData(args.path + 'rel_triples_1', args.path + 'rel_triples_2', args.path + 'ent_links', train_ratio = args.rate)
        elif args.dataset == 'en_fr':
            path = 'SRPRS/en_fr/'
            d = EAData(path + 'rel_triples_1', path + 'rel_triples_2', path + 'ent_links', train_ratio = args.rate)
        elif args.dataset in ['dbp_wd', 'dbp_yg']:
            path = 'DWY100K/' + args.dataset + '/'
            d = EAData(path + 'rel_triples_1', path + 'rel_triples_2', path + 'ent_links', train_ratio = args.rate)
        elif args.dataset == 'fb_dbp_2m':
            path = 'fb_dbp_2m/'
            d = EAData(path + 'rel_triples_1', path + 'rel_triples_2', path + 'ent_links', train_ratio=args.rate)
        d.save(default_folder + 'dataset_{0}_{1}'.format(args.dataset, str(args.rate)))

    # print(d)
    data = args.dataset
    # print(argprint(data=data, rdm=args.random_split))

    t_total = time.time()
    print('total semi pairs', str(len(d.train)))

    if args.approx2 == 0:
        if args.skip_training is True:
            try:
                path = args.path + args.model + '_sim_' + args.with_redundancy + '_' + str(args.src_split)
                stru_sim = torch.load(path)
            except:
                stru_sim = run_batched_ea(d, args.src_split, args.trg_split, 0, args.topk_corr, args)
                save_sim(stru_sim, args)
        else:
            stru_sim = run_batched_ea(d, args.src_split, args.trg_split, 0, args.topk_corr, args)
            save_sim(stru_sim, args)

        print("total time elapsed: {:.4f} s".format(time.time() - t_total))

        if args.infer == "direct":
            new_semi_pairs = sparse_acc_direct(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
        elif args.infer == "recip":
            new_semi_pairs = sparse_acc_recip(stru_sim, d.ill(d.test, 'cpu'), args.use_semi)
        elif args.infer == "recip_approx1":
            new_semi_pairs = sparse_acc_recip_approx1(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
    else:
        new_semi_pairs = run_batched_ea(d, args.src_split, args.trg_split, 0, args.topk_corr, args)
    print("total time elapsed: {:.4f} s".format(time.time() - t_total))

    if args.use_semi:
        print("\n#####################################1#####################################\n")
        d.train = new_semi_pairs
        print('total semi pairs', str(len(d.train)))

        if args.approx2 == 0:
            stru_sim = run_batched_ea(d, args.src_split, args.trg_split, 1, args.topk_corr, args)
            if args.infer == "direct":
                new_semi_pairs = sparse_acc_direct(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
            elif args.infer == "recip":
                new_semi_pairs = sparse_acc_recip(stru_sim, d.ill(d.test, 'cpu'), args.use_semi)
            elif args.infer == "recip_approx1":
                new_semi_pairs = sparse_acc_recip_approx1(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
        else:
            new_semi_pairs = run_batched_ea(d, args.src_split, args.trg_split, 1, args.topk_corr, args)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))
        print("\n#####################################2#####################################\n")
        d.train = new_semi_pairs
        print('total semi pairs', str(len(d.train)))

        if args.approx2 == 0:
            stru_sim = run_batched_ea(d, args.src_split, args.trg_split, 2, args.topk_corr, args)
            if args.infer == "direct":
                new_semi_pairs = sparse_acc_direct(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
            elif args.infer == "recip":
                new_semi_pairs = sparse_acc_recip(stru_sim, d.ill(d.test, 'cpu'), args.use_semi)
            elif args.infer == "recip_approx1":
                new_semi_pairs = sparse_acc_recip_approx1(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
        else:
            new_semi_pairs = run_batched_ea(d, args.src_split, args.trg_split, 2, args.topk_corr, args)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))
        print("\n#####################################3#####################################\n")
        d.train = new_semi_pairs
        print('total semi pairs', str(len(d.train)))

        if args.approx2 == 0:
            stru_sim = run_batched_ea(d, args.src_split, args.trg_split, 3, args.topk_corr, args)
            if args.infer == "direct":
                new_semi_pairs = sparse_acc_direct(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
            elif args.infer == "recip":
                new_semi_pairs = sparse_acc_recip(stru_sim, d.ill(d.test, 'cpu'), args.use_semi)
            elif args.infer == "recip_approx1":
                new_semi_pairs = sparse_acc_recip_approx1(stru_sim, d.ill(d.test, 'cpu'), t_total, args.use_semi)
        else:
            new_semi_pairs = run_batched_ea(d, args.src_split, args.trg_split, 3, args.topk_corr, args)
        print("total time elapsed: {:.4f} s".format(time.time() - t_total))