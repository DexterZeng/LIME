from utils import *
import os.path as osp

import codecs
from collections.abc import Iterable
import pickle
import json


def save_array(arr, path, print_len=False, sort_by=None, descending=False, sep=u'\t', encoding='utf-8'):
    if sort_by:
        arr = sorted(arr, key=lambda x: x[sort_by], reverse=descending)

    with codecs.open(path, 'w', encoding) as f:
        if print_len:
            f.write('{}\n'.format(len(arr)))
        for item in arr:
            if sep and isinstance(item, Iterable):
                f.write('{}\n'.format(sep.join([str(i) for i in item])))
            else:
                f.write('{}\n'.format(item))


def make_file(path):
    save_array([], path)


def save_map(mp, path, reverse_kv=False, sort_by_key=False, **kwargs):
    arr = [(v, k) if reverse_kv else (k, v) for k, v in mp.items()]
    if sort_by_key:
        kwargs['sort_by'] = 0
    save_array(arr, path, **kwargs)


def saveobj(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def readobj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def to_json(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=False, indent=4)


class EAData:
    def __init__(self, triple1_path, triple2_path, ent_links_path,
                 shuffle_pairs=False, train_ratio=0.2, unsup=False, **kwargs):

        rel1, ent1, triple1 = self.process_one_graph(triple1_path)
        rel2, ent2, triple2 = self.process_one_graph(triple2_path)
        self.unsup = unsup
        if self.unsup:
            print('use unsupervised mode')
        self.rel1, self.ent1, self.triple1 = rel1, ent1, triple1
        self.rel2, self.ent2, self.triple2 = rel2, ent2, triple2
        self.link = self.process_link(ent_links_path, ent1, ent2)
        self.rels = [rel1, rel2]
        self.ents = [ent1, ent2]
        self.triples = [triple1, triple2]
        self.train_cnt = 0 if unsup else int(train_ratio * len(self.link))
        # if shuffle_pairs:
        #     shuffle(self.link)

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def load(path):
        return readobj(path)

    def save(self, path):
        saveobj(self, path)

    def get_train(self):
        if self.unsup:
            if hasattr(self, 'semi_link'):
                return self.semi_link
            else:
                raise RuntimeError('No unsupervised pairs!!')
        now = np.array(self.link[:self.train_cnt])
        if hasattr(self, 'semi_link') and self.semi_link is not None:
            return np.concatenate([now, self.semi_link], axis=0)
        return now

    def set_train(self, link):
        self.semi_link = link

    train = property(get_train, set_train)

    @property
    def test(self):
        # return self.link[self.train_cnt + int(0.1 * len(self.link)): ]
        return self.link[self.train_cnt:]
    def save_eakit_format(self, path):
        lg1e = len(self.ent1)
        lg1r = len(self.rel1)

        new_g2e = {k: v + lg1e for k, v in self.ent2.items()}
        new_g2r = {k: v + lg1r for k, v in self.rel2.items()}
        new_g2t = [(h + lg1e, r + lg1r, t + lg1e) for h, r, t in self.triple1]
        new_pair = [(e1, e2 + lg1e) for e1, e2 in self.link]
        ents = [self.ent1, new_g2e]
        rels = [self.rel1, new_g2r]
        triples = [self.triple1, new_g2t]
        for i in range(1, 3):
            save_map(ents[i - 1], osp.join(path, 'ent_ids_{}'.format(i)),
                     reverse_kv=True, sort_by_key=True)
            save_map(rels[i - 1], osp.join(path, 'rel_ids_{}'.format(i)),
                     reverse_kv=True, sort_by_key=True)
            make_file(osp.join(path, 'training_attrs_{}'.format(i)))
            save_array(triples[i - 1], osp.join(path, 'triples_{}'.format(i)))
        # raise NotImplementedError
        save_array(new_pair, osp.join(path, 'ill_ent_ids'), sort_by=0)

    def save_openea_format(self, path):
        pass

    @staticmethod
    def process_one_graph(rel_pos: str):
        triples, rel_idx, ent_idx = [], {}, {}
        with codecs.open(rel_pos, "r", 'utf-8') as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                ent_idx, s = add_cnt_for(ent_idx, now[0])
                rel_idx, p = add_cnt_for(rel_idx, now[1])
                ent_idx, o = add_cnt_for(ent_idx, now[2])
                triples.append([s, p, o])
        return rel_idx, ent_idx, triples

    @staticmethod
    def process_link(links_pos, ent1, ent2):
        link = []
        with codecs.open(links_pos, "r", 'utf-8') as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                ent1, src = add_cnt_for(ent1, now[0])
                ent2, trg = add_cnt_for(ent2, now[1])
                link.append((src, trg))
        return link

    @staticmethod
    def ill(pairs, device='cuda'):
        return torch.tensor(pairs, dtype=torch.long, device=device).t()

    def get_pairs(self, device='cuda'):
        return self.ill(self.link, device)

    def size(self, which=None):
        if which is None:
            return [self.size(0), self.size(1)]

        return len(self.ents[which])

    def __repr__(self):
        return argprint(
            triple1=len(self.triple1),
            triple2=len(self.triple2),
            ent1=len(self.ent1),
            ent2=len(self.ent2),
            rel1=len(self.rel1),
            rel2=len(self.rel2),
            link=len(self.link)
        )