import networkx as nx
import nxmetis

from utils import *
from dataset import EAData, LargeScaleEAData
from random import sample
import numpy as np
from dto import *
from collections import defaultdict
from tqdm import tqdm, trange
from typing import *
import argparse, logging, random, time


def stat(array, name, print_all=False):
    args = dict(name=name,
                min=np.min(array),
                max=np.max(array),
                mean=np.mean(array),
                std=np.std(array))
    if print_all:
        args['data'] = array
    print(argprint(**args))


def overlaps(src: List[set], trg: List[set]):
    return np.array([[float(len(s.intersection(t))) / (float(len(s)) + 0.01) for t in trg] for s in src])


class Partition:
    def __init__(self, data: EAData, seed_rate: float):
        self.data = data
        self.train_set, self.train_map, self.link_map, self.link_set, self.test_set = self.get_train_node_sets(data, seed_rate=seed_rate)

    # @staticmethod
    # def get_train_node_sets(data, seed_rate=0.0):
    #     train_pairs = data.train
    #     all_pairs = data.link
    #     test_pairs = data.test
    #     node_sets = [set(tp[0] for tp in train_pairs),
    #                  set(tp[1] for tp in train_pairs)]
        
    #     train_map = [{tp[0]: tp[1] for tp in all_pairs},
    #                  {tp[1]: tp[0] for tp in all_pairs}]
    #     link_sets = [set(tp[0] for tp in all_pairs),
    #                  set(tp[1] for tp in all_pairs)]

    #     test_sets = [set(tp[0] for tp in all_pairs),
    #                  set(tp[1] for tp in all_pairs)]
    #     return node_sets, train_map, link_sets, test_sets
    
    @staticmethod
    def get_train_node_sets(data, seed_rate=0.0):
        all_pairs = data.link
        train_cnt = int(len(all_pairs) * seed_rate)
        train_pairs = all_pairs[:train_cnt]
        test_pairs = all_pairs[train_cnt:]

        train_sets = [set(tp[0] for tp in train_pairs),
                     set(tp[1] for tp in train_pairs)]
        train_map = [{tp[0]: tp[1] for tp in train_pairs},
                     {tp[1]: tp[0] for tp in train_pairs}]
        
        link_map = [{tp[0]: tp[1] for tp in all_pairs},
                     {tp[1]: tp[0] for tp in all_pairs}]
        link_sets = [set(tp[0] for tp in all_pairs),
                     set(tp[1] for tp in all_pairs)]

        test_sets = [set(tp[0] for tp in all_pairs),
                     set(tp[1] for tp in all_pairs)]
        return train_sets, train_map, link_map, link_sets, test_sets

    @staticmethod
    def construct_edge_graph(triples, important_nodes=None, weight=1000):
        edges_map = defaultdict(list)
        for n, tr in enumerate(triples):
            h, _, t = tr
            edges_map[h].append(n)
            edges_map[t].append(n)

        print("total nodes with triple:", len(edges_map))

        g = nx.Graph()
        now, total = 0, len(edges_map)
        merged_important_nodes = set()
        if important_nodes is not None:
            for nodes in important_nodes:
                merged_important_nodes.update(nodes)

        for node, edges in edges_map.items():

            if important_nodes is None:
                node_weight = 1
            else:
                node_weight = weight if node in merged_important_nodes else 1
            curr_node_graph: nx.DiGraph = nx.complete_graph(edges)
            for u, v in curr_node_graph.edges:
                g.add_edge(u, v, weight=g.get_edge_data(u, v, {}).get('weight', 0) + node_weight)

            now += 1
            if now % 50000 == 0:
                print('create graph', now, 'complete')

        return g

    def partition_by_edge(self, src=0, k=30):
        g0 = self.construct_edge_graph(self.data.triples[src])
        trg = 1 - src
        print('construct src graph complete, total nodes={0}, total edges={1}'
              .format(len(g0.nodes), len(g0.edges)))
        mincut, src_edges = nxmetis.partition(g0, k)
        print('src graph partition complete, mincut=', mincut)

    @staticmethod
    def make_cnt_edges(lst):
        mp = defaultdict(int)
        for item in lst:
            # mp[item] += 1
            mp[tuple(item)] += 1

        return [(k[0], k[1], v) for k, v in mp.items()]

    @staticmethod
    def construct_graph(triples, important_nodes=None, known_weight=1000, known_size=None,
                        cnt_as_weight=False, keep_inter_edges=True):
        g = nx.Graph()
        # edges = [(t[0], t[2]) for t in triples]
        edges = triples
        ## 两个if 感觉没有区别？
        if cnt_as_weight:
            edges = Partition.make_cnt_edges(Partition.make_cnt_edges(edges))
            g.add_weighted_edges_from(edges)
        else:
            g.add_edges_from(edges)
            nx.set_edge_attributes(g, 1, 'weight')
        # g.edges.data('weight', default=1)
        if important_nodes:
            subgraphs = []
            print('set important node weights:')
            for nodes in tqdm(important_nodes):
                sn = nodes[0]
                g.add_edges_from([(sn, n) for n in nodes])
                subgraph = g.subgraph(nodes)
                if known_size:
                    nx.set_node_attributes(subgraph, known_size, 'size')
                nx.set_edge_attributes(subgraph, known_weight, 'weight')
                subgraphs.append(subgraph)
                # subgraphs.append(g.subgraph(nodes))
                # subgraphs[-1].edges.data('weight', default=weight)
            print('compose subgraphs')
            for sg in tqdm(subgraphs):
                g = nx.compose(g, sg)

            if keep_inter_edges:
                return g

            merged_important_nodes = set()
            for nodes in important_nodes:
                merged_important_nodes.update(nodes)
            # print('all important nodes merged')

            print('del inter edges:')
            for nodes in tqdm(important_nodes):
                all_neighbors = [g.neighbors(n) for n in nodes]
                # neighbors = set()
                choices = []
                for n in all_neighbors:
                    choices.append(merged_important_nodes.intersection(n) - set(nodes))
                edges = [(e1, e2) for idx, e1 in enumerate(nodes) for e2 in choices[idx]]
                g.remove_edges_from(edges)
        return g

    def subgraph_trainset(self, node_lists, src=0, no_trg=False):
        src_train = []
        train = self.train_set[src]
        mp = self.train_map[src]
        for i, nodes in enumerate(node_lists):
            curr = []
            for n in nodes:
                if n in train:
                    curr.append(n)
            src_train.append(curr)
        if no_trg:
            return src_train
        trg_train = [[mp[e] for e in curr] for curr in src_train]
        return src_train, trg_train

    @staticmethod
    def share_triplets(src_triplet, trg_triplet, train_set, node_mapping, rel_mapping=None):
        if rel_mapping is None:
            rel_mapping = lambda x: x

        new_trg = []

        print('share triplet')
        for triplet in tqdm(src_triplet):
            h, r, t = triplet
            if h in train_set and t in train_set:
                new_trg.append([node_mapping[h], rel_mapping(r), node_mapping[t]])

        return trg_triplet + new_trg
    
    @staticmethod
    def cut_triplets(src_triplet, trg_triplet, link):
        """ src_triplet: [[0,0,1], [0,0,2],...]
            link:[(,), (,)]
            去掉outliers, 并重新编码
        """
        src_idx = [tup[0] for tup in link]
        tgt_idx = [tup[1] for tup in link]
        
        src_assoc = {v: k for k, v in enumerate(src_idx)}
        tgt_assoc = {v: k for k, v in enumerate(tgt_idx)}
        new_src, new_trg = [], []

        for triplet in tqdm(src_triplet):
            h, r, t = triplet
            if h in src_idx and t in src_idx:
                # new_src.append(triplet)
                new_src.append((src_assoc[h], r, src_assoc[t]))
        for triplet in tqdm(trg_triplet):
            h, r, t = triplet
            if h in tgt_idx and t in tgt_idx:
                # new_trg.append(triplet)
                new_trg.append((tgt_assoc[h], r, tgt_assoc[t]))
        
        return new_src, new_trg

    def random_partition(self, src=0, src_k=20, trg_k=20, *args, **kwargs):
        trg = 1 - src
        assert trg_k == src_k
        src_node_len, trg_node_len = len(self.data.ents[src]), len(self.data.ents[trg])
        src_nodes = set(range(src_node_len))
        trg_nodes = set(range(trg_node_len))
        src_train, trg_train = self.train_set[src], self.train_set[trg]
        src_test, trg_test = map(lambda x, y: x - y, [src_nodes, trg_nodes], [src_train, trg_train])

        def split_k_parts(nodes: set, k) -> List[set]:
            nodes_list = list(nodes)
            random.shuffle(nodes_list)
            print('total {} of nodes to split'.format(len(nodes_list)))

            batch_size = int(len(nodes_list) / k)
            ret = []
            for i_batch in trange(0, len(nodes_list), batch_size):
                i_end = min(i_batch + batch_size, len(nodes_list))
                ret.append(set(nodes_list[i_batch:i_end]))
            return ret

        src_test, src_train, trg_test = map(split_k_parts, [src_test, src_train, trg_test], [src_k] * 3)
        src_nodes = [x for x in map(lambda x, y: x.union(y), src_test, src_train)]
        src_train, trg_train = self.subgraph_trainset(src_train, src)
        trg_nodes = [x for x in map(lambda x, y: x.union(y), trg_test, trg_train)]
        trg_train = self.subgraph_trainset(trg_train, trg, True)
        return src_nodes, trg_nodes, src_train, trg_train

    def partition(self, src=0, src_k=30, trg_k=125, share_triplets=True):
        trg = 1 - src
        if share_triplets:
            trg_triplets = self.share_triplets(self.data.triples[src], self.data.triples[trg],
                                               self.train_set[src], self.train_map[src])
            src_triplets = self.share_triplets(self.data.triples[trg], self.data.triples[src],
                                               self.train_set[trg], self.train_map[trg])
        else:
            src_triplets, trg_triplets = reversed(self.data.triples)
        g0 = self.construct_graph(src_triplets, cnt_as_weight=True)
        print('construct src graph complete, total nodes={0}, total edges={1}'
              .format(len(g0.nodes), len(g0.edges)))
        mincut, src_nodes = nxmetis.partition(g0, src_k)
        print('src graph partition complete, mincut=', mincut)
        src_train, trg_train = self.subgraph_trainset(src_nodes, src)
        print('filter trainset complete')
        # g1 = self.construct_graph(trg_triplets, None, keep_inter_edges=True)
        g1 = self.construct_graph(trg_triplets, trg_train, keep_inter_edges=False)
        print('construct trg graph complete')
        mincut, trg_nodes = nxmetis.partition(g1, trg_k)
        print('trg graph partition complete, mincut=', mincut)
        return src_nodes, trg_nodes, src_train, self.subgraph_trainset(trg_nodes, trg, True)

    def _apply_set_op(self, src_sets, trg_sets, which, opname, rhs):
        src_sets = [set(l) for l in src_sets]
        trg_sets = [set(l) for l in trg_sets]
        if opname == 'minus':
            op = lambda x, y: x - y
        elif opname == 'cross':
            op = lambda x, y: x.intersection(y)
        else:
            raise NotImplementedError
        if rhs:
            src_sets = [op(s, set(rhs[which])) for i, s in enumerate(src_sets)]
            trg_sets = [op(t, set(rhs[1 - which])) for i, t in enumerate(trg_sets)]

        return src_sets, trg_sets

    def eval_align(self, src_sets, trg_sets, which=0, opname='minus', rhs=None):

        def filter_has_corr(w, st):
            return set(filter(lambda x: x in self.train_map[w], st))

        src_sets = [set(filter_has_corr(which, l)) for l in src_sets]
        trg_sets = [set(filter_has_corr(1 - which, l)) for l in trg_sets]
        if opname == 'minus':
            op = lambda x, y: x - y
        elif opname == 'cross':
            op = lambda x, y: x.intersection(y)
        else:
            raise NotImplementedError
        if rhs:
            src_sets = [op(s, set(rhs[0])) for i, s in enumerate(src_sets)]
            trg_sets = [op(t, set(rhs[1])) for i, t in enumerate(trg_sets)]

        src_sets = [set(self.train_map[which][i] for i in s) for s in src_sets]

        s2t, t2s = overlaps(src_sets, trg_sets), overlaps(trg_sets, src_sets)
        s2t, t2s = torch.from_numpy(s2t), torch.from_numpy(t2s)
        s2t, s2t_ind = torch.topk(s2t, k=1)
        t2s, t2s_ind = torch.topk(t2s, k=1)

        print(s2t.flatten().tolist())
        print(t2s.flatten().tolist())
        print('s2t mean\n', s2t.mean().item())
        print('t2s mean\n', t2s.mean().item())
        return s2t.mean().item(), t2s.mean().item()

    def eval_partition(self, src_nodes, trg_nodes, src_train, trg_train, *args, **kwargs):
        print("--Nodes")
        a, b = self.eval_align(src_nodes, trg_nodes)

        print("--Train Nodes")
        c, d = self.eval_align(src_nodes, trg_nodes, opname='cross', rhs=self.train_set)

        print('--Eval Nodes')
        e, f = self.eval_align(src_nodes, trg_nodes, rhs=self.train_set)
        return a, b, c, d, e, f
