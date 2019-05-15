#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-14 14:12
"""

import itertools
import numpy as np


class tree_node:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.next_same = None


def create_C_1(dataset):
    """
    生成候选1项集

    :param dataset: 数据集：列表，每一项为一个列表
    :return: 候选1项集：列表，每一项为一个frozenset
    """
    C_1 = set(itertools.chain(*dataset))
    return [frozenset([i]) for i in C_1]


def scan_D(dataset, C_k, min_support):
    """
    计算项集支持度

    :param dataset:     数据集： 列表，每一行为一个列表
    :param C_k:         候选k项集：列表，每一项为一个frozenset
    :param min_support: 最小支持度：浮点数，0~1之间
    :return: 满足最小支持度的频繁k项集：列表，元素为元组，包含项集与支持度
    """
    support = {}
    for i in dataset:
        for j in C_k:
            if j.issubset(i):
                support[j] = support.get(j, 0) + 1
    n = len(dataset)
    return [(k, v / n) for k, v in support.items() if v / n >= min_support]


def insert_tree_1(pP, T):
    p = pP[0]
    P = pP[1:]
    if len(T.children_node) > 0 and p.name in T.children_node
        T.children_node[p.name].inc(1)
    else:
        T.children_node[p.name] = tree_node(p.name, 1, T)


class header_table_items:
    def __init__(self, id, sup_count, next_same_node):
        self.id = id
        self.sup_count = sup_count
        self.next_same_node = next_same_node

header_table = {}

def fp_growth(dataset, min_support):
    C_1 = create_C_1(dataset)
    L_1 = scan_D(dataset, C_1, min_support)
    L_1.sort(key=lambda x: (x[1]), reverse = True)
    root_tree = tree_node('null', 0, None)
    for trans in dataset:
        L_1.sort(key=lambda x: (x[1]), reverse=True)


