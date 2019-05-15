"""
Apriori算法
"""

import itertools


def create_C_1(dataset):
    """
    生成候选1项集

    :param dataset: 数据集：列表，每一行为一个列表
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
    :return: 满足最小支持度的频繁k项集：字典，键为项集，值为支持度
    """
    support = {}
    for i in dataset:
        for j in C_k:
            if j.issubset(i):
                support[j] = support.get(j, 0) + 1
    n = len(dataset)
    return {k: v / n for k, v in support.items() if v / n >= min_support}


def apriori_gen(L_k):
    """
    使用频繁k项集生成候选k+1项集

    :param L_k: 频繁k项集：字典，键为项集，值为支持度
    :return: 候选k+1项集：列表，每一项为一个frozenset
    """
    len_L_k = len(L_k)
    k = len(L_k[0])
    if len_L_k > 1 and k > 0:
        return set([
            L_k[i].union(L_k[j]) for i in range(len_L_k - 1)
            for j in range(i + 1, len_L_k) if len(L_k[i] | L_k[j]) == k + 1
        ])


def apriori(dataset, min_support=0.5):
    """
    Apriori算法，计算频繁项集

    :param dataset:     数据集：列表，每一行为一个列表
    :param min_support: 最小支持度：浮点数，0~1之间，默认为0.5
    :return: 所有频繁项集：字典，键为项集，值为支持度
    """
    C_1 = create_C_1(dataset)
    L_1 = scan_D(dataset, C_1, min_support)
    L = [L_1]
    k = 2
    while len(L[k - 2]) > 1:
        C_k = apriori_gen(list(L[k - 2].keys()))
        L_k = scan_D(dataset, C_k, min_support)
        if len(L_k) > 0:
            L.append(L_k)
            k += 1
        else:
            break
    d = {}
    for L_k in L:
        d.update(L_k)
    return d


def rules_gen(iterable):
    """
    分拆频繁项集，生成左手、右手规则

    :param iterable: 可迭代对象（如列表）
    :return: 所有可能出现的规则：
               列表，由元组构成，每个元组由两个frozenset组成，分别代表条件和结果
    """
    subset = []
    for i in range(1, len(iterable)):
        subset.extend(itertools.combinations(iterable, i))
    return [(frozenset(lhs), frozenset(iterable.difference(lhs)))
            for lhs in subset]


def arules(dataset, min_support=0.5):
    """
    对数据集使用Apriori算法计算频繁项集和候选规则，
    并计算规则的支持度、置信度、提升度

    :param dataset:     数据集：列表，每一行为一个列表
    :param min_support: 最小支持度：浮点数，0~1之间，默认为0.5
    :return: 频繁项集的候选规则，及其支持度、置信度和提升度：
                列表，每一项为字典，有以下键值：
                    'LHS'：左手规则；
                    'RHS'：右手规则；
                    'support'：支持度；
                    'confidence'：置信度；
                    'lift'：提升度
    """
    L = apriori(dataset, min_support)
    rules = []
    for Lk in L.keys():
        if len(Lk) > 1:
            rules.extend(rules_gen(Lk))
    scl = []
    for rule in rules:
        lhs = rule[0]
        rhs = rule[1]
        support = L[lhs | rhs]
        confidence = support / L[lhs]
        lift = confidence / L[rhs]
        scl.append({
            'LHS': lhs,
            'RHS': rhs,
            'support': support,
            'confidence': confidence,
            'lift': lift
        })
    return scl


if __name__ == '__main__':
    dataset = [['A', 'C', 'D'], ['B', 'C', 'E'], ['A', 'B', 'C', 'E'],
               ['B', 'E']]
    min_support = 0.4
    res = arules(dataset, min_support)
    import pandas as pd
    print(pd.DataFrame(res))
