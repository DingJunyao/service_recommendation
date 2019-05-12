import itertools
import sys
import os
try:
    from seq_patterns.apriori import apriori
except ImportError:
    from apriori import apriori


def create_ls_1(dataset, min_support):
    '''
    找出序列模式中各序列的所有的频繁序列，
        组成序列模式的频繁1序列和支持度，并为它们映射一个值，便于后续计算

    输入：dataset     ：数据集：
                          列表，每行为一个列表，表示序列（对应客户的购买记录）；
                            序列由列表组成（单次购买的商品）；
          min_support ：最小支持度，0~1之间
    输出：频繁1序列的映射表；
            字典：键为frozen，表示频繁序列；值为映射的值
          频繁1序列的支持度
            字典：键为由映射值组成的元组；值为支持度
    '''
    n = len(dataset)
    flatten_set = list(itertools.chain(*dataset))
    flatten_n = len(flatten_set)
    min_support_new = min_support * n / flatten_n
    l_item_sets = apriori(flatten_set, min_support_new)
    mapping = {v: k for k, v in enumerate(l_item_sets)}
    support_ls_1 = {(mapping[k], ): v * flatten_n / n
                    for k, v in l_item_sets.items()}
    return mapping, support_ls_1


def seq_mapping(seq, mapping):
    '''
    对每个序列进行映射

    输入：seq     ：序列：
                      列表：数据集中的某个序列；
          mapping ：映射表：
                      字典：create_Ls_1()生成的映射表
    输出：映射后序列
    '''
    new_seq = []
    for i_set in seq:
        new_set = [v for k, v in mapping.items() if k <= set(i_set)]
        if new_set != []:
            new_seq.append(new_set)
    return new_seq


def transform(dataset, mapping):
    '''
    使用映射值转换数据集

    输入：dataset：数据集；
          mapping：映射表：
                      字典：create_Ls_1()生成的映射表
    输出：转换后的数据集
    '''
    transform_ds = []
    for seq in dataset:
        new_seq = seq_mapping(seq, mapping)
        if new_seq != []:
            transform_ds.append(new_seq)
    return transform_ds


def seq_gen(seq_a, seq_b):
    '''
    通过两个频繁k序列生成候选k+1序列

    输入：seq_a, seq_b：两个频繁k序列：列表
    输出：候选k+1序列：列表
    '''
    new_a, new_b = seq_a.copy(), seq_b.copy()
    if seq_a[:-1] == seq_b[:-1]:
        new_a.append(seq_b[-1])
        new_b.append(seq_a[-1])
        return [new_a, new_b]


def cs_gen(large_seq):
    '''
    根据频繁k-序列生成所有候选k+1序列

    输入：large_seq：频繁k序列：列表
    输出：候选k+1序列：列表
    '''
    cs = []
    for seq_a, seq_b in itertools.combinations(large_seq, 2):
        new_seqs = seq_gen(seq_a, seq_b)
        if new_seqs is not None:
            cs.extend(new_seqs)
    return [seq for seq in cs if seq[1:] in large_seq]


def is_subseq(seq, cus_seq):
    '''
    判断候选序列是否是某序列的子序列（转换阶段）

    输入：seq     ：要比较序列（候选序列）
                      列表
          cus_seq ：被比较序列（用户序列）
                      列表
    输出：布尔值，表示是否是子序列
    '''
    n_seq, n_cus_seq = len(seq), len(cus_seq)
    if n_seq > n_cus_seq:
        return False
    if n_seq == 1:
        return any([seq[0] in i for i in cus_seq])
    if n_seq > 1:
        head = [seq[0] in i for i in cus_seq]
        if any(head):
            split = head.index(True)
            return is_subseq(seq[1:], cus_seq[split + 1:])
        else:
            return False


def calc_support(transform_ds, cs, min_support):
    '''
    计算每个候选序列的支持度，根据最小支持度筛选，产生频繁k序列

    输入：transform_ds ：转换后数据集
          cs           ：候选序列
          min_support  ：最小支持度
    输出：频繁序列
          字典：键为频繁序列，值为支持度
    '''
    support_ls_k = {}
    n = len(transform_ds)
    if len(cs) >= 1:
        for seq in cs:
            support = sum(
                [is_subseq(seq, cus_seq) for cus_seq in transform_ds]) / n
            if support >= min_support:
                support_ls_k.update({tuple(seq): support})
    return [list(k) for k in support_ls_k], support_ls_k


def is_subseq2(seq, cus_seq):
    '''
    判断候选序列是否是某序列的子序列（最大化序列阶段）

    输入：seq     ：要比较序列（候选序列）
                      列表
          cus_seq ：被比较序列（用户序列）
                      列表
    输出：布尔值，表示是否是子序列
    '''
    n_seq, n_cus_seq = len(seq), len(cus_seq)
    if n_seq > n_cus_seq:
        return False
    if n_seq == 1:
        return any([seq[0].issubset(i) for i in cus_seq])
    if n_seq > 1:
        head = [seq[0].issubset(i) for i in cus_seq]
        if any(head):
            split = head.index(True)
            return is_subseq2(seq[1:], cus_seq[split:])
        else:
            return False


def not_proper_subseq(seq, cus_seq):
    '''
    若候选序列不是某个序列的非空真子序列，返回True

    输入：seq     ：要比较序列（候选序列）
                      列表
          cus_seq ：被比较序列（用户序列）
                      列表
    输出：布尔值，表示是否不是非空真子序列
    '''
    if seq == cus_seq:
        return True
    else:
        return not is_subseq2(seq, cus_seq)


def max_ls(ls, support_ls):
    '''
    将候选序列中最大化序列保留下来

    输入：ls         ：频繁序列
          support_ls ：频繁序列的支持度
    输出：ls, support_ls， 但是保留了最大化的序列
    '''
    ls_cp = ls.copy()
    len_l, len_c = len(ls), len(ls_cp)
    while len_c > 1 and len_l > 1:
        if ls_cp[len_c - 1] in ls:
            mask = [not_proper_subseq(seq, ls_cp[len_c - 1]) for seq in ls]
            ls = list(itertools.compress(ls, mask))
            len_l = len(ls)
        len_c -= 1
    support_ls = {tuple(seq): support_ls[tuple(seq)] for seq in ls}
    return ls, support_ls


def seq_patterns(dataset, min_support=0.4):
    '''
    使用Apriori算法寻找序列模式

    输入：dataset     ：数据集：
                          列表，每行为一个列表，表示序列（对应客户的购买记录）；
                            序列由列表组成（单次购买的商品）；
          min_support ：最小支持度，0~1之间，默认为0.4
    输出：列表，由元组组成，元组有两项，分别为：
            序列：元组，由子序列组成，子序列为frozenset；
            序列模式的支持度
    '''
    # 频繁项集阶段
    mapping, support_ls_1 = create_ls_1(dataset, min_support)
    ls_1 = [list(k) for k in support_ls_1]
    # 转换阶段
    transform_ds = transform(dataset, mapping)
    # 序列阶段
    ls_list = [ls_1]
    support_ls = support_ls_1.copy()
    k = 1
    while k >= 1 and len(ls_list[-1]) > 1:
        cs_k = cs_gen(ls_list[-1])
        ls_k, support_ls_k = calc_support(transform_ds, cs_k, min_support)
        if len(ls_k) > 0:
            ls_list.append(ls_k)
            support_ls.update(support_ls_k)
            k += 1
        else:
            break
    ls = list(itertools.chain(*ls_list))
    tr_mapping = {v: k for k, v in mapping.items()}
    ls = [[tr_mapping[k] for k in seq] for seq in ls]
    support_ls = {
        tuple([tr_mapping[i] for i in k]): v
        for k, v in support_ls.items()
    }
    # 最大化阶段
    ls, support_ls = max_ls(ls, support_ls)
    return list(support_ls.items())


if __name__ == '__main__':
    dataset = [[[30], [90]], [[10, 20], [30], [40, 60, 70]], [[30, 50, 70]],
               [[30], [40, 70], [90]], [[90]]]
    min_support = 0.25
    print(seq_patterns(dataset, min_support=0.25))
