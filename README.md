# service_recommendation

本仓库为我的毕业论文《群智化云服务API推荐方法》中相关代码的存放仓库。

## 语言与运行前准备

本仓库中的代码使用 Python 3 语言，在 Python 3.6 和 3.7 的环境下都可以运行成功。

所需的运行库如下：

- numpy
- pandas
- scikit-learn（矩阵准备、Mean Shift算法需要使用）
- mlxtend（Apriori算法需要使用）
- gensim（Word2Vec算法需要使用）

本仓库中的代码使用MovieLens的1m和20m数据集，并拼接成所需数据集。由于MovieLens的提供方不允许第三方再发布数据集，故请到[MovieLens数据集页面][1]自行下载，并按以下步骤拼接：

1. 在项目目录中建立`dataset`目录，在其下再建立`ml-out`目录；
2. 将1m和20m的数据集内地所有文件分别解压到`dataset`目录下的`ml-1m`和`ml-20m`目录中；
3. 在`data_prepare`目录中执行`data_prepare.py`。

如果还需要生成缩减数据集，请按以下步骤执行：

1. 在`dataset`目录下建立`ml-out-sample`目录；
2. 在`data_prepare`目录中执行`data_reduce.py`。执行前对该文件内参数进行检查，配置自己想要的参数。

建立矩阵的步骤：

1. 在项目目录下建立`temp`目录；
2. 在`data_prepare`目录中执行`matrix_preparation.py`。执行前对该文件内参数进行检查，配置自己想要的参数。

各算法的文件分开存放，请在代码所在目录执行。执行前对该文件内参数进行检查，配置自己想要的参数（尤其是路径，默认为缩减后的数据）。

## 协议

使用署名-相同方式共享 4.0 国际（CC-BY-SA 4.0）协议分享。也就是说，在使用代码时，请标注有效的来源（如名称、GitHub链接）。如果作为学术研究，希望可以引用该论文（虽然我的论文写得并不好）。

## 相关论文

> 基于本论文的研究成果，另外还有出版的论文，可优先引用。

```
# 原论文附注
Wang L, Zhang Y, Zheng X, Yu Q, Chen S, Ding J. Singular value decomposition-based behavior-aware cloud service application programming interfaces recommendation for large-scale software cloud directory platforms. Concurrency Computat Pract Exper. 2022;e7121. doi: 10.1002/cpe.7121

# GB/T 7714
Wang L, Zhang Y, Zheng X, et al. Singular value decomposition‐based behavior‐aware cloud service application programming interfaces recommendation for large‐scale software cloud directory platforms[J]. Concurrency and Computation: Practice and Experience, 2022, 34(21): e7121.

# MLA
Wang, Lei, et al. "Singular value decomposition‐based behavior‐aware cloud service application programming interfaces recommendation for large‐scale software cloud directory platforms." Concurrency and Computation: Practice and Experience 34.21 (2022): e7121.

# APA
Wang, L., Zhang, Y., Zheng, X., Yu, Q., Chen, S., & Ding, J. (2022). Singular value decomposition‐based behavior‐aware cloud service application programming interfaces recommendation for large‐scale software cloud directory platforms. Concurrency and Computation: Practice and Experience, 34(21), e7121.
```

> 鉴于该论文为学士学位论文，未普遍出版，请自行决定是否引用。

```
# GB/T 7714
丁俊尧. 群智化云服务API推荐方法[D]. 南京林业大学, 2019.

# MLA
丁俊尧. 群智化云服务API推荐方法. Diss. 南京林业大学, 2019.

# APA
丁俊尧. (2019) 群智化云服务API推荐方法. dissertation. 
```

[1]: https://grouplens.org/datasets/movielens/
