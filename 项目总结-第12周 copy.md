## 要解决的问题
观察到，在图神经网络中，features过于稀疏，使得不能从features中学习到和邻居的差异化的信息。并且在梯度传递的时候，有边的邻居具有相同的梯度，使得attention的取值在一个邻域内取向同质化。
## 如何评判一个邻域内的attention取值同质化

跟踪在训练的时候，节点attention的方差

## 如何解决

### 用编码代替features

编码方式GAE
#### GAE(1hop 1W)  256 64
 data(cora_geom)        GAT(8heads qkv qkv)         86.5392, 0.89

 data(cora_geom)        GAT(8heads qkv Graph)       86.5191, 0.8

 data(cora_std)         GAT(8heads qkv qkv)         79.4800, 1.25

 data(cora_std)         GAT(8heads qkv Graph)       79.6600, 1.91

 data(citeseer_std)     GAT(8heads qkv qkv)         68.3800, 1.32

 data(citeseer_std)     GAT(8heads qkv Graph)       69.2000, 0.41

 data(citeseer_geom)     GAT(8heads qkv qkv)         75.3817, 1.57

 data(citeseer_geom)     GAT(8heads qkv Graph)       75.1372, 1.41

总结：总体来看，区别不大，因为GAE编码本身就是基于1hop邻居编码，会使得邻居之间的编码相似，这一点其实和attention是相同的
#### GAE(2hop 1W)  256 64
data(cora_geom)     86.3179, 0.89
data(cora_std)      79.0500, 1.14
data(citeseer_geom) 73.0952, 1.95
data(citeseer_std)  67.3700, 1.06

总结：没有本质上的提升，用2hop矩阵代替临界矩阵似乎并没有能够提升模型的效果。院士是没有解决attention传递问题

### 预训练 

用有self-loop的adj预训练

### 初始化

用编码矩阵初始化features的降维矩阵

引入预训练模块

GAE(1hop W) GAT(QKV QKV) 100epoch 加入 W 1e-4

cora 86.5191, 1.49
citeseer 75.1436, 1.56


## 总结

GAE所采用的邻居不能是图空间的邻居，这是由于GAT或者GAT本身就是在沿着图的边传递的，如果把原始的adj作为目标去做预训练，本质上和GNN做的是一件事。

所以需要找到另外一个尺度的邻居来作为预训练的目标

## 下面，我们找到选取什么样的节点最为合适

GEOM提供了思路，但是由于数学复杂性，并没有打算采用

features越接近，越可能是同一类。合理

1. 对features点积排名,排名前x的节点中，类别相同的占比率
chameleon

| x        | rate   |
| -------- | ------ |
| baseline | 0.2    |
| edge     | 0.2490 |
| 10       | 0.5480 |
| 20       | 0.4776 |
| 30       | 0.4326 |
| 40       | 0.4009 |
| 50       | 0.3821 |
| 60       | 0.3648 |
| 70       | 0.3515 |
| 80       | 0.3404 |
| 90       | 0.3317 |
squirrel

| x        | rate   |
| -------- | ------ |
| baseline | 0.2    |
| edge     | 0.2190 |
| 10       | 0.4356 |
| 20       | 0.3531 |
| 30       | 0.3190 |
| 40       | 0.2995 |
| 50       | 0.2870 |
| 60       | 0.2790 |
| 70       | 0.2730 |
| 80       | 0.2676 |
| 90       | 0.2646 |









cora

| x        | rate   |
| -------- | ------ |
| baseline | 0.167  |
| edge     | 0.82   |
| 10       | 0.5989 |
| 20       | 0.5448 |
| 30       | 0.5168 |
| 40       | 0.4977 |
| 50       | 0.4848 |
| 60       | 0.4752 |
| 70       | 0.4679 |
| 80       | 0.4617 |
| 90       | 0.4566 |

<!-- 1. adj相似的越可能是同一类
chameleon

| x        | rate   |
| -------- | ------ |
| baseline | 0.2    |
| 10       | 0.4491 |
| 20       | 0.4089 |
| 30       | 0.3886 |
| 40       | 0.3734 |
| 50       | 0.3644 |
| 60       | 0.3648 |
| 70       | 0.3568 |
| 80       | 0.3395 |
| 90       | 0.3395 |

squirrel

| x        | rate   |
| -------- | ------ |
| baseline | 0.2    |
| 10       | 0.4201 |
| 20       | 0.3451 |
| 30       | 0.3152 |
| 40       | 0.2951 |
| 50       | 0.2811 |
| 60       | 0.2704 |
| 70       | 0.2604 |
| 80       | 0.2551 |
| 90       | 0.2512 |

cora

| x        | rate   |
| -------- | ------ |
| baseline | 0.2    |
| 10       | 0.7470 |
| 20       | 0.7058 |
| 30       | 0.6699 |
| 40       | 0.6453 |
| 50       | 0.6256 |
| 60       | 0.6092 |
| 70       | 0.5966 |
| 80       | 0.5824 |
| 90       | 0.5689 | -->

小尝试
MODEL：mix-hop+GCN+norm

GCN的adj改成

adj*a+(features相关性)*b+(adj相关性)*c

a,b,c是初始化为1，可学习的参数

features相关性:

features=adj.dot(features)

features相关性=(features,features.T)

adj相关性

adj=adj.dot(adj)

adj相关性=(adj,adj.T)

sota:GEOM cora 88.1690, 1.17

citeseer 76.8703, 1.66

进一步调整超参数，可能可以使得模型能力进一步提升，在少部分数据集上达到sota，但是由于方法本身过于简单缺少创新，并且目测难以在大多数数据集上达到sota，暂时不打算对上述模型进一步改进。

但是能够发现，直接根据features寻找邻居在某种程度上可能可行。

其实GEOM和我们做了相同的工作，isomap是GEOM论中文的一个表现突出的降维方案。只不过他计算的features的相关度是根据瑞士卷进行降维的，在每个点仅仅取用少数邻居的情况下，两种方法是等价的。

有一个细节就是

features采取几阶邻域，是否需要归一化

adj_i采取几阶邻域，是否需要归一化

根据实验来说，经验上features采取1hop邻居，需要归一化。**如果确定要做，这一步需要进一步探究**

adj_i需要1hop，需要归一化。如果不归一化，相当于承认度数大的节点有天生的attention取值的优势，在经验看来，这似乎并不成立.

总的来说GEOM遇到的问题是，没有采用1hop邻居的features作为降维，使得邻居的信息并不为完整。

## 下一阶段

首先考虑Q,K的梯度传递方式

考虑
dataset: cora , k : 0 
baseLine : 0.1429
edge homogeny: 0.8252
top 10 : 0.3909
top 20 : 0.3849
top 30 : 0.3802
-------------------------
dataset: cora , k : 1 
baseLine : 0.1429
edge homogeny: 0.8252
top 10 : 0.5989
top 20 : 0.5448
top 30 : 0.5168
-------------------------
dataset: cora , k : 2 
baseLine : 0.1429
edge homogeny: 0.8252
top 10 : 0.6311
top 20 : 0.5942
top 30 : 0.5675
-------------------------
dataset: cora , k : 3 
baseLine : 0.1429
edge homogeny: 0.8252
top 10 : 0.5643
top 20 : 0.5473
top 30 : 0.5254
-------------------------
dataset: citeseer , k : 0 
baseLine : 0.1667
edge homogeny: 0.7222
top 10 : 0.5339
top 20 : 0.4917
top 30 : 0.4679
-------------------------
dataset: citeseer , k : 1 
baseLine : 0.1667
edge homogeny: 0.7222
top 10 : 0.6058
top 20 : 0.5738
top 30 : 0.5563
-------------------------
dataset: citeseer , k : 2 
baseLine : 0.1667
edge homogeny: 0.7222
top 10 : 0.6441
top 20 : 0.6085
top 30 : 0.5899
-------------------------
dataset: citeseer , k : 3 
baseLine : 0.1667
edge homogeny: 0.7222
top 10 : 0.6271
top 20 : 0.5979
top 30 : 0.5841
-------------------------
dataset: pubmed , k : 0 
baseLine : 0.3333
edge homogeny: 0.7924
top 10 : 0.6560
top 20 : 0.6492
top 30 : 0.6451
-------------------------
dataset: pubmed , k : 1 
baseLine : 0.3333
edge homogeny: 0.7924
top 10 : 0.6871
top 20 : 0.6718
top 30 : 0.6637
-------------------------
dataset: pubmed , k : 2 
baseLine : 0.3333
edge homogeny: 0.7924
top 10 : 0.6916
top 20 : 0.6887
top 30 : 0.6850
-------------------------
dataset: pubmed , k : 3 
baseLine : 0.3333
edge homogeny: 0.7924
top 10 : 0.6780
top 20 : 0.6711
top 30 : 0.6652
-------------------------
dataset: chameleon , k : 0 
baseLine : 0.2000
edge homogeny: 0.2490
top 10 : 0.2288
top 20 : 0.2143
top 30 : 0.2121
-------------------------
dataset: chameleon , k : 1 
baseLine : 0.2000
edge homogeny: 0.2490
top 10 : 0.5480
top 20 : 0.4776
top 30 : 0.4326
-------------------------
dataset: chameleon , k : 2 
baseLine : 0.2000
edge homogeny: 0.2490
top 10 : 0.4239
top 20 : 0.3855
top 30 : 0.3654
-------------------------
dataset: chameleon , k : 3 
baseLine : 0.2000
edge homogeny: 0.2490
top 10 : 0.3572
top 20 : 0.3340
top 30 : 0.3207
-------------------------
dataset: squirrel , k : 0 
baseLine : 0.2000
edge homogeny: 0.2190
top 10 : 0.2122
top 20 : 0.1961
top 30 : 0.1914
-------------------------
dataset: squirrel , k : 1 
baseLine : 0.2000
edge homogeny: 0.2190
top 10 : 0.4356
top 20 : 0.3531
top 30 : 0.3190
-------------------------
dataset: squirrel , k : 2 
baseLine : 0.2000
edge homogeny: 0.2190
top 10 : 0.3427
top 20 : 0.2973
top 30 : 0.2773
-------------------------
dataset: squirrel , k : 3 
baseLine : 0.2000
edge homogeny: 0.2190
top 10 : 0.2860
top 20 : 0.2600
top 30 : 0.2534
-------------------------