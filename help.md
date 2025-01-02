===== 项目文档 ======

流程与设定

分类任务：暂定MNIST数据集，先实现最基本的fedavg+wireless结构。环境有env包实现。

联邦过程：

1.各客户端完成数据集的初始化。服务端和客户端初始化一个分类模型。

2.**选取客户端**。

3.被选中的客户端集合，使用本地数据集和收到的全局模型参数进行训练固定epoch，得到更新后的本地参数

4.上传梯度。**梯度有噪声**。

5.服务端全局聚合，更新参数。然后下发（也或许可以考虑噪声？），回到2。



=== env ===

只需考虑下传线路的时延。每个客户端有：发射功率p，信道增益h，相隔距离d（均为固定的）。传输速率：
$$
R_{k}^{r}=(n_i(r)·B)\log_{2}(1+\frac{p_kh_k^r(d_k^r)^{-\alpha}}{N_0n_i(r)})
$$
​	其中B一个资源块带宽大小，n是分配的资源块数量，显然有$\sum{n}<=n_{total}$。令Z为上传参数大小，则传输时间：$T_k^{com,r} = Z_k^r/R_k$。同时，每个客户端有：E:本地epochs、D：处理的样本数（数据集大小）、C:处理一个样本需要的cycle数、f：CPU频率。则计算时间：$T_k^{cmp,r}=EC_iD_i/f_k$。

​	则一轮联邦学习中，计算时间是$T=max(T^{cmp}+T^{com})$。能耗也可以获得，Ecmp+Ecom。

​	最后要按照gym的标准封装。



=== class design ===

client类：id、local_dataset、model、optimizer、attr_dict（包括：p，h，d，f）



==== args ====

--num_clients

==dataset相关

--dataset：数据集 MNIST/CIFAR10/CIFAR100

--alpha：狄利克雷特分布参数因子

--batch_size

--data_dir

--non_iid（store true）

--seed 随机数种子

==训练相关

--local_rounds

--global_rounds



