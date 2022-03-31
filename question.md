BEiT把同一张图用两个视角去看，一个是patch，也就是像素分割后的平铺的张量[n*patch_size]。另一个是visual_token，这个是用图片经过离散自编码器得到的结果。
​

用mask部分的patches，也就是置零部分的patch。输入模型中，以完整的visual_token作为结果，相当于学习图片的embeding。

如果在graph中对应去看，自然的思路是：将节点的feature用某种方式降维，得到token，之后用mask后的features，学习token，用预训练后的模型，加上简单的nn.Linear(token_size,node_class)得到分类结果。难点和重点在于学习token上。在BeIt中有一句话**Because the latent visual
tokens are discrete, the model training is non-differentiable**，翻译成中文是**由于潜在的visual token是离散的，模型的训练是无差异的**（在论文2.1.2 visual token中）。整句话我不太清楚是什么意思，为什么visual_token需要是离散的。

而且对于graph而言，离散自编码器可能不太行。因为graph中的feature往往是稀疏的。自编码器的原理是根据一个分布的映射，稀疏向量的映射恐怕不太行。我个人的想法是PCA降维得了，但是肯定不是离散的。关于生成token的角度，学长有什么建议吗，可以给我一个方向那就太感谢了。