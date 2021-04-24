# 注意力机制（Attention Mechanism）

人工智能领域里的注意力机制借鉴了人类视觉的选择性注意机制，即将注意力集中在需要关注的重点目标区域（注意力焦点），这是人类利用有限的注意力资源从大量信息中快速筛选出高价值信息的手段，是人类在长期进化中形成的一种生存机制，人类视觉注意力机制极大地提高了视觉信息处理的效率与准确性。

在深度学习中，由于计算能力和优化算法的限制，Attention Mechanism借助人脑处理信息过载的方式，提高了神经网络处理信息的能力。随着Attention is All You Need 一文的发布，注意力模型在最近这几年被广泛使用在自然语言处理、图像识别及语音识别等各种不同类型的深度学习任务中，是深度学习技术中最值得关注与深入了解的核心技术之一。

## Look through of Attention Mechanism

**Attention机制的实质其实就是一个寻址（addressing）的过程**，如图所示：给定一个和任务相关的查询**Query**向量 **q**，和以(K,V)形式储存的上下文，通过计算与**Key**的注意力分布a并附加在**Value**上，从而计算**Attention Value**，其中s为注意力打分机制，可代表Q,K 之间的相关性。这个过程实际上是**Attention机制缓解神经网络模型复杂度的体现**：不需要将所有的N个输入信息都输入到神经网络进行计算，只需要从X中选择一些和任务相关的信息输入给神经网络。

![General Attention Detail](\pic\General Attention Detail.png)

由上图可见，注意力机制的通用表达式可以写为：

​										Attention(Q,K,V) = softmax(QK(T))·V

注意力打分机制为点积打分，即QK(T)。

下面是一些不同Attention机制的原理介绍。（本项目中用到了Scaled Dot-Product Attention 和 Multi-head Attention）

### Soft&Hard Attention

Soft Attention也被叫做Traditional Attention，是确定的(deterministic)。

下图是一个Soft Attention Unit，其中Query为C，Key，Value均为yi。

![Soft Attention](\pic\Soft Attention.png)

与传统Attention不同的是，soft attention的打分机制为**加性模型**，即用tanh激活函数限制过大或过小数值，并为Q，K乘上两个可被学习的权重矩阵来决定该条query或key的重要性。[nlp中的Attention注意力机制+Transformer详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/53682800)

公式如下：

​													mi = tanh(yiWyi + CWc)

​													z = sum(sn·yn)

而Hard Attention是随机的(Stochastic)，注意力只随机地关注到某一个位置上的信息。其有两种实现方式，

1. 选取最高概率的输入信息
2. 可以通过在注意力分布式上随机采样的方式实现

![Hard Attention](\pic\Hard Attention.png)

与Soft Attention一样，Hard Attention也采用了加性模型的打分机制，而在输出最终的Attention Value时，不再像Soft Attention一样采用将所有输入和其对应分数点积和的形式，而是根据上述两种实现方式，选取一个输入对应的输出作为结果。

### Multi-head Attention

在介绍多头注意力机制之前，对Scaled Dot-Product Attention和Self Attention的了解是必要的。

#### Scaled Dot-Product Attention

如果我们采用如下图所示的缩放点积打分机制，在K与Q点积运算后除以K的维度的开方，其他计算步骤保持不变，就得到了缩放点积注意力模型。除以K维度开方的目的是防止结果过大。

![Scaled Dot-Product Attention](\pic\Scaled Dot-Product Attention.png)

该操作可以表示为

![formula1](\pic\formula1.png)

值得注意的是，在进行scale操作之后，图中有一个可选择的(OPT)Mask模块，其作用在下一节会介绍。

#### Self-Attention

对于Self-Attention，Q,K,V三个矩阵均来自同一个输入x，对x乘以不同的初始可学习权重矩阵即可依次得到，然后将他们使用在Scaled Dot-Product Attention Model中。对于一个给定序列，使用Self-Attention可以获得序列中任意元素与所有元素之间的相关性。它简化了计算，解决了循环神经网络不能进行并行计算的问题，并且提升了序列任务中对长距离依赖的捕捉能力[Attention is All you Need]。

#### Multi-head Attention

在多头注意力机制中，我们讲Q,K,V 以h个不同的形式表示，即使用不同的初始化权重矩阵，最后再将其结果结合起来，每一个不同的表示形式被称作一个header，如下图所示：

<img src="D:\NLP\基础知识\pic\multi-head-attention.png" alt="multi-head-attention" style="zoom:50%;" />

Multi-head Attention是对self-attention的完善，它为attention层提供了多个representation subspaces，对于每一个subspace，都独立维护一套Q/K/V的权值矩阵，并且在h个时间点去计算这些矩阵。最后将这h个输出的header concatenate在一起，乘以一个权重矩阵W0，即线性层，将最终的输出转化成希望的大小。