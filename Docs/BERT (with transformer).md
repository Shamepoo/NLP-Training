# BERT (Bi-directional Encoder Representation from Transformers)

BERT是一种预训练技术，其网络架构使用的是多层Transformer结构，最大的特点是抛弃了传统的RNN和CNN，通过Attention机制将任意位置的两个单词的距离转换成1，有效的解决了NLP中棘手的长期依赖问题[1]。Transformer的结构在NLP领域中已经得到了广泛应用。

## 1. Transformer

### 1. Encoder-Decoder

注意力模型可以看作一种通用的思想，本身并不依赖于特定框架。但目前大多数注意力模型附着在Encoder-Decoder框架下。

简单地说，encoder–decoder是模拟人类认知的一个过程。由从字面意思上来看，其由encoder和decoder两部分组成。Encoder可以记忆和理解信息，并提炼信息通常会形成一个低秩的向量（相对于输入），在这一过程中，可以只依赖于输入的信息，也可以在模型构建时添加先验规则、attention机制等等。就如同人在接触到一些新的信息之后通常也会利用常识与先验知识去试图理解与抽象化这些信息一样。而中间的编码形式就对应于人脑中的记忆（which is invisible）。而decoder回忆与运用这些信息，再将低秩的加工后的信息抽取出来，这时也可以混合其它信息，解码成需要用的形式。Encoder–Decoder中的参数训练对应人脑对这种信息处理和运用的方法的能力习得过程。

在自然语言处理的Seq2Seq模型中，Encoder一般用来将输入序列转化成一个固定长度的向量，而Decoder则是将输入的固定长度向量解码成输出序列。一般采用循环神经网络，如RNN，LSTM，GRU，进行编码和解码。因为神经网络具有学习且记忆的能力，Encoder-Decoder模型可以将所需要的信息提取并利用，常用在神经机器翻译（NMT）,Question&Answer等领域。

### 2. Network Structure

Transformer的网络架构如图1所示，Transformer是一个encoder-decoder的结构，由N个编码器和解码器堆叠形成。图1的左侧部分为编码器，由Multi-Head Attention和一个全连接组成，用于将输入语料转化成特征向量。右侧部分是解码器，其输入为编码器的输出以及已经预测的结果，由Masked Multi-Head Attention, Multi-Head Attention以及一个全连接和softmax函数组成，用于输出最后结果的条件概率。

![](\pic\transformer.png)

上图中，有一些模块需要进一步解释。

#### 1) Add & Normalize

Transformer中采用了残差网络[1]中的short-cut结构，目的是解决深度学习中的退化问题。Add&Norm模块中Add进行的操作可用如下公式表示：

​											h(x) = x + f(x)

而Norm的操作为Layer Normalization, 通过对层激活值的归一化，加速训练过程。

#### 2) Encoder-Decoder Attention

在解码器中，Transformer block比编码器中多了个encoder-cecoder attention。在encoder-decoder attention中， ![[公式]](https://www.zhihu.com/equation?tex=Q) 来自于解码器的上一个输出， ![[公式]](https://www.zhihu.com/equation?tex=K) 和 ![[公式]](https://www.zhihu.com/equation?tex=V) 则来自于与编码器的输出。

由于在机器翻译中，解码过程是一个顺序操作的过程，也就是当解码第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个特征向量时，我们只能看到第 ![[公式]](https://www.zhihu.com/equation?tex=k-1) 及其之前的解码结果，需要将第k个向量以后的特征mask掉，因此这种情况下的multi-head attention也被叫做masked multi-head attention。**这里即是在Scaled dot-product,图x中的mask模块？？？**

#### 3) Positional Encoding

值得注意的是，Transformer仍然需要一个捕捉顺序序列的方式去更完整的获取context。

Here comes Positional Embedding.

它会在词向量中加入了单词的位置信息，这样Transformer就能区分不同位置的单词了。

这里使用到位置编码的公式为：

![](\pic\formula2.png)

采用三角函数的周期性，对不同的word embedding采用不同频率的三角函数（sin，cos）来表示该特征所在的位置。

## 2. BERT



### 1. Network Structure

BERT的网络架构如下图所示：



![](\pic\BERT.png)

其中Trm模块是由Transformer的Encoder组成。

### 2. Input Representation

上图中网络输入模块Ei的结构如下图所示，由 Token Embedding，Position Embedding，Segment Embedding三部分组成。

![](\pic\BERT input.png)

#### 1) Token Embedding

WordPiece Embedding[2] 是一个将单词划分成一组有限的公共子词单元的算法。该算法能在单词的有效性和字符的灵活性之间取得一个折中的平衡。上图中的playing被拆分为play和###ing。

在这一层所进行的操作还有加入了两个特殊符号[CLS], [SEP]。[CLS]符号代表该特征用于分类模型，对于非分类模型，可以不使用该符号。而[SEP]符号表示断句，用于断开输入文本中的两个句子。

#### 2) Position Embedding

As mentioned before

#### 3) Segment Embedding

在一个句子对A,B中，如果A是B的前文，B是A的后文，则句子A的特征值EA为0，句子B的特征值EB为1。

### 3. Pre-training Task

BERT是一个多任务模型，其任务有两个自监督任务组成，MLM和NSP。

#### 1) Masked Language Model

Masked Language Model[3]是指在训练的时候随即从输入预料上mask掉一些单词，然后通过的上下文预测该单词。

在BERT实验中，15%的WordPiece Token会被随机Mask掉。在训练模型时，一个句子会被多次喂到模型中用于参数学习，但是Google并没有在每次都mask掉这些单词，而是在确定要Mask掉的单词之后，80%的时候会直接替换为[Mask]，10%的时候将其替换为其它任意单词，10%的时候会保留原始Token。这一步的目的是在Fine-tuning时降低模型未见过单词的可能性，保持Transformer对每个输入token的分布式表征。

#### 2) Next Sentence Prediction

NSP的任务是判断句子B是否是句子A的下文。如果是的话输出’IsNext‘，否则输出’NotNext‘。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存在图4中的`[CLS]`符号中。

### 4. Fine-tune

对于不同的NLP任务，使用同一个pre-train的大模型的参数时，需要进行fine-tuning来使模型更容易学习到task specific的信息并且可以针对不同的需求对模型进行优化，加速训练过程，并且可以适用于较小的数据集。



[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[2] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144.

[3] Wilson L Taylor. 1953. cloze procedure: A new tool for measuring readability. Journalism Bulletin, 30(4):415–433.

