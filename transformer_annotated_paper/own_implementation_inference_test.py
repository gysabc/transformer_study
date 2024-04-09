import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# 编码器解码器的基本结构和逻辑定义
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        '''

        :param encoder:
        :param decoder:
        :param src_embed: 包含embedding和位置编码的Sequential容器
        :param tgt_embed:
        :param generator:
        '''
        # 下面这句话的用处：EncoderDecoder类继承nn.Module
        # super(EncoderDecoder, self).__init__()就是对继承自父类nn.Module的属性进行初始化。
        # 而且是用nn.Module的初始化方法来初始化继承的属性。
        super(EncoderDecoder, self).__init__()  # 继承了父类nn.Module的属性
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        '''
        对src进行编码操作，然后与src的mask、目标以及目标的mask一起进行解码操作
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        '''对源序列进行编码的具体操作：先对源序列进行embedding操作，然后使用encoder函数(外部传入的)对嵌入进行编码'''
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        '''
        对经过编码的源文本进行解码的具体操作：先将目标文本进行embedding操作，然后再与经过编码的源文本以及两者的mask一起输入到解码器decoder
        中进行解码
        :param memory: 理解为经过编码的源文本
        :param src_mask:
        :param tgt:
        :param tgt_mask:
        :return:
        '''
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# linear + softmax层的基本结构
class Generator(nn.Module):
    "Define standard linear + softmax generation step.即解码器之后的线性变换和softmax操作"

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  # 输出维度是词典的大小，然后softmax之后就是每个词的概率

    def forward(self, x):
        '''
        会先对当前模型进行线性变换操作，然后进行log_softmax操作
        self.proj(x)的调用方式也是因为__call__ 方法的存在
        '''
        return log_softmax(self.proj(x), dim=-1)


# 用于将一个模块克隆N次
def clones(module, N):
    '''
    Produce N identical layers.
    :param module:某一个网络模块
    :param N: 克隆的次数，即从原来的一个变成几个
    :return: 返回克隆之后的模块的列表
    '''

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 编码器结构类
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 每一个layer相当于一个编码器模块，总共需要n个编码器模块
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  # 建立了一个层规范化的对象

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            # 编码器有n个，用循环将输入input (and mask)依次输入到每一个编码器模块中
            # 所谓依次即：前一个编码器模块的输出作为下一个编码器模块的输入
            x = layer(x, mask)
        # 经过n个编码器模块的计算之后，对最后一个编码器模块的输出施加一个层规范化，然后返回
        '''
         nn.Module 类实现了 Python 中的特殊方法 __call__，使得我们可以将模型实例对象当作函数来调用,
         并且将输入张量作为参数传递给该函数;
         在 LayerNorm 类的 __call__ 方法中，又会调用 forward 方法，并将输入张量 x 作为参数传递给该方法;
         即因为__call__的存在，使得可以把模型实例对象当作函数来调用，而__call__会帮我们去调用forward 方法。
        '''
        return self.norm(x)


# 层规范化类
class LayerNorm(nn.Module):
    "Construct a layerNorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # nn.Parameter类则用于将某些张量作为可学习的参数进行训练
        # 在PyTorch中，nn.Parameter和nn.parameter.Parameter是等价的，都是用于将一个Tensor转换为可训练的参数。
        # nn.parameter.Parameter是nn.Parameter的别名，两者可以互换使用。
        # self.a_2 = nn.Parameter(torch.ones(features))
        # self.b_2 = nn.Parameter(torch.zeros(features))
        self.a_2 = nn.parameter.Parameter(torch.ones(features))
        self.b_2 = nn.parameter.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # PyTorch会自动进行广播，将self.a_2的第一维扩展为batch_size和seq_len，以便进行逐元素相乘的操作
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差连接类
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        Apply residual connection to any sublayer with the same size.
        dropout作用在子层的输出上；
        此处为了代码的简单性，先对输入x进行规范化，然后经过子层之后施加dropout，最后再残差连接
        '''
        return x + self.dropout(sublayer(self.norm(x)))


# 定义每个编码器的结构：一个自注意力层，一个FFN
class EncoderLayer(nn.Module):
    "Encoder is m ade up of self-attn and feed forward (defined below)即实现了transformer整个模型架构的左侧的一个编码器的所有逻辑结构"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 使用clones方法制造两个一样的残差连接模块
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # 第一个残差连接模块作用在自注意力子层上：
        # 会调用残差连接类SublayerConnection的forward方法，先进行规范化，再进行自注意力的操作，然后dropout，最后残差连接
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个残差连接模块作用在FFN子层上：
        # 将自注意子层最终的输出作为x，先进行规范化，再进行FFN的操作，然后dropout，最后残差连接
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking.带有masking的通用n层解码器结构"

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            # 调用单个解码器类（包含单个解码器要进行的所有操作）
            x = layer(x, memory, src_mask, tgt_mask)
        # 和解码器一样，经过n个解码器模块的计算之后，对最后一个解码器模块的输出施加一个层规范化，然后返回
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)，即多了一个对编码器堆栈输出的多头注意力子层"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size

        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  # 3个子层对应3个残差连接

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        # 第一个残差连接模块作用在自注意力子层上，没有memory的参与
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个残差连接模块作用在新增子层上，有memory的参与
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    '''Mask out subsequent positions.遮挡后续位置(只关注当前词及其前面的位置)，翻译为掩码矩阵。实现原理如下：
    构建一个维度是(size, size)的矩阵，size就是目标序列的长度，每次在预测序列的某个词（即一行中的某个元素）的时候，
    看该元素前一个元素所在列中0从哪里开始（实际上subsequent_mask每一列自主对角线开始到最后一个元素都被设置了0，
    这样在预测当前词的时候就不被允许去考虑当前词之后的词）'''
    attn_shape = (1, size, size)
    # 这里应该是生成一个和目标序列一样维度的Tensor：torch.ones(attn_shape)；
    # 然后将这个Tensor的主对角线以上的元素保留，其余元素置为0，这样就可以起到“在预测当前位置的元素时不去考虑后续的位置”
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # subsequent_mask是经过处理之后的只包含0和1的Tensor，将其与0比较，返回一个只包含true和false的Tensor
    return subsequent_mask == 0


# # 下面这两个函数用来举例子，加深对掩码矩阵的理解
# # 定义了一个布尔变量RUN_EXAMPLES，并定义了一个函数show_example，该函数接受一个函数和一个参数列表作为参数。
# # 如果当前文件是主文件（即被直接运行的文件）并且RUN_EXAMPLES为True，则调用传入的函数并传入参数列表。
# # 这段代码的作用是方便在调试时运行一些示例代码，但在正式运行时可以将RUN_EXAMPLES设置为False以避免运行示例代码。
# RUN_EXAMPLES = True


# def show_example(fn, args=[]):
#     if __name__ == "__main__" and RUN_EXAMPLES:
#         return fn(*args)


# def example_mask():
#     LS_data = pd.concat(
#         [
#             pd.DataFrame(
#                 {
#                     "Subsequent Mask": subsequent_mask(4)[0][x, y].flatten(),
#                     "Window": y,
#                     "Masking": x,
#                 }
#             )
#             for y in range(4)
#             for x in range(4)
#         ]
#     )

#     return (
#         alt.Chart(LS_data)
#         .mark_rect()
#         .properties(height=250, width=250)
#         .encode(
#             alt.X("Window:O"),
#             alt.Y("Masking:O"),
#             alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
#         )
#         .interactive()
#     )


# show_example(example_mask)

def attention(query, key, value, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention'。这里的计算是一次性计算了h个头的注意力（主要体现在Tensor的维度上）
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    '''
    # query, key, value的维度都是(nbatches, self.h, -1, self.d_k)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # key.transpose(-2, -1) 是将张量 key 在倒数第二维和倒数第一维之间进行转置。
    # 例如，如果 key 的形状为  (nbatches, self.h, -1, self.d_k)，
    # 则 key.transpose(-2, -1) 的形状为  (nbatches, self.h, self.d_k, -1)
    # 而 query 的形状为  (nbatches, self.h, -1, self.d_k)，两个张量相乘后的形状为  (nbatches, self.h, -1, -1)，
    # 所以 scores 的形状为  (nbatches, self.h, -1, -1)。
    # scores的维度理解为：nbatches个(self.h, -1, -1),里面的(self.h, -1, -1)表示h个(-1,-1),
    # (-1,-1)表示一个矩阵,-1表示序列长度
    if mask is not None:
        # scores是一个PyTorch张量，并且它可以调用masked_fill方法，因为该方法是PyTorch张量的一部分
        '''
        masked_fill方法的作用是将张量中的一些元素替换为其他值。在这个例子中，当mask张量中的元素为0时，
        scores张量中对应的元素将被替换为-1e9。这个值通常被用作一个极小的数，以便在后续的计算中过滤掉这些元素
        '''
        # 将需要mask的位置的评分值置为一个很大的负数，这样在softmax之后就会趋近于0
        scores = scores.masked_fill(mask == 0, -1e9)
    # dim=-1参数指定了在最后一个维度上进行softmax操作
    # p_attn的维度还是(nbatches, self.h, -1, -1)，但是最后一个维度的含义变了，变成了每个位置的注意力权重(即归一化之后的概率值)，
    # 所以最后两个维度(-1,1)理解为序列中每一个词(即第一个-1)都对应一串注意力权重(即第二个-1)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        '''
        在深度学习中，dropout是一种常用的正则化技术，可以有效地减少过拟合。
        在transformer中，dropout被应用于注意力机制中的softmax操作，以减少模型对某些输入的过度依赖。
        具体来说，dropout会随机地将一些神经元的输出置为0，从而强制模型在训练过程中学习到更加鲁棒的特征表示。
        在这里，dropout被应用于注意力机制的输出p_attn上，以减少模型对某些输入的过度依赖。
        '''
        p_attn = dropout(p_attn)  # p_attn的维度没有变化(nbatches, self.h, -1, -1),即(nbatches, self.h, -1_seq_length, -1_seq_length_prob)
    # value的维度是(nbatches, self.h, -1, self.d_k),即(nbatches, self.h, -1_seq_length, self.d_k)
    # matmul(p_attn, value)的维度应该是(nbatches, self.h, -1, self.d_k)，即(nbatches, self.h, -1_seq_length, self.d_k)；
    # 从最后两个维度的含义上看，这个乘法的含义是：对于每个词，都有一个当前词的上下文向量的子向量(d_k个，因为原来是512，然后8个头，每个头d_k个)
    return torch.matmul(p_attn, value), p_attn  # 由于p_attn和value至少是二维的，所以这里的乘法还需要进一步搞清楚具体的意义？


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        '''
        断言语句用于检查d_model是否能够被h整除。
        如果这个条件不为真，那么程序将会抛出一个AssertionError异常，以便在调试时能够快速定位问题
        '''
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # nn.Linear是一个类，这里它可以将d_model维度的张量映射到d_model维度的张量。
        # 这里将线性层克隆4次是因为在一次注意力操作中有4个linear层，分别用于对qkv的线性变换以及最终输出的线性变换。
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2(多头注意力的具体操作)"
        if mask is not None:
            # Same mask applied to all h heads.
            '''
            mask.unsqueeze(1):在mask的第二个维度上增加一个维度，这样做的目的是为了方便后续的计算(因为后面qkv的维度都是(nbatches, self.h, -1, self.d_k))。
            注意：下标从0开始，所以第二个维度的下标为1。
            所以如果原来的维度为[batch_size, seq_len]，那么增加维度之后的维度为[batch_size, 1, seq_len]。
            (这里关于mask的每一个维度分别代表什么还需要进一步分析？)
            '''
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        '''
        一些变量的含义还需要进一步分析，例如nbatches？
        .transpose(1, 2)转置操作不仅是维度的名称发生了变化，元素是怎么变化的还需要进一步分析？
        '''
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.即同时处理一批的数据，也同时处理每一条数据的多个头
        # x的维度是(nbatches, self.h, -1_seq_length, self.d_k)
        # self.attn的维度是(nbatches, self.h, -1_seq_length, -1_seq_length_prob)
        # 这里为什么要返回注意力，还要用self.attn这个类成员变量来存储，有什么用，也没看到再使用它啊？
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        # 关于contiguous()方法的详细解释，见https://zhuanlan.zhihu.com/p/64551412
        # 连续和不连续：Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致
        # transpose方法：与原Tensor是共享内存中的数据，不会改变底层数组的存储，因此会导致转置之后的Tensor不连续
        # 因此需要调用contiguous方法，将Tensor在内存中变成连续分布(即让逻辑上转置后的Tensor按行展开的结果和实际一维数组存储的顺序一致)，
        # 这样才能调用view方法(view 方法要求Tensor是连续的)
        # 先将x的维度变为(nbatches, -1_seq_length, self.h, self.d_k)，然后再经过一个线性变换，得到最终的输出
        x = (
            x.transpose(1, 2)  # 变成(nbatches, -1_seq_length, self.h, self.d_k)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)  # 将最后两个维度合并(按行)，变成(nbatches, -1_seq_length, d_model)
        )
        del query
        del key
        del value
        # 用之前克隆的最后一个线性层对x进行线性变换，得到最终的输出
        # 经过线性变换之后，输出的维度应该是(nbatches, -1_seq_length, d_model)？
        # 根据nn.Linear的官方参数解释，这里的d_model应该是称为特征数
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 包含两个线性层，第一个线性层的输入维度是d_model，输出维度是d_ff，第二个线性层的输入维度是d_ff，输出维度是d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        # 用dropout来防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 并不是所有的层都需要dropout，这里只对第一个线性层的输出进行dropout
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 和解码器后面的线性层的维度正好相反，这里将词嵌入的维度变成d_model
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 如同论文里说的，将嵌入之后的结果乘以sqrt(d_model)，这样做的目的是为了防止嵌入之后的结果过小？
        # 可以参考：https://arxiv.org/abs/1608.05859
        # 论文里说将权重乘以sqrt(d_model)，这里是将嵌入之后的结果乘以sqrt(d_model)，这两者有什么区别？
        # embedding层的权重矩阵是指什么？指的是网络结构里面的参数吗？如果是的话，这里为什么将将嵌入之后的结果乘以sqrt(d_model)呢？
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # dropout应用于输入到编码器和解码器的input embedding和position embedding之和上
        # 基础模型时，dropout的值是0.1
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # max_len和d_model不一样，那分别是什么意思呢？
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # 增加一个维度，position的维度是(max_len, 1)
        # 计算正余弦函数中的频率，即位置编码函数中的除法项，即分母
        # 这里为什么要先取对数，再取指数？理论上等于没有操作，但这么做的目的是什么？
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # position的维度是(max_len, 1)，div_term的维度是(d_model / 2, )，两者相乘之后的维度是(max_len, d_model / 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加一个维度，pe的维度是(1, max_len, d_model)
        # 第一个维度是batch_size，但这里位置编码的第一个维度就是1，目的就是为了和输入的x的第一个维度对应上
        pe = pe.unsqueeze(0)
        # 将pe注册为buffer，buffer里面的变量不会作为模型参数进行更新，是固定不变的，但是会被转移到GPU上
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 当这两个张量相加时，pe的第一个维度会被广播到与x的第一个维度相同的长度，以便进行相加操作
        # x是input或者output embedding，维度是(batch_size, seq_length, d_model)
        # 在PyTorch中，每个张量都有一个requires_grad属性，用于指示是否需要计算梯度，这里将其设置为False，表示不需要计算梯度;
        # 这个方法是一个in-place操作，它会直接修改原始张量，而不是返回一个新的张量,所以这里直接设置pe的requires_grad属性为False
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
        src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    # 这段代码定义了一个函数make_model，用于构建一个Transformer模型。
    # 函数的参数包括源语言词汇表src_vocab、目标语言词汇表tgt_vocab、
    # Transformer模型的超参数N、d_model、d_ff、h和dropout。
    # 函数内部首先使用copy.deepcopy函数复制了多头注意力机制、前馈神经网络和位置编码器等模块，
    # 然后使用这些模块构建了一个Encoder-Decoder模型。Encoder-Decoder模型由一个编码器和一个解码器组成，
    # 其中编码器由多个编码器层组成，每个编码器层包括一个多头注意力机制和一个前馈神经网络；
    # 解码器也由多个解码器层组成，每个解码器层包括两个多头注意力机制和一个前馈神经网络。
    # 在编码器和解码器的输入端，分别嵌入了源语言和目标语言的词嵌入向量，并加上了位置编码向量。
    # 在解码器的输出端，使用一个生成器将解码器的输出转换为目标语言的词汇表中的单词。
    # 最后，使用Xavier初始化方法对模型的参数进行初始化。"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # 会调用EncoderDecoder类的__init__方法(构造函数)，这样就会将每个部分按照构造函数的名称进行命名
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    # Xavier初始化方法是一种常用的神经网络参数初始化方法，优点是可以加速神经网络的收敛速度，提高模型的泛化能力。
    for p in model.parameters():
        if p.dim() > 1:
            # 这个判断语句是为了避免对bias进行Xavier初始化。
            # 因为bias是一维的，而Xavier初始化是针对二维及以上的参数进行的，
            # 所以对bias进行Xavier初始化是没有意义的。因此，这个判断语句可以避免对bias进行无效的初始化。
            nn.init.xavier_uniform_(p)
    return model


# 网络运行测试（未训练）
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()  # 在评估模式下，模型不会进行反向传播，也不会更新权重，而是只会进行前向传播，以便计算模型的输出。
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # 维度是(1, 10)
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    # 这行代码的作用是创建一个形状为(1, 1)的张量ys，其中所有元素都为0，
    # 并且数据类型与src张量相同。
    # 在这里，type_as()方法用于指定新张量的数据类型与src张量相同。
    # 这个张量在后续的代码中用于存储模型的输出
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # 第一次循环时，i=0，out的维度是(1, 1, 512)，其中1表示batch_size，1表示序列长度，512表示词嵌入向量的维度，也就是特征数。
        # 则out[:, -1]表示取out张量的所有行，但只取每行的最后一个元素。
        # 因为out张量的维度是(1, 1, 512)，即只有1个(1,512)，所以out[:, -1]的维度是(1, 512)。
        # 第三次循环时，i=2，out的维度是(1, 3, 512)
        # 之所以只需要取out的最后一个词的特征,是因为在计算注意力的时候（在mask之后，得到概率），p_attn与values相乘时，每个词都已经将之前的词的信息都包含在内了
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


# 主函数入口
if __name__ == "__main__":
    run_tests()
