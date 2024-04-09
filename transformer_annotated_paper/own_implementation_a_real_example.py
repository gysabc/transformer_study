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
from torchviz import make_dot


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
        p_attn = dropout(
            p_attn)  # p_attn的维度没有变化(nbatches, self.h, -1, -1),即(nbatches, self.h, -1_seq_length, -1_seq_length_prob)
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
    # model.cuda()

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


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        # pad表示填充符的索引
        # 在机器翻译任务中，源语言句子和目标语言句子的长度是不一定相等的。为了方便模型的训练和推理，
        # 我们通常会将源语言句子和目标语言句子都填充到相同的长度，不足的部分用特殊的填充符（如0）进行填充；
        # 因此，在进行模型训练和推理时，我们需要遮盖掉源语言句子中的填充符，以避免填充符对模型的影响(所谓的影响，可能就是避免考虑上下文信息的时候受到填充符号的影响吧？)。
        self.src = src
        # src_mask是一个ByteTensor类型的张量，其中元素为0的位置表示需要被遮盖的位置，元素为1的位置表示不需要被遮盖的位置。
        # (src != pad)的作用就是将源语言句子中的填充符对应的位置标记为False，这样在计算P_attn*values时，对应的p_attn的值就是0，就不会受到填充符的影响。
        self.src_mask = (src != pad).unsqueeze(-2)
        # 在机器翻译任务中，通常会在目标语言句子的开头加上一个特殊的开始符（如<s>），在结尾加上一个特殊的结束符（如</s>），以表示这是一个新的句子。
        # 为什么tgt保留了开始符，tgt_y保留了结束符？
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # 去掉目标语言句子中的结束符，用于计算目标语言句子中每个单词的预测概率，即用于输入到模型中去计算模型的输出
            self.tgt_y = tgt[:, 1:]  # 去掉目标语言句子中的开始符，用于计算目标语言句子中每个单词的真实概率，即模型的标签？
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()  # 计算非填充符的个数

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total of examples used
    tokens: int = 0  # total of tokens processed


def run_epoch(
        data_iter,
        model,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
):
    '''
    
    :param data_iter:是一个数据迭代器，用于迭代训练数据集中的每个 batch。
    :param model:
    :param loss_compute:
    :param optimizer:
    :param scheduler:
    :param mode:
    :param accum_iter:是指梯度累积的步数，即在更新模型参数之前，累积多少个batch的梯度。
                      在训练过程中，如果batch size较小，可能会导致梯度下降的不稳定性，
                      而梯度累积可以在一定程度上缓解这个问题。
    :param train_state:每次都会传入一个TrainState对象，用于记录训练过程中的一些状态，如当前训练的步数、样本数、token数等。所以每一个epoch都会创建一个新的TrainState对象。
    :return:
    '''
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0  # 当前epoch中所有Batch数据的目标序列的累计有效token数
    total_loss = 0  # 当前epoch中所有Batch数据的累计损失，损失未平均，直接相加
    tokens = 0  # 用于计算单位时间处理的token数
    n_accum = 0  # 记录参数更新的次数
    for i, batch in enumerate(data_iter):
        # forward函数里面已经定义了先进行编码，再进行解码的过程
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        # 为什么要记录两次tokens数？即train_state.tokens和total_tokens，
        # 非训练模式下只有total_tokens有用，训练模式下两个都有用，两者的数值是一样的
        if mode == "train" or mode == "train+log":
            # make_dot(out,params=dict(model.named_parameters()))
            loss_node.backward()  # 通过计算图计算梯度
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()  # 实现参数的更新
                optimizer.zero_grad(set_to_none=True)  # 清空优化器中的梯度信息，set_to_none=True表示将梯度张量的值设置为None，以释放内存
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # if i % 40 == 1 and (mode == "train" or mode == "train+log"):
        if (i + 1) % 40 == 0 and (mode == "train" or mode == "train+log"):
            # 每40个batch输出一次日志信息，以便在训练过程中观察模型的训练情况
            lr = optimizer.param_groups[0]["lr"]  # 获取当前学习率
            elapsed = time.time() - start  # 计算上次输出日志到当前所经历的时间
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


'''
以下是训练过程：
We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. 
Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens. 
For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences 
and split tokens into a 32000 word-piece vocabulary.

Sentence pairs were batched together by approximate sequence length. 
Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.
'''


# adam优化器：先在最初的预热steps中线性地增加学习率，并在此后按步长的反平方根比例递减。
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    我们必须将LambdaLR函数的步长默认为1。以避免零点上升到负数。
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# 虚拟优化器
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]  # 保持与PyTorch中的优化器类的接口一致
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


# 虚拟学习率调度器
class DummyScheduler:
    def step(self):
        None


# 标签平滑：是机器学习领域的一种正则化方法，通常用于分类问题，目的是防止模型在训练时过于自信地预测标签，改善泛化能力差的问题。
'''
关于标签平滑，还需要进一步学习？代码看一下regularization_test.py

'''


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx  # 表示填充符的索引
        self.confidence = 1.0 - smoothing  # 超参数，表示置信度，用于控制预测结果的可信度
        self.smoothing = smoothing  # 是一个超参数，表示平滑因子，用于控制平滑的程度
        self.size = size  # 表示目标词典的大小
        self.true_dist = None  # 用于保存真实的分布

    def forward(self, x, target):
        '''

        :param x: 模型的预测结果
        :param target: 真实的标签
        :return:
        '''
        assert x.size(1) == self.size
        # 创建一个大小为x的张量，将其填充为平滑/（size-2）
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 平滑处理。这里就简单的将真实分布的所有值都设置为平滑因子/（size-2）
        # 将正确单词的置信度分配给目标张量
        # scatter_函数会将self.confidence按照target.data.unsqueeze(1)中的索引写入到true_dist张量中，从而得到一个新的概率分布？
        # 目前的理解：真实类别分别是2,1,0,3,3，因此将confidence赋值给true_dist中对应的位置
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 在计算损失函数时，需要将预测为填充符的概率设置为0
        true_dist[:, self.padding_idx] = 0
        # target.data == self.padding_idx：创建一个大小与target张量相同的张量，其中每个元素表示对应位置上的元素是否等于self.padding_idx,如果相等则为1，否则为0
        # torch.nonzero:找到一个张量中所有非零元素的索引。给定一个大小为n的张量input，
        # torch.nonzero(input)会返回一个大小为(m, k)的张量output，
        # 其中m表示input中非零元素的数量，k表示input的维度数。output中的每一行都表示一个非零元素在input中的索引。详见pytorch学习种的记录

        # 在深度学习中，torch.nonzero常用于处理稀疏张量。例如，在计算损失函数时，
        # 可以使用torch.nonzero找到标签张量中所有非零元素的索引，然后根据这些索引来计算损失函数。
        # target.data == self.padding_idx：tensor([False, False,  True, False, False])
        mask = torch.nonzero(target.data == self.padding_idx)  # 因此最终的mask里面存放的是填充符的索引
        if mask.dim() > 0:
            # 条件成立说明mask不是一个空张量，即target中包含填充符
            # squeeze()：从mask张量中删除大小为1的任何维度，这是为了确保index_fill_方法能够正确工作

            # 第一个参数0表示要在true_dist张量的第一维度上进行填充
            # 0.0表示在true_dist张量上将所有填充符的概率设置为0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  # 把真实的标签里面是填充符的那个位置，在预测的那一行概率全部设置为0
        self.true_dist = true_dist
        # detach()方法的作用是将张量从计算图中分离出来，用于计算损失函数，而不会影响计算图的反向传播
        # x是最初的模型的预测的概率分布，true_dist是根据target将模型预测的概率分布进行平滑处理、以及按照真实的标签分配了confidence，
        # 所以相当于是真实的概率分布，而且单单一个target，它只是标签，并不是概率分布，所以用经过处理的模型预测的分布来代替真实的概率分布
        return self.criterion(x, true_dist.clone().detach())


# Here we can see an example of how the mass is distributed to the words based on confidence.什么意思？


# 计算损失
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator  # 用于在计算损失之前将模型前向计算之后输出`out`的维度从`d_model`变成`vocab_size`
        self.criterion = criterion  # 标签平滑

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        return sloss.data * norm, sloss


def load_tokenizers():
    try:
        # 加载德语分词器
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        # 加载英语分词器
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    # 将输入的文本text进行分词，返回分词后的结果
    # 使用tokenizer对象的tokenizer方法对文本进行分词，然后将每个分词结果的text属性提取出来，组成一个列表返回
    return [tok.text for tok in tokenizer.tokenizer(text)]


# 生成器函数
def yield_tokens(data_iter, tokenizer, index):
    '''
    
    :param data_iter:迭代器
    :param tokenizer:分词器对象
    :param index:一个整数，表示对每个元组中第几个元素进行分词
    :return:
    '''
    for from_to_tuple in data_iter:
        # 每次从元组中提取指定位置的文本，使用tokenizer对象对文本进行分词，然后将分词结果作为生成器的一个元素返回
        yield tokenizer(from_to_tuple[index])


# ------------------------这里构建一下数据集----------------------------
def build_dataset():
    my_file_en = "dataset/europarl-v7.de-en.en"
    my_file_de = "dataset/europarl-v7.de-en.de"

    # May need to adjust these parameters for new translation tasks.
    size_train = 100000
    size_val = 4000
    size_test = 4000

    train = []
    val = []
    test = []

    de_list = []
    en_list = []
    count = 0
    # 读取德语文本
    with open(my_file_de, encoding="utf8") as fp:
        for line in fp:
            de_list.append(line)
            if count >= (size_train + size_val + size_test - 1):
                # 因为是先判断再自增，所以这里要减1
                break
            count += 1

    count = 0
    # 读取英语文本
    with open(my_file_en, encoding="utf8") as fp:
        for line in fp:
            en_list.append(line)
            if count > (size_train + size_val + size_test - 1):
                break
            count += 1

    # 两个列表中的元素一一对应打包成元组，然后将这些元组组成一个列表dataset
    dataset = list(zip(de_list, en_list))
    # 切片，获取训练集、验证集和测试集
    train = dataset[:size_train]
    val = dataset[size_train:size_train + size_val]
    test = dataset[size_train + size_val:size_train + size_val + size_test]
    print(len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------
def build_vocabulary(spacy_de, spacy_en):
    '''

    :param spacy_de:德语分词器对象
    :param spacy_en:英语分词器对象
    :return:
    '''

    def tokenize_de(text):
        # 对德语文本进行分词，传入tokenize函数的参数是德语分词器对象和待分词的文本
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        # 对英语文本进行分词，传入tokenize函数的参数是英语分词器对象和待分词的文本
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    # 使用了torchtext.datasets.Multi30k函数，该函数会自动下载数据集并返回训练集、验证集和测试集
    # train, val, test = datasets.Multi30k(language_pair=("de", "en"), root='data/')
    train, val, test = build_dataset()
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    # train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    # 将词汇表的默认索引设置为`<unk>`
    # 可以通过vocab_src["<unk>"]来访问词汇表中某一个词的索引
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


# 加载词汇表
def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")  # vocab.pt是一个PyTorch中的文件，通常用于保存文本数据的词汇表
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


# Batch处理，希望数据分布的更加均匀，以减少填充
# 因此这里需要对默认的torchtext batching函数进行修改
def collate_batch(
        batch,
        src_pipeline,
        tgt_pipeline,
        src_vocab,
        tgt_vocab,
        device,
        max_padding=128,
        pad_id=2,
):
    '''

    :param batch:
    :param src_pipeline: 德语分词器对象
    :param tgt_pipeline: 英语分词器对象
    :param src_vocab: 源语言序列(德语)的词汇表
    :param tgt_vocab: 目标语言序列(英语)的词汇表
    :param device:
    :param max_padding:
    :param pad_id: 填充符在词汇表中的索引(传进来的是blank的索引，src和tgt中填充符的索引是一样的)
    :return:
    '''
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=12000,
        max_padding=128,
        is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        # 对当前batch中的数据进行处理，以元组的形式返回处理后的数据
        # 这里传入的参数是batch，batch是一个列表，列表中的每一个元素是一个元组，元组中的第一个元素是德语句子，第二个元素是英语句子
        # 因此不同于之前那个Batch类，要区分开来
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    # 这里也要做相应的修改
    # train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=("de", "en"))
    train, val, test = build_dataset()
    train_iter, valid_iter, test_iter = train, val, test

    # to_map_style_dataset将train_iter转换为map-style的数据集，这个数据集可以被DataLoader使用，
    # 以便使用DistributedSampler进行分布式训练
    train_iter_map = to_map_style_dataset(train_iter)  # DistributedSampler needs a dataset len()
    # 构建训练数据集的采样器
    # 如果is_distributed为True，则创建一个分布式采样器DistributedSampler，
    # 并将训练数据集train_iter_map作为参数传递给它；否则，将采样器设置为None
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    # 在训练时，DataLoader会自动从数据集中加载数据，并将其组成一个batch返回给模型进行训练
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # 如果是分布式训练，会有分布式采样器，此时不需要shuffle
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        # 由于每个样本的大小可能不同，因此需要将它们进行填充或截断，以便组成一个固定大小的batch
        # 因此使用collate_fn对输入的样本列表进行处理，返回一个batch的数据
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def train_worker(
        gpu,
        ngpus_per_node,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distributed=False,
):
    # 标准输出通常是缓冲的，这意味着输出的内容不会立即显示在屏幕上，而是在缓冲区中等待。
    # 当缓冲区被填满或者程序结束时，缓冲区的内容才会被输出。如果在程序运行时需要立即将输出显示在屏幕上，可以使用flush参数
    # 详见：https://blog.csdn.net/Granthoo/article/details/82880562
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    # 设置当前进程使用的GPU设备
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)  # 还需要将模型移动到指定的GPU设备上进行计算
    module = model
    is_main_process = True
    if is_distributed:
        # dist就是torch.distributed
        # 在分布式训练中，每个进程都有自己的GPU，而且每个进程都需要加载模型和数据，进行训练。
        # 为了协调不同进程之间的训练，需要使用分布式工具包来进行进程间的通信和同步
        # 这里调用了dist.init_process_group函数，这个函数会初始化进程组，以便进程之间可以进行通信
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            # 如果output文件夹不存在，则创建
            if not os.path.exists("output"):
                os.makedirs("output")
            file_path = "output/%s%.2d.pt" % (config["file_prefix"], epoch)  # 设置保存模型的文件名
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        # 如果output文件夹不存在，则创建
        if not os.path.exists("output"):
            os.makedirs("output")
        file_path = "output/%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, config
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False
        )


# def load_config():
#     config = {
#         "batch_size": 32,
#         "distributed": False,
#         "num_epochs": 8,
#         "accum_iter": 10,
#         "base_lr": 1.0,
#         "max_padding": 72,
#         "warmup": 3000,
#         "file_prefix": "multi30k_model_",
#     }
#     return config


def load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en):
    # 可以作为主函数的入口
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "output/multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    # 加载的是模型参数等信息，而不是模型本身，需要重新构建模型结构
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    return model


# 贪婪解码
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        # 因为ys中已经包含了一个符号，所以这里的输出是从第二个符号开始的，总共需要解码9个元素
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)  # torch.max返回的索引下标从0开始
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def check_outputs(
        valid_dataloader,
        model,
        vocab_src,
        vocab_tgt,
        n_examples=15,
        pad_idx=2,
        eos_string="</s>",
):
    results = [()] * n_examples  # 定义了一个空的列表 results，长度为 n_examples，其中每个元素都是一个空元组 ()。这个列表将用于存储模型的输出结果
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))  # iter 函数用于将 valid_dataloader 转换为一个迭代器，next 函数用于从迭代器中取出一个元素，这里就是一个 batch 的数据
        rb = Batch(b[0], b[1], pad_idx)
        # greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        # print("stop")
        model_txt=[vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
        model_txt=" ".join(model_txt)
        model_txt=model_txt.split(eos_string, 1)[0]
        model_txt=model_txt+eos_string
        # model_txt = (
        #         " ".join(
        #             [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
        #         ).split(eos_string, 1)[0] # split的第二个参数表示最多分割一次
        #         + eos_string
        # )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, model, n_examples=5):
    # global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    # print("Loading Trained Model ...")
    #
    # model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    # model.load_state_dict(
    #     torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    # )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return example_data


# Python主函数入口
if __name__ == "__main__":
    # build_dataset()
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    # config = load_config()
    # train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    # 加载训练好的模型，如果不存在，则训练模型
    model = load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en)
    example_data=run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, model, n_examples=5)
