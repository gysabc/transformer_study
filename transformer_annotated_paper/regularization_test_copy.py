import torch
import torch.nn as nn


# We implement label smoothing using the KL div loss.
# Instead of using a one-hot target distribution, 
# we create a distribution that has confidence of the correct word 
# and the rest of the smoothing mass distributed throughout the vocabulary.

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx  # 表示填充符的索引
        self.confidence = 1.0 - smoothing  # 超参数，表示置信度，用于控制预测结果的可信度
        self.smoothing = smoothing  # 是一个超参数，表示平滑因子，用于控制平滑的程度
        self.size = size
        self.true_dist = None  # 用于保存真实的分布

    def forward(self, x, target):
        '''

        :param x: 真实的数据分布
        :param target: 模型的预测结果
        :return:
        '''
        assert x.size(1) == self.size
        # 创建一个大小为x的张量，将其填充为平滑/（size-2）
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 平滑处理。这里就简单的将真实分布的所有值都设置为平滑因子/（size-2）
        # 将正确单词的置信度分配给目标张量
        # scatter_函数会将self.confidence中的每个元素按照target.data.unsqueeze(1)中的索引写入到true_dist张量中，从而得到一个新的概率分布
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 在计算损失函数时，需要将填充符的概率设置为0
        true_dist[:, self.padding_idx] = 0
        # target.data == self.padding_idx：创建一个大小与target张量相同的张量，其中每个元素表示对应位置上的元素是否等于self.padding_idx,如果相等则为1，否则为0
        # torch.nonzero:找到一个张量中所有非零元素的索引。给定一个大小为n的张量input，
        # torch.nonzero(input)会返回一个大小为(m, k)的张量output，
        # 其中m表示input中非零元素的数量，k表示input的维度数。output中的每一行都表示一个非零元素在input中的索引。详见pytorch学习种的记录

        # 在深度学习中，torch.nonzero常用于处理稀疏张量。例如，在计算损失函数时，
        # 可以使用torch.nonzero找到标签张量中所有非零元素的索引，然后根据这些索引来计算损失函数。
        mask = torch.nonzero(target.data == self.padding_idx) # 因此最终的mask里面存放的是填充符的索引
        if mask.dim() > 0:
            # 条件成立说明mask不是一个空张量，即target中包含填充符
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
