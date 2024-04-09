import torch
import torch.nn as nn
import altair as alt
import pandas as pd


# We implement label smoothing using the KL div loss.
# Instead of using a one-hot target distribution, 
# we create a distribution that has confidence of the correct word 
# and the rest of the smoothing mass distributed throughout the vocabulary.什么意思？

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
            true_dist.index_fill_(0, mask.squeeze(), 0.0) # 把真实的标签里面是填充符的那个位置，在预测的那一行概率全部设置为0
        self.true_dist = true_dist
        # detach()方法的作用是将张量从计算图中分离出来，用于计算损失函数，而不会影响计算图的反向传播
        # x是最初的模型的预测的概率分布，true_dist是根据target将模型预测的概率分布进行平滑处理、以及按照真实的标签分配了confidence，
        # 所以相当于是真实的概率分布，而且单单一个target，它只是标签，并不是概率分布，所以用经过处理的模型预测的分布来代替真实的概率分布
        return self.criterion(x, true_dist.clone().detach()) 


# Here we can see an example of how the mass is distributed to the words based on confidence.什么意思？

# Example of label smoothing.


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    # 预测结果，每一行应该表示一个待预测的词，一行有5个元素(每个元素对应于词表中每个词的概率值)，因为词表的大小是5
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    # predict.log()是将predict张量中的每个元素取对数，这是因为在计算KL散度时，需要将两个分布的概率值取对数，然后再计算它们的差值。
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(), # flatten()用于将张量展平为一维,原来是0维的话，还是会被展平为1维的
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
        .show()
    )


# example_label_smoothing()
# 生成的图中，第3行(即下标为2)的概率全为0是因为：在target中第三个词是填充符，因此在计算损失函数时，需要将预测为填充符的概率设置为0；
# 第0列的所有元素概率值都是0是因为：填充符在词表中的索引是0，因此对于模型的预测输出，需要将其第一个元素的概率值设置为0。

def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]]) # 随着x的增加，模型对第二个词的预测概率越来越大。
    return crit(predict.log(), torch.LongTensor([1])).data # 而真实的标签是1，即词表中第二个词。


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1) # confidence等于0.9，说明此时模型对预测的输出很自信。
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)], # 在x越来越大的时候，真实标签一直是1，confidence也一直处于较高水平0.9
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


penalization_visualization()
# Label smoothing actually starts to penalize the model if it gets very confident about a given choice.如何理解？
