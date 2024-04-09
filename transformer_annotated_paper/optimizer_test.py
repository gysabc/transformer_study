import torch
from torch.optim.lr_scheduler import LambdaLR
import altair as alt
import pandas as pd

'''
torch.optim.Adam是一个实现了Adam算法的优化器。Adam是一种自适应学习率的优化算法，
它可以根据每个参数的梯度的一阶矩估计和二阶矩估计自适应地调整每个参数的学习率。
Adam算法的优点是可以自适应地调整学习率，同时也可以像SGD一样处理稀疏梯度。
Adam算法的缺点是需要存储每个参数的一阶矩估计和二阶矩估计，因此需要更多的内存。
'''

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


def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        # 每一步的学习率都会根据step的值进行更新
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        
        # learning_rates是一个列表，其中包含了三个列表，每个列表中包含了20000个学习率
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :], # 是一个一维数组,包含了20000个学习率
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ], # 某一个模型大小和预热步数的组合
                    "step": range(20000), # 一个一维数组，包含了20000个步数，用于画图时对应每一个学习率
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
        .show()
    )

# 从最后生成的图来看，预热阶段学习率是线性增加的，并在此后按步长的反平方根比例递减。
example_learning_schedule()
