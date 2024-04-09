import torch
import pandas as pd
import altair as alt


# alt.renderers.enable('altair_viewer')

def subsequent_mask(size):
    '''Mask out subsequent positions.遮挡后续位置，翻译为掩码矩阵。实现原理如下：
    构建一个维度是(size, size)的矩阵，size就是目标序列的长度，每次在预测序列的某个词（即一行中的某个元素）的时候，
    看该元素前一个元素所在列中0从哪里开始（实际上subsequent_mask每一列自主对角线开始到最后一个元素都被设置了0，
    这样在预测当前词的时候就不被允许去考虑当前词之后的词）'''
    attn_shape = (1, size, size)
    # 这里应该是生成一个和目标序列一样维度的Tensor：torch.ones(attn_shape)；
    # 然后将这个Tensor的主对角线以上的元素保留，其余元素置为0，这样就可以起到“在预测当前位置的元素时不去考虑后续的位置”
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # subsequent_mask是经过处理之后的只包含0和1的Tensor，将其与0比较，返回一个只包含true和false的Tensor
    return subsequent_mask == 0


print(subsequent_mask(4))
a = [
    pd.DataFrame(
        {
            "Subsequent Mask": subsequent_mask(5)[0][x, y].flatten(),
            "Window": y,
            "Masking": x,
        }
    )
    for y in range(5)
    for x in range(5)
]
for i in range(16):
    print(a[i])
    print("----------------------------------------------------")
print('----------------------------------------------------')
b = pd.concat(a)
print(b)

RUN_EXAMPLES = True


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(5)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=450, width=450)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
        .show()

    )


show_example(example_mask)
