import torch
from torch import nn
from d2l import torch as d2l

# 互相关
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    s = 1 # stride
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[s*i : s*i+h, s*j : s*j+w] * K).sum()
    return Y

# 二维卷积实现
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# %% 验证
X = torch.ones((6, 8))
X[:, 2:6] = 0
# X = 
# tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.]])

# 边缘检测
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
# Y= 
# tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])

# %% 学习由`X`生成`Y`的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

# 迭代10次
for i in range(10):
    # 正向预测
    Y_hat = conv2d(X)
    # 求损失函数
    loss = (Y_hat - Y)**2
    # 梯度清零
    conv2d.zero_grad()
    # 反向传播
    loss.sum().backward()
    # 梯度下降法更新权重
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {loss.sum():.3f}')

# batch 2, loss 18.826
# batch 4, loss 6.257
# batch 6, loss 2.319
# batch 8, loss 0.909
# batch 10, loss 0.365

# 所学的卷积核的权重张量
conv2d.weight.data.reshape((1, 2))
# 学习到的权重：tensor([[ 0.9252, -1.0493]])
# 真实的权重：K = torch.tensor([[1.0, -1.0]])

# %%
# X含有n个通道，则卷积核的深度为n
# 对应通道做卷积，再在通道维度相加，得到一个feature map
def corr2d_multi_in(X, K):
    return sum(corr2d(x, k) for x, k in zip(X, K))

# test
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

# corr2d_multi_in(X, K)
# tensor([[ 56.,  72.],
#         [104., 120.]])

# %%
# K个卷积核 输出K个通道feature map张量
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)

K = torch.stack((K, K + 1, K + 2), 0)
# K.shape
# torch.Size([3, 2, 2, 2])
corr2d_multi_in_out(X, K)
# tensor([[[ 56.,  72.],
#          [104., 120.]],

#         [[ 76., 100.],
#          [148., 172.]],

#         [[ 96., 128.],
#          [192., 224.]]])

# %% 1x1 卷积 等效于全连接层，等效于矩阵相乘
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    return Y.reshape((c_o, h, w))

# X = torch.normal(0, 1, (3, 3, 3))
# K = torch.normal(0, 1, (2, 3, 1, 1))

# Y1 = corr2d_multi_in_out_1x1(X, K)
# Y2 = corr2d_multi_in_out(X, K)
# assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
# %%
