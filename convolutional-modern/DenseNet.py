import torch
from torch import nn
from d2l import torch as d2l

# 卷积块：BN-ReLU-Conv
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

# 稠密层
# 每一层卷积块的输出通道连接
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        # 多个卷积块。每个卷积块的输入维度递增，输出不变
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        # 每一层卷积块的输出通道连接
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

# blk = DenseBlock(2, 3, 10)
# X = torch.randn(4, 3, 8, 8)
# Y = blk(X)
# Y.shape

# 过渡层：BN-ReLU-Conv-Pool 
# 降低模型复杂度
# 1x1卷积 通道减半
# avgpooling 尺寸减半
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1), 
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

# blk = transition_block(23, 10)
# blk(Y).shape

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32 
num_convs_in_dense_blocks = [4, 4, 4, 4]
blk = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:     
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(b1, *blks, 
                    nn.ReLU(),
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(num_channels, 10))


lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())