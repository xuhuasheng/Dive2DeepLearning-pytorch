import torch
from torch import nn
from torch.nn import functional as F 
from d2l import torch as d2l

# 残差单元
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        # 第一层3x3卷积（conv-bn-relu）
        Y = F.relu(self.bn1(self.conv1(X)))
        # 第二层3x3卷积（conv-bn）
        Y = self.bn2(self.conv2(Y))
        # 恒等映射（可能需要1x1 conv）
        if self.conv3:
            X = self.conv3(X)
        Y += X
        # relu激活
        return F.relu(Y)

# 残差块
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 第一个残差单元：通道翻倍，尺度减半（恒等映射线路需要1x1卷积升维，保持通道一致）
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            # 第二个残差单元：保持通道和尺度不变
            blk.append(Residual(num_channels, num_channels, use_1x1conv=False, strides=1))
    return blk

# ResNet-18: 1 conv + 16 conv + 1 fc
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), 
                   nn.BatchNorm2d(64), 
                   nn.ReLU(), 
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 4个残差块，每个有4层conv，共16层conv
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
# 全局平均值池化，全连接层
net = nn.Sequential(b1, b2, b3, b4, b5, 
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, 10))

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


