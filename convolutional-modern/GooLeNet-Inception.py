# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %% [markdown]
# # 含并行连结的网络（GoogLeNet）
# 
# Inception块

# %%
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1：1x1卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2：1x1卷积 + 3x3卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3：1x1卷积 + 5x5卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4：3x3最大值池化 + 1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 4个线路输出 在通道维度拼接
        return torch.cat((p1, p2, p3, p4), dim=1)

# %% [markdown]
# GoogLeNet模型

# %%
# GoogLeNet 由5个block串行构成
# b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
#                    nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2,
#                                            padding=1))

# b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
#                    nn.Conv2d(64, 192, kernel_size=3, padding=1),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
#                    Inception(256, 128, (128, 192), (32, 96), 64),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
#                    Inception(512, 160, (112, 224), (24, 64), 64),
#                    Inception(512, 128, (128, 256), (24, 64), 64),
#                    Inception(512, 112, (144, 288), (32, 64), 64),
#                    Inception(528, 256, (160, 320), (32, 128), 128),
#                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
#                    Inception(832, 384, (192, 384), (48, 128), 128),
#                    nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

# net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

class GooLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))    

        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                Inception(256, 128, (128, 192), (32, 96), 64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                Inception(512, 160, (112, 224), (24, 64), 64),
                                Inception(512, 128, (128, 256), (24, 64), 64),
                                Inception(512, 112, (144, 288), (32, 64), 64),
                                Inception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                Inception(832, 384, (192, 384), (48, 128), 128),
                                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.fc(x)
        return x


# %% [markdown]
# 为了使Fashion-MNIST上的训练短小精悍，我们将输入的高和宽从224降到96

# %%
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# %% [markdown]
# 训练模型

# %%
# lr, num_epochs, batch_size = 0.1, 10, 128
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# %%
model = GooLeNet()
# loss
loss_func = F.cross_entropy
# 优化器设置
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# 数据
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False)

# 训练
for epoch in range(epochs):
	# 设置为训练模式
    model.train()
    # iterate: 每次一个batch
    for xb, yb in train_dl:
    	# 前向传播
        pred = model(xb)
        # 计算损失
        loss = loss_func(pred, yb)
		# 反向传播，计算loss关于各权重参数的偏导，更新grad
        loss.backward()
        # 优化器基于梯度下降原则，更新（学习）权重参数parameters
        opt.step()
        # 各权重参数的偏导清零 grad=>0
        opt.zero_grad()
	# 设置为评估（推理）模式，设置BN、dropout等模块
    model.eval()
    # 不更新梯度
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))