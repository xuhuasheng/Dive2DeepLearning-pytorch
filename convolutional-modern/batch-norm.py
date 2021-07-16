import torch
from torch import nn
from d2l import torch as d2l


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        """
        num_features: nums of outputs for fc
                      nums of output channels for convolution
        num_dims: 2 for fc,
                  4 for conv
        """
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参数赋形初始化
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
        # 推理模式
        if not torch.is_grad_enabled():
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        # 训练模式
        else:
            assert len(X.shape) in (2, 4)
            # 当X为全连接层上的小批量，对小批量N的一维样本求均值和方差
            if len(X.shape) == 2:
                mean = X.mean(dim=0, keepdim=True) # shape = (1, X.shape[1])
                var = ((X - mean)**2).mean(dim=0, keepdim=True)
            # 当X为卷积层上的小批量，对小批量N的每一个通道求HxW的均值和方差
            else:
                mean = X.mean(dim=(0,2,3), keepdim=True) # shape = (1, C, 1, 1)
                var = ((X - mean)**2).mean(dim=(0,2,3), keepdim=True)
            # 归一化
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # 持续更新所有样本的均值和方差，为推理阶段的BN准备
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        # 可训练的伸缩参数和偏移参数，输出最终的BN结果
        Y = gamma * X_hat + beta
        return Y, moving_mean, moving_var

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, 
                                                          self.moving_mean, self.moving_var, 
                                                          eps=1e-5, momentum=0.9)
        return Y


# 应用`BatchNorm`于LeNet模型
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), 
                    BatchNorm(num_features=6, num_dims=4),
                    nn.Sigmoid(), 
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), 
                    BatchNorm(num_features=16, num_dims=4),
                    nn.Sigmoid(), 
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten(), 
                    nn.Linear(16 * 4 * 4, 120),
                    BatchNorm(num_features=120, num_dims=2), 
                    nn.Sigmoid(),
                    nn.Linear(120, 84), 
                    BatchNorm(num_features=84, num_dims=2),
                    nn.Sigmoid(), 
                    nn.Linear(84, 10))

# 在Fashion-MNIST数据集上训练网络
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
