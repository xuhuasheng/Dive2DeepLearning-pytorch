import torch
from torch import nn
from d2l import torch as d2l

# ######################################
# dropout丢弃法
# ######################################
def dropout(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 0:
        return X
    if dropout == 1:
        return torch.zeros_like(X)
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()                                              
    return mask * X / (1.0 - dropout)
# ######################################

X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

# tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
# tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
# tensor([[ 0.,  0.,  4.,  6.,  0., 10.,  0.,  0.],
#         [ 0., 18.,  0., 22.,  0., 26.,  0., 30.]])
# tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, dropout1, dropout2, is_trian=True):
        super.__init__(Net, self)
        self.num_inputs = num_inputs
        self.fc1 = nn.Linear(num_inputs, num_hiddens1)
        self.fc2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.fc3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.fc1(X.reshape((-1, self.num_inputs))))
        if is_train:
            H1 = dropout(H1, dropout1)
        H2 = self.relu(self.fc2(H1))
        if is_train:
            H2 = dropout(H2, dropout2)
        out = self.relu(self.fc2(H2))
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# 精简实现
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Dropout(dropout1), nn.Linear(256, 256), nn.ReLU(),
                    nn.Dropout(dropout2), nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)