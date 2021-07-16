from torch import nn
class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace==True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y).view(b, c, 1, 1)
        return x * y

from torch import nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentioin(nn.Module):
    def __init__(self, kernek_size=7):
        super(SpatialAttentioin, self).__init__()
        assert kernek_size() in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernek_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernek_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmode()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

        