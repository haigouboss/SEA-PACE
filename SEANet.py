import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Asymmetric_Feature_Fusion(nn.Module):
    def __init__(self, channels, r=4, act=None):
        super(Asymmetric_Feature_Fusion, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            act,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            act,
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class ContextModeling(nn.Module):

    def __init__(self, n_feat, bias=True, act=None):
        super(ContextModeling, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

class Conv_Block(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=False, groups=1, act=None):
        super(Conv_Block, self).__init__()
        
        # act = nn.LeakyReLU(0.1, False)
        padding = kernel_size // 2 

        self.body = nn.Sequential(
            
            nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=padding, bias=bias, groups=groups),
            act, 
            nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=padding, bias=bias, groups=groups)
        )

        self.act = act
        
        self.gcnet = ContextModeling(n_feat, bias=bias, act=act)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res
    
class Down(nn.Module):
    def __init__(self, in_channels, channel_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*channel_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, channel_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, channel_factor))
            in_channels = int(in_channels * channel_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, channel_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//channel_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, channel_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, channel_factor))
            in_channels = int(in_channels // channel_factor)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class sea_net(nn.Module):
    def __init__(self, n_feat = 32, channel_factor = 2, bias = True, groups = 1):
        super(sea_net, self).__init__()

        self.n_feat = n_feat
        self.act = nn.LeakyReLU(0.1, False)
        self.conv_in = nn.Conv2d(3, n_feat, kernel_size=3, padding=1, bias=bias)

        self.op_full = nn.Sequential(
            Conv_Block(int(n_feat), kernel_size=7, bias=bias, groups=groups, act=self.act),
            Conv_Block(int(n_feat), kernel_size=7, bias=bias, groups=groups, act=self.act)
            )
        self.op_half = nn.Sequential(
            Conv_Block(int(n_feat * 2), kernel_size=5, bias=bias, groups=groups, act=self.act),
            Conv_Block(int(n_feat * 2), kernel_size=5, bias=bias, groups=groups, act=self.act) 
            )
        self.op_quarter = nn.Sequential(
            Conv_Block(int(n_feat * 4), kernel_size=3, bias=bias, groups=groups, act=self.act),
            Conv_Block(int(n_feat * 4), kernel_size=3, bias=bias, groups=groups, act=self.act) 
            )
        
        self.down2 = DownSample(int(n_feat), 2, channel_factor)
        self.down4 = nn.Sequential(
            DownSample(int(n_feat), 2, channel_factor), 
            DownSample(int(n_feat * 2), 2, channel_factor)
        )

        self.up0_h2f = UpSample(int((channel_factor ** 1) * n_feat), 2, channel_factor)
        self.up1_h2f = UpSample(int((channel_factor ** 1) * n_feat), 2, channel_factor)
        self.up0_q2h = UpSample(int((channel_factor ** 2) * n_feat), 2, channel_factor)
        self.up1_q2h = UpSample(int((channel_factor ** 2) * n_feat), 2, channel_factor)

        self.conv_out_0 = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        self.AFF_0 = Asymmetric_Feature_Fusion(int(n_feat),act=self.act)
        self.AFF_1 = Asymmetric_Feature_Fusion(int(n_feat * 2),act=self.act)
        
        self.conv_out_1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_out_final = nn.Conv2d(n_feat, 3, kernel_size=3, padding=1, bias=bias)

    def MSFFE(self, x):
        x_full = self.conv_in(x)
        og = x_full
        x_half = self.down2(x_full)
        x_quarter = self.down4(x_full)

        x_full = self.op_full(x_full)
        x_half = self.op_half(x_half)
        x_quarter = self.op_quarter(x_quarter)

        x_half = self.AFF_1(x_half, self.up0_q2h(x_quarter))
        x_full = self.AFF_0(x_full, self.up0_h2f(x_half))

        feature = F.interpolate(x_full, scale_factor=0.125)

        return x_full, x_half, x_quarter, og, feature

    def FRD(self, x, x_full, x_half, x_quarter):
        x_full = self.op_full(x_full)
        x_half = self.op_half(x_half)
        x_quarter = self.op_quarter(x_quarter)

        x_half = self.AFF_1(x_half, self.up1_q2h(x_quarter))
        x_full = self.AFF_0(x_full, self.up1_h2f(x_half))

        out = self.conv_out_0(x_full)
        # out = self.conv_out(x_full)
        out = out + x

        out = self.conv_out_1(out)
        out = self.conv_out_final(out)

        return out

    def forward(self, x):
        
        x_full, x_half, x_quarter, og, feature = self.MSFFE(x)

        out = self.FRD(og, x_full, x_half, x_quarter)

        return out, feature
