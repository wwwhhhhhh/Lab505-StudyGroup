import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings(action='ignore')


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int) )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv1(x)))

class INFAttConv(nn.Module):
    def __init__(self, inf_c, x_c_in):
        super(INFAttConv, self).__init__()
        self.Att = Attention_block(F_g=inf_c, F_l = x_c_in, F_int=int(x_c_in/2))
        self.conva = ConvBnRelu2d(x_c_in,x_c_in)
    def forward(self, inf, x):
        att = self.Att(inf,x)
        x = x + att
        x = self.conva(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class INFAttNet(nn.Module):
    def __init__(self,n_class, deep_supervision=False):
        super(INFAttNet, self).__init__()
        img_n_channels = 8
        inf_n_channels = 3
        self.deep_supervision = deep_supervision
        self.inc = inconv(img_n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.inf_inc = inconv(inf_n_channels, 32)
        self.inf_down1 = down(32, 64)
        self.inf_down2 = down(64, 128)
        self.inf_down3 = down(128, 256)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64,n_class)

        self.dsoutc4 = outconv(256,n_class)
        self.dsoutc3 = outconv(128,n_class)
        self.dsoutc2 = outconv(64,n_class)
        self.dsoutc1 = outconv(64,n_class)

        inf_ch = [64,128,256,512]
        filters=[32, 64, 128, 256,512]

        self.conv1_inf   = ConvBnRelu2d(3, inf_ch[0],3)
        self.conv_inf_1 = ConvBnRelu2d(inf_ch[0], inf_ch[1],3)
        self.conv_inf_2 = ConvBnRelu2d(inf_ch[1], inf_ch[2],3)
        self.conv_inf_3 = ConvBnRelu2d(inf_ch[2],inf_ch[3],3)

        self.up_inf_att1 = INFAttConv(inf_ch[0], filters[1])
        self.up_inf_att2 = INFAttConv(inf_ch[0], filters[1])
        self.up_inf_att3 = INFAttConv(inf_ch[1], filters[2])
        self.up_inf_att4 = INFAttConv(inf_ch[2], filters[3])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_inf = x[:,3:6]
        x_inf_1 = self.inf_inc(x_inf)
        x_inf_2 = self.inf_down1(x_inf_1)
        x_inf_3 = self.inf_down2(x_inf_2)
        x_inf_4 = self.inf_down3(x_inf_3)
        
        x44 = self.up1(x5, x4)
        x33 = self.up_inf_att4(x_inf_4,x44)
        x33 = self.up2(x44, x3)
        x33 = self.up_inf_att3(x_inf_3,x33)
        x22 = self.up3(x33, x2)
        x22 = self.up_inf_att2(x_inf_2,x22)
        x11 = self.up4(x22, x1)
        x0 = self.outc(x11)
        return x0