import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Softmax

class ResDouble1Conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(ResDouble1Conv, self).__init__()
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

class ResDoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, mid_channels=None):
        super(ResDoubleConv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x = x +  residual
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
            # ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # self.conv = ResDoubleConv(in_channels, out_channels, in_channels // 2)
            
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            # self.conv = DoubleConv(in_channels, out_channels)
            self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PamAtt(nn.Module):
    def __init__(self, x_dim):
        super(PamAtt, self).__init__()
        self.chanel_in = x_dim

        self.query_conv = nn.Conv2d(
            in_channels=x_dim, out_channels=x_dim//16, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=x_dim, out_channels=x_dim//16, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=x_dim, out_channels=x_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        # print (x.shape)
        # print(x.shape)
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # print('x_query:',proj_query.shape)
        # print('x_key:',proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # print('attention:', attention.shape)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # print('proj_value:', proj_value.shape)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        # print('out:', out.shape)
        out = self.gamma*out + x
        # print('out:', out.shape)
        return out

class CamAtt(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CamAtt, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class ForestUNetNoRes(nn.Module):
    def __init__(self, n_class, bilinear=False):
        super(ForestUNetNoRes, self).__init__()
        self.n_channels = 1
        self.n_class = n_class
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.down5 = Down(512, 1024//factor)
        
        
        self.pam_att1 = PamAtt(1024)
        self.pam_att2 = PamAtt(512)
        self.pam_att3 = PamAtt(256)
        
        self.cam_att1 = CamAtt(512)
        self.cam_att2 = CamAtt(256)
        self.cam_att3 = CamAtt(128)
        self.cam_att4 = CamAtt(64)
        
        
        self.up1 = Up(1024, 512//factor,bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_class)

    def forward(self, x):
        # x_ori = torch.cat([x,x],dim = 1)
        # x_ori = torch.cat([x_ori,x],dim = 1)
        # x = x_ori
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.pam_att3(x4)
        x5 = self.down4(x4)
        x5 = self.pam_att2(x5)
        x6 = self.down5(x5)
        x6 = self.pam_att1(x6)
        
        x = self.up1(x6, x5)
        x = self.cam_att1(x)
        x = self.up2(x, x4)
        x = self.cam_att2(x)
        x = self.up3(x, x3)
        x = self.cam_att3(x)
        x = self.up4(x, x2)
        x = self.cam_att4(x)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     # print(classname)
    #     if classname.find('Conv') != -1:
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             torch.nn.init.constant_(m.bias.data, 0.0)

    model = ForestUNetNoRes(2).cuda()
    # model.apply(weights_init)

    x = torch.randn((3, 1, 1000, 1000)).cuda()

    # for i in range(10):
    #     y0 = model(x)
    # print(y0)
    y0 = model(x)
