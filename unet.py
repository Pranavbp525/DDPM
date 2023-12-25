import math

import torch


import torch.nn as nn
import torch.nn.functional as F





class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):

        h = self.bn1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb

        h = self.bn2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeds = math.log(10000) / (half_dim - 1)
        embeds = torch.exp(torch.arange(half_dim, device=device) * -embeds)
        embeds = time[:, None] * embeds[None, :]
        embeds = torch.cat((embeds.sin(), embeds.cos()), dim=-1)
        return embeds


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        """"
        in_channels: input channels of the incoming image
        out_channels: output channels of the incoming image
        """
        super(UNet, self).__init__()

        # ------------------------ Encoder ------------------------#

        self.ini = self.doubleConvolution(inC=in_channels, oC=16)
        self.down1 = self.Down(inputC=16, outputC=32)
        self.down2 = self.Down(inputC=32, outputC=64)

        # ------------------------ Decoder ------------------------#

        self.time_emb2 = self.timeEmbeddings(1, 64)
        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.afterup2 = self.doubleConvolution(inC=64, oC=32)

        self.time_emb1 = self.timeEmbeddings(1, 32)
        self.up1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2)
        self.afterup1 = self.doubleConvolution(inC=32, oC=16, kS1=5, kS2=4)

        # ------------------------ OUTPUT ------------------------#

        self.out = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t=None):
        assert t is not None

        # ------------------------ Encoder ------------------------#

        x1 = self.ini(x)  # Initial Double Convolution
        x2 = self.down1(x1)  # Downsampling followed by Double Convolution
        x3 = self.down2(x2)  # Downsampling followed by Double Convolution

        # ------------------------ Decoder ------------------------#

        t2 = self.time_emb2(t)[:, :, None, None]
        y2 = self.up2(x3 + t2)  # Upsampling
        y2 = self.afterup2(torch.cat([y2, self.xLikeY(x2, y2)],
                                     axis=1))  # Crop corresponding Downsampled Feature Map, Double Convolution

        t1 = self.time_emb1(t)[:, :, None, None]
        y1 = self.up1(y2 + t1)  # Upsampling
        y1 = self.afterup1(torch.cat([y1, self.xLikeY(x1, y1)],
                                     axis=1))  # Crop corresponding Downsampled Feature Map, Double Convolution

        outY = self.out(y1)  # Output Layer (ks-1, st-1, pa-0)

        return outY

    # --------------------------------------------------------------------------------------------------- Helper Functions Within Model Class

    def timeEmbeddings(self, inC, oSize):
        """
        inC: Input Size, (for example 1 for timestep)
        oSize: Output Size, (Number of channels you would like to match while upsampling)
        """
        return nn.Sequential(nn.Linear(inC, oSize),
                             nn.ReLU(),
                             nn.Linear(oSize, oSize))

    def doubleConvolution(self, inC, oC, kS1=3, kS2=3, sT=1, pA=1):
        """
        Building Double Convolution as in original paper of Unet
        inC : inputChannels
        oC : outputChannels
        kS1 : Kernel_size of first convolution
        kS2 : Kernel_size of second convolution
        sT: stride
        pA: padding
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=inC, out_channels=oC, kernel_size=kS1, stride=sT, padding=pA),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=oC, out_channels=oC, kernel_size=kS2, stride=sT, padding=pA),
            nn.ReLU(inplace=True),
        )

    def Down(self, inputC, outputC, dsKernelSize=None):
        """
        Building Down Sampling Part of the Unet Architecture (Using MaxPool) followed by double convolution
        inputC : inputChannels
        outputC : outputChannels
        """

        return nn.Sequential(
            nn.MaxPool2d(2),
            self.doubleConvolution(inC=inputC, oC=outputC)
        )

    def xLikeY(self, source, target):
        """
        Helper function to resize the downsampled x's to concatenate with upsampled y's as in Unet Paper
        source: tensor whose shape will be considered ---------UPSAMPLED TENSOR (y)
        target: tensor whose shape will be modified to align with target ---------DOWNSAMPLED TENSOR (x)
        """
        x1 = source
        x2 = target
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x1
