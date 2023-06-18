
import torch
from torch import nn


class UNet(nn.Module):

    def __init__(self, in_channel, out_channels, width) -> None:
        #? Why?
        super(UNet, self).__init__()

        widths = [width * 2 ** i for i in range(4)]

        self.encoder1 = self.encoder(in_channel, widths[0] // 2, widths[0])
        self.encoder2 = self.encoder(widths[0], widths[1] // 2, widths[1])
        self.encoder3 = self.encoder(widths[1], widths[2] // 2, widths[2])
        self.encoder4 = self.encoder(widths[2], widths[3] // 2, widths[3])

        self.bottleneck1 = self.bottleneck(widths[3], widths[3])

        self.bottleneck2 = self.bottleneck(widths[3]*2, widths[2])

        self.upsample1 = self.upsample(widths[2])
        self.decoder1 = self.decoder(widths[2] * 2, widths[1] * 2, widths[1])

        self.upsample2 = self.upsample(widths[1])
        self.decoder2 = self.decoder(widths[1] * 2, widths[0] * 2, widths[0])

        self.upsample3 = self.upsample(widths[0])
        self.decoder3 = self.decoder(widths[0] * 2, in_channel * 2, in_channel)

        self.output = nn.Conv2d(in_channel, out_channels, kernel_size=(1, 1))

        self.weights_initialization()

    def convolve(self, in_channels, mid_channel, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, (3, 3), padding="same"),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channels, (3, 3), padding="same"),
            nn.LeakyReLU(inplace=True),
        )

    def encoder(self, in_channels, mid_channel, out_channels):
        return nn.Sequential(
            self.convolve(in_channels, mid_channel, out_channels),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5)
        )

    def bottleneck(self, in_channel, out_channel):
        return nn.Sequential(
            self.convolve(in_channel, in_channel, out_channel),
            nn.Dropout(0.5)
        )

    def decoder(self, in_channels, mid_channel, out_channels):
        return nn.Sequential(
            self.convolve(in_channels, mid_channel, out_channels),
            nn.Dropout(0.5)
        )

    def upsample(self, in_channels):
        def return_conv(input_size, kernel_size=2):
            dilation = input_size//kernel_size
            print(dilation)
            return nn.ConvTranspose2d(in_channels, in_channels*2, kernel_size=(kernel_size, kernel_size), dilation=dilation)
        return return_conv

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b1 = self.bottleneck1(e4)
        b2 = self.bottleneck2(torch.cat([e4, b1], dim=1))

        up1 = self.upsample1(e3.shape[-1])
        up1 = up1(b2)
        print(up1.shape, e3.shape)
        d1 = self.decoder1(torch.cat([up1, e3], dim=1))

        up2 = self.upsample1(e2.shape[-1])(d1)
        print(up2.shape, e2.shape)
        d2 = self.decoder2(torch.cat([up2, e2], dim=1))

        up3 = self.upsample1(e2.shape[-1])(d2)
        d3 = self.decoder3(torch.cat([up3, e1], dim=1))

        return self.output(self.upsample(d3))
