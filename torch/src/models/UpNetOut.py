# UPNET
import torch
from torch import nn


class UpNet(nn.Module):

    def set_encoders(self, widths: list, i: int):
        """Add encoder attributes to the CNN

        Args:
            widths (list): List of ints to set the cannel band width
            i (int): how many decoders have been created - 1
        """
        setattr(self, f"encoder{i+1}", self.encoder(widths[0], widths[1] // 2, widths[1]))
        if len(widths[1:])>1:
            self.set_encoders(widths[1:], i+1)

    def set_decoders(self, widths: list, i: int):
        """Add decoder attributes to the CNN

        Args:
            widths (list): List of ints to set the cannel band width
            i (int): how many decoders have been created - 1
        """
        setattr(self, f"decoder{i}", self.decoder(widths[-1]*2, widths[-2] * 2, widths[-2]))
        if len(widths[:-1])>1:
            self.set_decoders(widths[:-1], i+1)

    def __init__(self, in_channel, out_channels, width, depth) -> None:
        #? Why?
        super().__init__()
        self.depth = depth

        widths = [width * 2 ** i for i in range(depth)]

        self.encoder0 = self.encoder(in_channel, widths[0] // 2, widths[0])

        self.set_encoders(widths, 0)

        self.bottleneck1 = self.bottleneck(widths[-1], widths[-1])

        self.bottleneck2 = self.bottleneck(widths[-1]*2, widths[-2])

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.set_decoders(widths[:-1], 0)

        setattr(self, f"decoder{len(widths)-2}", self.decoder(widths[0]*2, in_channel * 2, in_channel))

        self.output = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=(3, 3), padding=1),
        )

        self.weights_initialization()

    def convolve(self, in_channels, mid_channel, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, (3, 3), padding="same"),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channels, (3, 3), padding="same"),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def encoder(self, in_channels, mid_channel, out_channels):
        return nn.Sequential(
            self.convolve(in_channels, mid_channel, out_channels),
            nn.MaxPool2d((2, 2))
        )

    def bottleneck(self, in_channel, out_channel):
        return nn.Sequential(
            self.convolve(in_channel, in_channel, out_channel)
        )

    def decoder(self, in_channels, mid_channel, out_channels):
        return nn.Sequential(
            self.convolve(in_channels, mid_channel, out_channels)
        )

    def weights_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.double()
        inner_encoders = [x]
        for i in range(self.depth): 
            encoder = getattr(self, f"encoder{i}")
            inner_encoder = encoder(inner_encoders[-1])
            inner_encoders.append(inner_encoder)
        inner_encoders = inner_encoders[1:]

        b1 = self.bottleneck1(inner_encoders[-1])
        b2 = self.bottleneck2(torch.cat([inner_encoders[-1], b1], dim=1))
        inner_encoders = inner_encoders[:-1]

        inner_decoder = b2
        for i in range(self.depth - 1):
            upsample = self.upsample(inner_decoder)
            decoder = getattr(self, f"decoder{i}")
            inner_decoder = decoder(torch.cat([upsample, inner_encoders[-1]], dim=1))
            inner_encoders.pop(-1)

        out = self.output(self.upsample(inner_decoder))
        return out
