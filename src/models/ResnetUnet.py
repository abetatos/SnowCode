import torch
import torch.nn as nn
import torchvision.models as models

class ResnetUnet(nn.Module):
    
    def __init__(self, in_channels, out_channels, pretrained=True, **kwargs):
        super().__init__()

        self.scaler = nn.Conv2d(in_channels, 1, 3)
        # Encoder
        self.encoder = models.resnet18(pretrained=pretrained)
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.layer4 = nn.Identity()
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        # Encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        
        # Decoder
        x = self.decoder[0](x)
        x = torch.cat([x, self.encoder.layer2(x)], dim=1)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = self.decoder[3](x)
        x = torch.cat([x, self.encoder.layer1(x)], dim=1)
        x = self.decoder[4](x)
        x = self.decoder[5](x)
        x = self.decoder[6](x)
        x = torch.cat([x, self.encoder.conv1(x)], dim=1)
        x = self.decoder[7](x)
        
        return x