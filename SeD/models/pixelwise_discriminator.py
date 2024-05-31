import torch
import torch.nn as nn
from models.semantic_aware_fusion_block import SemanticAwareFusionBlock

class UNetPixelDiscriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super(UNetPixelDiscriminator, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, num_filters),
            self._conv_block(num_filters, num_filters),
            self._conv_block(num_filters, num_filters * 2),
            self._conv_block(num_filters * 2, num_filters * 4),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            self._upconv_block(num_filters * 4, num_filters * 4),
            self._upconv_block(num_filters * 4, num_filters * 2),
            self._upconv_block(num_filters * 2, num_filters),
            self._upconv_block(num_filters, num_filters),
            nn.Conv2d(num_filters, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, fs):
        # Encoder
        # fs = 64
        enc1 = self.encoder[0](fs) # 32x32x64
        enc2 = self.encoder[1](enc1) # 16x16x64
        enc3 = self.encoder[2](enc2) # 8x8x128
        enc4 = self.encoder[3](enc3) # 4x4x256

        #Bottleneck
        bottleneck = self.bottleneck(enc4) # 2x2x256

        #Decoder with skip connections using addition
        dec = self.decoder[0](bottleneck) # 4x4x256
        dec = self.decoder[1](dec + enc4) # 8x8x128
        dec = self.decoder[2](dec + enc3) # 16x16x64
        dec = self.decoder[3](dec + enc2) # 32x32x64
        dec = self.decoder[4](dec + enc1) # 64x64x1

        return dec


#---------------------------------------------

class DownSamplerPx(nn.Module):
    #downsamples 4 times in a conv, bn, leaky relu fashion that halves the spatial dimensions in each step and doubles the number of filters
    def __init__(self, input_channels, num_filters=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x
    
class UNetPixelDiscriminatorwithSed(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super(UNetPixelDiscriminatorwithSed, self).__init__()

        #downsampler takes 256x256 images and downsamples to the 16x16
        #to make dimensionality compatible with semantic feature maps
        self.downsampler = DownSamplerPx(in_channels, num_filters)
        
        # Semantic Aware Fusion Blocks
        self.semantic_aware_fusion_block1 = SemanticAwareFusionBlock(channel_size_changer_input_nc=64)
        self.semantic_aware_fusion_block2 = SemanticAwareFusionBlock(channel_size_changer_input_nc=1024)
        self.semantic_aware_fusion_block3 = SemanticAwareFusionBlock(channel_size_changer_input_nc=1024)
        
        self.upconv1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.upconv2 = nn.Conv2d(1024, 64, kernel_size=1, stride=1)
        self.upconv3 = nn.Conv2d(64, 3, kernel_size=1, stride=1)


    def forward(self,semantic_feature_maps, fs):
        x = self.downsampler(fs)
        enc1 = self.semantic_aware_fusion_block1(semantic_feature_maps, x)
        enc2 = self.semantic_aware_fusion_block2(semantic_feature_maps, enc1)
        enc3 = self.semantic_aware_fusion_block3(semantic_feature_maps, enc2)
        
        dec = self.upconv1(enc3 + enc2)
        dec = self.upconv2(dec + enc1)
        dec = self.upconv3(dec + x)
        
        return dec