import torch.nn as nn

class UNet(nn.Module):
    """
    UNet is a convolutional neural network architecture for image segmentation.
    It consists of an encoder and a decoder, with skip connections between them.

    Args:
        in_channels (int): Number of input channels (default: 3)
        channels (list): List of channel sizes for each layer in the encoder and decoder (default: [64, 64, 96, 128, 128, 128, 128, 96])

    Attributes:
        encoder (nn.Sequential): Encoder module consisting of convolutional layers and max pooling layers
        decoder (nn.Sequential): Decoder module consisting of up-convolutional layers and upsampling layers

    """

    def __init__(self, out_channels, in_channels=3, channels=[64, 64, 96, 128, 128, 128, 128, 96]):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self._conv(in_channels, channels[0]),
            self._conv(channels[0], channels[1]),
            nn.MaxPool2d(2),
            self._conv(channels[1], channels[2]),
            self._conv(channels[2], channels[3]),
            nn.MaxPool2d(2),
            self._conv(channels[3], channels[4]),
            self._conv(channels[4], channels[5]),
            self._conv(channels[5], channels[6]),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(channels[6], channels[7], 4, 1, 0),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(
            self._upconv(channels[7], channels[6], 4, 1, 0),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            self._conv(channels[6], channels[5]),
            self._conv(channels[5], channels[4]),
            self._conv(channels[4], channels[3]),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            self._conv(channels[3], channels[2]),
            self._conv(channels[2], channels[1]),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            self._conv(channels[1], channels[0]),
            nn.Sequential(
                nn.ConvTranspose2d(channels[0], out_channels, 3, 1, 1),
                nn.Sigmoid()
            )
        )

    def _conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Helper function to create a convolutional layer with batch normalization and LeakyReLU activation.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the convolutional kernel (default: 3)
            stride (int): Stride of the convolution (default: 1)
            padding (int): Padding of the convolution (default: 1)

        Returns:
            nn.Sequential: Convolutional layer with batch normalization and LeakyReLU activation
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _upconv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Helper function to create an up-convolutional layer with batch normalization and LeakyReLU activation.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the up-convolutional kernel (default: 3)
            stride (int): Stride of the up-convolution (default: 1)
            padding (int): Padding of the up-convolution (default: 1)

        Returns:
            nn.Sequential: Up-convolutional layer with batch normalization and LeakyReLU activation
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        
        for idx, layer in enumerate(self.decoder):
            x = layer(x + encoder_outputs[-(idx + 1)])
        
        return x