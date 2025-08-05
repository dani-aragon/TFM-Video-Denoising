import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, base_channels: int =16):
        """
        U-Net 3D with two pooling layers. First spatial dimension is time.
        
        Args:
            base_channels (int): number of kernels in the first convolutional
                layer (n_in). The number of kernels in the following layers
                will be base_channels*2, base_channels*4, base_channels*2 and
                base_channels.
        """
        super(UNet3D, self).__init__()
        
        # Double 3D-convolutional block.
        def conv_block(in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv3d(
                    in_ch, out_ch, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    out_ch, out_ch, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        # Encoder.
        self.enc1 = conv_block(3, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = conv_block(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck.
        self.bottleneck = conv_block(base_channels*2, base_channels*4)
        
        # Decoder.
        self.up2 = nn.ConvTranspose3d(
            base_channels*4, base_channels*2, kernel_size=2, stride=2
        )
        self.dec2 = conv_block(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose3d(
            base_channels*2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = conv_block(base_channels*2, base_channels)
        
        # Last layer: to 3 channels (RGB).
        self.final = nn.Conv3d(base_channels, 3, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Input:
            x (torch.Tensor): tensor with shape (B, C=3, T, H, W).
        Output:
            torch.Tensor with shape (B, 3, T, H, W).
        """
        # Encoder.
        e1 = self.enc1(x) # -> (B, base, T, H, W)
        p1 = self.pool1(e1) # -> (B, base, T/2, H/2, W/2)
        
        e2 = self.enc2(p1) # -> (B, base*2, T/2, H/2, W/2)
        p2 = self.pool2(e2) # -> (B, base*2, T/4, H/4, W/4)
        
        # Bottleneck.
        b = self.bottleneck(p2) # -> (B, base*4, T/4, H/4, W/4)
        
        # Decoder.
        u2 = self.up2(b) # -> (B, base*2, T/2, H/2, W/2)
        c2 = torch.cat([u2, e2], dim=1) # skip connection
        d2 = self.dec2(c2) # -> (B, base*2, T/2, H/2, W/2)
        
        u1 = self.up1(d2) # -> (B, base, T, H, W)
        c1 = torch.cat([u1, e1], dim=1) # skip connection
        d1 = self.dec1(c1) # -> (B, base, T, H, W)
        
        # Output.
        out = self.final(d1) # -> (B, 3, T, H, W)
        return out


class UNet3D_3(nn.Module):
    def __init__(self, base_channels: int =16):
        """
        U-Net 3D with two pooling layers. First spatial dimension is time.
        
        Args:
            base_channels (int): number of kernels in the first convolutional
                layer (n_in). The number of kernels in the following layers
                will be base_channels*2, base_channels*4, base_channels*2 and
                base_channels.
        """
        super(UNet3D, self).__init__()
        
        # Double 3D-convolutional block.
        def conv_block(in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv3d(
                    in_ch, out_ch, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    out_ch, out_ch, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        # Encoder.
        self.enc1 = conv_block(3, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = conv_block(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = conv_block(base_channels, base_channels*4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck.
        self.bottleneck = conv_block(base_channels*4, base_channels*8)
        
        # Decoder.
        self.up3 = nn.ConvTranspose3d(
            base_channels*8, base_channels*4, kernel_size=2, stride=2
        )
        self.dec3 = conv_block(base_channels*8, base_channels*4)
        self.up2 = nn.ConvTranspose3d(
            base_channels*4, base_channels*2, kernel_size=2, stride=2
        )
        self.dec2 = conv_block(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose3d(
            base_channels*2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = conv_block(base_channels*2, base_channels)
        
        # Last layer: to 3 channels (RGB).
        self.final = nn.Conv3d(base_channels, 3, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Input:
            x (torch.Tensor): tensor with shape (B, C=3, T, H, W).
        Output:
            torch.Tensor with shape (B, 3, T, H, W).
        """
        # Encoder.
        e1 = self.enc1(x) # -> (B, base, T, H, W)
        p1 = self.pool1(e1) # -> (B, base, T/2, H/2, W/2)
        
        e2 = self.enc2(p1) # -> (B, base*2, T/2, H/2, W/2)
        p2 = self.pool2(e2) # -> (B, base*2, T/4, H/4, W/4)
        
        e3 = self.enc2(p2) # -> (B, base*4, T/4, H/4, W/4)
        p3 = self.pool2(e3) # -> (B, base*4, T/8, H/8, W/8)
        
        # Bottleneck.
        b = self.bottleneck(p3) # -> (B, base*8, T/8, H/8, W/8)
        
        # Decoder.
        u3 = self.up2(b) # -> (B, base*4, T/4, H/4, W/4)
        c3 = torch.cat([u3, e3], dim=1) # skip connection
        d3 = self.dec2(c3) # -> (B, base*4, T/4, H/4, W/4)

        u2 = self.up2(d3) # -> (B, base*2, T/2, H/2, W/2)
        c2 = torch.cat([u2, e2], dim=1) # skip connection
        d2 = self.dec2(c2) # -> (B, base*2, T/2, H/2, W/2)
        
        u1 = self.up1(d2) # -> (B, base, T, H, W)
        c1 = torch.cat([u1, e1], dim=1) # skip connection
        d1 = self.dec1(c1) # -> (B, base, T, H, W)
        
        # Output.
        out = self.final(d1) # -> (B, 3, T, H, W)
        return out
