import torch
import torch.nn as nn


class UNet3D_Cond(nn.Module):
    def __init__(self, base_channels: int = 16, time_dim: int = 32, timesteps: int = 1000):
        super().__init__()
        self.time_embed = nn.Embedding(timesteps, time_dim)

        def conv_block(in_ch, out_ch) -> nn.Module:
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
            )

        # Ahora in_ch = 3 (x_t) + 3 (y) + time_dim
        in_channels = 3 + 3 + time_dim
        self.enc1 = conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = conv_block(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(base_channels*2, base_channels*4)

        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_channels*2, base_channels)

        self.final = nn.Conv3d(base_channels, 3, kernel_size=1)

    def forward(self, x_t, y, t):
        """
        x_t: (B, 3, T, H, W)  # ruido añadido
        y:   (B, 3, T, H, W)  # clip ruidoso original
        t:   (B,) índices de timestep
        """
        B, C, T, H, W = x_t.shape

        # Embedding temporal
        te = self.time_embed(t)                       # (B, time_dim)
        te = te.view(B, -1, 1, 1, 1).expand(-1, -1, T, H, W)

        # Concatenar x_t, y y embedding: (B, 3+3+time_dim, T, H, W)
        xt = torch.cat([x_t, y, te], dim=1)

        e1 = self.enc1(xt)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)  # (B, 3, T, H, W)
