import torch
import torch.nn as nn
import torchvision.transforms.functional
from torchvision.ops import DeformConv2d

class DoubleConvolution(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.first(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.second(x)
        x = self.bn2(x)
        return self.act2(x)

class Deform_Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offsets = self.offset_conv(x)
        x = self.deform_conv(x, offsets)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DownSample(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)

class UpSample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class CropAndConcat(nn.Module):

    def forward(self, x, contracting_x):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x

class U_Deform(nn.Module):

    def __init__(self, options: dict):
        super().__init__()
        print('\nInitializing UNet_Deform (Dual Stream UNet with Deformable Convs in SAR stream)...')
        train_vars = options.get('train_variables', [])
        self.sar_indices = [i for i, v in enumerate(train_vars) if 'sar' in v.lower()]
        self.aux_indices = [i for i, v in enumerate(train_vars) if 'sar' not in v.lower()]
        n_sar = len(self.sar_indices)
        n_aux = len(self.aux_indices)
        n_classes = options.get('n_classes', {})
        f = [32, 32, 64, 128, 256]
        print(f'Channels config: {f}')
        self.sar_enc1 = DoubleConvolution(n_sar, f[0])
        self.aux_enc1 = DoubleConvolution(n_aux, f[0])
        self.down1 = DownSample()
        self.sar_enc2 = Deform_Conv(f[0], f[1])
        self.aux_enc2 = DoubleConvolution(f[0], f[1])
        self.down2 = DownSample()
        self.sar_enc3 = Deform_Conv(f[1], f[2])
        self.aux_enc3 = DoubleConvolution(f[1], f[2])
        self.down3 = DownSample()
        self.sar_enc4 = Deform_Conv(f[2], f[3])
        self.aux_enc4 = DoubleConvolution(f[2], f[3])
        self.down4 = DownSample()
        self.bottle_fusion = nn.Conv2d(f[3] * 2, f[4], 1)
        self.bottle_conv = DoubleConvolution(f[4], f[4])
        self.concat = CropAndConcat()
        self.up4 = UpSample(f[4], f[3])
        self.dec4 = DoubleConvolution(f[3] * 3, f[3])
        self.up3 = UpSample(f[3], f[2])
        self.dec3 = DoubleConvolution(f[2] * 3, f[2])
        self.up2 = UpSample(f[2], f[1])
        self.dec2 = DoubleConvolution(f[1] * 3, f[1])
        self.up1 = UpSample(f[1], f[0])
        self.dec1 = DoubleConvolution(f[0] * 3, f[0])
        final_c = f[0]
        self.sic_head = nn.Conv2d(final_c, n_classes.get('SIC', 1), 1)
        self.sod_head = nn.Conv2d(final_c, n_classes.get('SOD', 1), 1)
        self.floe_head = nn.Conv2d(final_c, n_classes.get('FLOE', 1), 1)

    def forward(self, x):
        x_sar = x[:, self.sar_indices]
        x_aux = x[:, self.aux_indices]
        s1 = self.sar_enc1(x_sar)
        a1 = self.aux_enc1(x_aux)
        s1_d = self.down1(s1)
        a1_d = self.down1(a1)
        s2 = self.sar_enc2(s1_d)
        a2 = self.aux_enc2(a1_d)
        s2_d = self.down2(s2)
        a2_d = self.down2(a2)
        s3 = self.sar_enc3(s2_d)
        a3 = self.aux_enc3(a2_d)
        s3_d = self.down3(s3)
        a3_d = self.down3(a3)
        s4 = self.sar_enc4(s3_d)
        a4 = self.aux_enc4(a3_d)
        s4_d = self.down4(s4)
        a4_d = self.down4(a4)
        b_in = torch.cat([s4_d, a4_d], dim=1)
        b = self.bottle_fusion(b_in)
        b = self.bottle_conv(b)
        d4 = self.up4(b)
        d4 = self.concat(d4, torch.cat([s4, a4], dim=1))
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = self.concat(d3, torch.cat([s3, a3], dim=1))
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self.concat(d2, torch.cat([s2, a2], dim=1))
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self.concat(d1, torch.cat([s1, a1], dim=1))
        d1 = self.dec1(d1)
        return {'SIC': self.sic_head(d1), 'SOD': self.sod_head(d1), 'FLOE': self.floe_head(d1)}
