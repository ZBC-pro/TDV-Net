import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
from typing import Optional, List, Tuple

class FrFTKernel(nn.Module):

    def __init__(self, kernel_size: int, num_prototypes: int, alpha_init: Optional[torch.Tensor]=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_prototypes = num_prototypes
        if alpha_init is None:
            alpha_init = torch.linspace(0.2 * math.pi, 0.8 * math.pi, num_prototypes)
        self.alpha = nn.Parameter(alpha_init.clone())
        coords = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
        x1, x2 = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('x1', x1)
        self.register_buffer('x2', x2)
        self.register_buffer('x1_sq_plus_x2_sq', x1 ** 2 + x2 ** 2)
        self.register_buffer('x1_times_x2', x1 * x2)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        real_kernels = []
        imag_kernels = []
        for c in range(self.num_prototypes):
            alpha_c = self.alpha[c]
            sin_alpha = torch.sin(alpha_c)
            cos_alpha = torch.cos(alpha_c)
            eps = 1e-07
            sin_alpha_safe = sin_alpha + eps * torch.sign(sin_alpha + eps)
            cot_alpha = cos_alpha / sin_alpha_safe
            csc_alpha = 1.0 / sin_alpha_safe
            amplitude_mag = torch.pow(1 + cot_alpha ** 2, 0.25)
            amplitude_phase = alpha_c / 2 - math.pi / 4
            exp_phase = math.pi * (self.x1_sq_plus_x2_sq * cot_alpha - 2 * self.x1_times_x2 * csc_alpha)
            total_phase = amplitude_phase + exp_phase
            real_kernel = amplitude_mag * torch.cos(total_phase)
            imag_kernel = amplitude_mag * torch.sin(total_phase)
            real_kernels.append(real_kernel)
            imag_kernels.append(imag_kernel)
        return (torch.stack(real_kernels, dim=0), torch.stack(imag_kernels, dim=0))

class FractionalGaborFilter(nn.Module):

    def __init__(self, kernel_size: int, num_prototypes: int, num_orientations: int, num_scales: int, alpha_init: Optional[torch.Tensor]=None, sigma_base: float=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_prototypes = num_prototypes
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.sigma_base = sigma_base
        self.frft_kernel = FrFTKernel(kernel_size, num_prototypes, alpha_init)
        orientations = torch.linspace(0, math.pi, num_orientations + 1)[:-1]
        self.register_buffer('orientations', orientations)
        scales = torch.tensor([2.0 ** v for v in range(num_scales)])
        self.register_buffer('scales', scales)
        coords = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('x', x)
        self.register_buffer('y', y)

    def _compute_gabor_envelope(self, orientation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.cos(orientation)
        sin_theta = torch.sin(orientation)
        x_theta = self.x * cos_theta + self.y * sin_theta
        y_theta = -self.x * sin_theta + self.y * cos_theta
        sigma = self.sigma_base * scale
        envelope = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2 + 1e-08))
        return envelope

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        frft_real, frft_imag = self.frft_kernel()
        filters_real = []
        filters_imag = []
        for c in range(self.num_prototypes):
            scale_filters_real = []
            scale_filters_imag = []
            for v in range(self.num_scales):
                scale = self.scales[v]
                orient_filters_real = []
                orient_filters_imag = []
                for u in range(self.num_orientations):
                    orientation = self.orientations[u]
                    envelope = self._compute_gabor_envelope(orientation, scale)
                    frgt_filter_real = envelope * frft_real[c]
                    frgt_filter_imag = envelope * frft_imag[c]
                    magnitude = torch.sqrt(frgt_filter_real ** 2 + frgt_filter_imag ** 2)
                    norm_factor = magnitude.sum() + 1e-08
                    frgt_filter_real = frgt_filter_real / norm_factor
                    frgt_filter_imag = frgt_filter_imag / norm_factor
                    orient_filters_real.append(frgt_filter_real)
                    orient_filters_imag.append(frgt_filter_imag)
                scale_filters_real.append(torch.stack(orient_filters_real, dim=0))
                scale_filters_imag.append(torch.stack(orient_filters_imag, dim=0))
            filters_real.append(torch.stack(scale_filters_real, dim=0))
            filters_imag.append(torch.stack(scale_filters_imag, dim=0))
        return (torch.stack(filters_real, dim=0), torch.stack(filters_imag, dim=0))

class TextureAdaptiveConv(nn.Module):

    def __init__(self, in_channels: int, out_channels_per_group: int, kernel_size: int, num_prototypes: int, num_orientations: int, num_scales: int, alpha_init: Optional[torch.Tensor]=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels_per_group = out_channels_per_group
        self.kernel_size = kernel_size
        self.num_prototypes = num_prototypes
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.base_kernel = nn.Parameter(torch.randn(out_channels_per_group, in_channels, kernel_size, kernel_size) * 0.02)
        self.frgt_filters = FractionalGaborFilter(kernel_size=kernel_size, num_prototypes=num_prototypes, num_orientations=num_orientations, num_scales=num_scales, alpha_init=alpha_init)
        self.out_channels = out_channels_per_group * num_prototypes * num_orientations

    def _kernel_convolution(self, base_kernel: torch.Tensor, psi_filter: torch.Tensor) -> torch.Tensor:
        out_c, in_c, k, _ = base_kernel.shape
        base_flat = base_kernel.reshape(out_c * in_c, 1, k, k)
        psi = psi_filter.unsqueeze(0).unsqueeze(0)
        padding = k // 2
        modulated_flat = F.conv2d(base_flat, psi, padding=padding)
        out_size = modulated_flat.shape[-1]
        if out_size > k:
            start = (out_size - k) // 2
            modulated_flat = modulated_flat[:, :, start:start + k, start:start + k]
        return modulated_flat.reshape(out_c, in_c, k, k)

    def forward(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        psi_real, psi_imag = self.frgt_filters()
        modulated_kernels = []
        for c in range(self.num_prototypes):
            for u in range(self.num_orientations):
                psi_real_cuv = psi_real[c, scale_idx, u]
                psi_imag_cuv = psi_imag[c, scale_idx, u]
                modulated_real = self._kernel_convolution(self.base_kernel, psi_real_cuv)
                modulated_imag = self._kernel_convolution(self.base_kernel, psi_imag_cuv)
                modulated = torch.sqrt(modulated_real ** 2 + modulated_imag ** 2 + 1e-08)
                modulated_kernels.append(modulated)
        kernel = torch.cat(modulated_kernels, dim=0)
        padding = self.kernel_size // 2
        y = F.conv2d(x, kernel, padding=padding)
        return y

class MFSU(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_prototypes: int=4, num_orientations: int=4, num_scales: int=4, kernel_size: int=3, alpha_init: Optional[List[float]]=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_prototypes = num_prototypes
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.kernel_size = kernel_size
        assert in_channels % num_scales == 0, f'in_channels ({in_channels}) must be divisible by num_scales ({num_scales})'
        self.channels_per_scale = in_channels // num_scales
        self.out_per_group = max(1, self.channels_per_scale // (num_prototypes * num_orientations))
        if alpha_init is not None:
            alpha_tensor = torch.tensor(alpha_init, dtype=torch.float32)
        else:
            alpha_tensor = torch.linspace(0.2 * math.pi, 0.8 * math.pi, num_prototypes)
        self.texture_convs = nn.ModuleList([TextureAdaptiveConv(in_channels=self.channels_per_scale, out_channels_per_group=self.out_per_group, kernel_size=kernel_size, num_prototypes=num_prototypes, num_orientations=num_orientations, num_scales=num_scales, alpha_init=alpha_tensor) for _ in range(num_scales)])
        texture_out_channels = num_scales * num_prototypes * num_orientations * self.out_per_group
        self.proj = nn.Sequential(nn.Conv2d(texture_out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_scales = torch.split(x, self.channels_per_scale, dim=1)
        y_scales = []
        for v, (x_v, conv_v) in enumerate(zip(x_scales, self.texture_convs)):
            y_v = conv_v(x_v, scale_idx=v)
            y_scales.append(y_v)
        y = torch.cat(y_scales, dim=1)
        y = self.proj(y)
        y = y + self.shortcut(x)
        return y

class MFSUBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_prototypes: int=4, num_orientations: int=4, num_scales: int=4):
        super().__init__()
        mid_channels = out_channels
        mid_channels = mid_channels // num_scales * num_scales
        if mid_channels == 0:
            mid_channels = num_scales
        self.first_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True))
        alpha_init = [0.2 * math.pi, 0.4 * math.pi, 0.6 * math.pi, 0.8 * math.pi][:num_prototypes]
        self.fpu = MFSU(in_channels=mid_channels, out_channels=out_channels, num_prototypes=num_prototypes, num_orientations=num_orientations, num_scales=num_scales, alpha_init=alpha_init)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.fpu(x)
        return x

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

class U_MFSU(nn.Module):

    def __init__(self, options: dict):
        super().__init__()
        train_vars = options.get('train_variables', [])
        self.sar_indices = [i for i, v in enumerate(train_vars) if 'sar' in v.lower()]
        self.aux_indices = [i for i, v in enumerate(train_vars) if 'sar' not in v.lower()]
        n_sar = len(self.sar_indices)
        n_aux = len(self.aux_indices)
        n_classes = options.get('n_classes', {})
        f = [32, 32, 64, 128, 256]
        self.num_prototypes = options.get('num_prototypes', 4)
        self.num_orientations = options.get('num_orientations', 4)
        self.num_scales = options.get('num_scales', 4)
        self.sar_enc1 = DoubleConvolution(n_sar, f[0])
        self.aux_enc1 = DoubleConvolution(n_aux, f[0])
        self.down1 = DownSample()
        self.sar_enc2 = MFSUBlock(f[0], f[1], num_prototypes=self.num_prototypes, num_orientations=self.num_orientations, num_scales=self.num_scales)
        self.aux_enc2 = DoubleConvolution(f[0], f[1])
        self.down2 = DownSample()
        self.sar_enc3 = MFSUBlock(f[1], f[2], num_prototypes=self.num_prototypes, num_orientations=self.num_orientations, num_scales=self.num_scales)
        self.aux_enc3 = DoubleConvolution(f[1], f[2])
        self.down3 = DownSample()
        self.sar_enc4 = MFSUBlock(f[2], f[3], num_prototypes=self.num_prototypes, num_orientations=self.num_orientations, num_scales=self.num_scales)
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
        print(f'Decoder: 4 stages with skip connections')
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
if __name__ == '__main__':
    pass
