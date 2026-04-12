"""
DDA-ViT - Dual Domain Adaptive Vision Transformer

Refer to Code/DDA-ViT/DDA-ViT.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SteelBackbone(nn.Module):
    """
    - Steel Backbone
    - SegFormer encoder
    """

    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder  # SegFormer encoder

    def forward(self, x):
        features = self.encoder(x)  # list of feature maps
        return features[-1]         # deepest feature map


class SugarBackbone(nn.Module):
    """
    - Sugar Backbone
    - Swin Transformer
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.forward_features(x)  # supported by timm
        return features


class FeatureProjector(nn.Module):
    """
    - Projects features into a shared embedding space
    """

    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CrossDomainAttention(nn.Module):
    """
    - Cross-Domain Feature Alignment
    - Done via Multi-Head Attention
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, Fs, Fq):
        out, _ = self.attn(Fs, Fq, Fq)
        return out


class DDAViT(nn.Module):
    """
    - Dual Domain Adaptive Vision Transformer

        - Wraps both steel and sugar backbones
        - Cross-domain attention module

    - During inference
        - pass EITHER x_steel OR x_sugar
        - NOT both simultaneously for a single image
    """

    def __init__(self, steel_model, sugar_model, embed_dim=256,
                 num_defect_classes=4, num_quality_classes=4):
        super().__init__()

        self.steel = SteelBackbone(steel_model)
        self.sugar = SugarBackbone(sugar_model)

        self.proj_s = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.proj_q = nn.Linear(768, embed_dim)

        self.cross_attn = CrossDomainAttention(embed_dim)

        self.seg_head = nn.Conv2d(embed_dim, num_defect_classes, kernel_size=1)
        self.sugar_head = nn.Linear(embed_dim, num_quality_classes)

    def forward(self, x_steel=None, x_sugar=None):

        Fs_map, Fq = None, None

        if x_steel is not None:
            Fs_map = self.steel(x_steel)                        # (B, C, H, W)
            Fs_map = self.proj_s(Fs_map)                        # (B, d, H, W)

            B, d, H, W = Fs_map.shape
            Fs = Fs_map.flatten(2).transpose(1, 2)              # (B, N, d)

        if x_sugar is not None:
            Fq = self.sugar(x_sugar)

            if Fq.dim() == 4:                                   # (B, H, W, C)
                B, H, W, C = Fq.shape
                Fq = Fq.view(B, H * W, C)

            if Fq.dim() == 3:                                   # (B, N, C)
                Fq = self.proj_q(Fq)
                Fq = Fq.mean(dim=1, keepdim=True)

            elif Fq.dim() == 2:                                 # (B, C)
                Fq = self.proj_q(Fq)
                Fq = Fq.unsqueeze(1)

            else:
                raise ValueError(f"Unexpected sugar feature shape: {Fq.shape}")

        if Fs_map is not None and Fq is not None:
            Fs_fused = self.cross_attn(Fs, Fq)  # (B, N, d)
            Fs_map = Fs_fused.transpose(1, 2).reshape(B, d, H, W)

        if x_steel is not None:
            seg_out = self.seg_head(Fs_map)

            # Upsample to input size
            seg_out = F.interpolate(
                seg_out,
                size=x_steel.shape[2:],  # (H, W)
                mode="bilinear",
                align_corners=False
            )

            return seg_out

        if x_sugar is not None:
            F_out = Fq.mean(dim=1)  # (B, d)
            return self.sugar_head(F_out)

        raise ValueError("Provide x_steel or x_sugar")