import math
from typing import List, Tuple

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Generic MLP classifier for tabular and image data (with flatten)."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        input_dim = 1
        for d in input_shape:
            input_dim *= d

        layers: List[nn.Module] = [nn.Flatten()]
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularCNN(nn.Module):
    """Simple 1D CNN for tabular data (treat features as a 1D sequence)."""

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, 1, F)
        x = x.view(x.size(0), 1, self.num_features)
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageCNN(nn.Module):
    """Basic CNN for CIFAR-10 / PCam images."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class TabularAttentionMLP(nn.Module):
    """Attention-based MLP for tabular data.

    Uses a simple feature-wise attention mechanism:
    - compute attention weights over features
    - reweight features before passing to an MLP
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_features),
        )

        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        scores = self.attn(x)  # (B, F)
        alpha = torch.softmax(scores, dim=1)
        x_att = x * alpha
        return self.mlp(x_att)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Minimal ViT-style encoder for images."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x


def create_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    hidden_dims: List[int],
    dropout: float,
    use_batchnorm: bool,
    dataset_name: str,
) -> nn.Module:
    """Factory for models based on config."""
    if model_name == "mlp":
        return MLPClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    if model_name == "cnn":
        if len(input_shape) == 1:
            # tabular
            return TabularCNN(num_features=input_shape[0], num_classes=num_classes)
        elif len(input_shape) == 3:
            in_channels = input_shape[0]
            return ImageCNN(in_channels=in_channels, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported input shape for CNN: {input_shape}")

    if model_name == "attention":
        if dataset_name == "adult":
            return TabularAttentionMLP(
                num_features=input_shape[0],
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        else:
            # image datasets -> Vision Transformer
            if len(input_shape) != 3:
                raise ValueError("Attention model for images expects CHW input shape")
            _, h, _ = input_shape
            patch_size = 8 if h >= 64 else 4
            return VisionTransformer(
                img_size=h,
                patch_size=patch_size,
                in_channels=input_shape[0],
                num_classes=num_classes,
            )

    raise ValueError(f"Unknown model name: {model_name}")


