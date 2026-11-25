"""Custom CNN and pretrained model factories."""

import torch
import torch.nn as nn
import torch.nn.functional as F

TRANSFER_MODELS = {
    "EfficientNetV2B0": ("tf_efficientnetv2_b0", "efficientnetv2_b0"),
    "ConvNeXtTiny": ("convnext_tiny",),
    "Xception": ("xception", "legacy_xception"),
}


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, inputs):
        batch, channels, _, _ = inputs.size()
        weights = inputs.mean(dim=(2, 3))
        weights = F.silu(self.fc1(weights))
        weights = torch.sigmoid(self.fc2(weights)).view(batch, channels, 1, 1)
        return inputs * weights


class ResidualSEBlock(nn.Module):
    """Residual convolutional block with squeeze-and-excitation attention."""

    def __init__(self, in_channels, out_channels, stride=1, drop_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs):
        outputs = F.silu(self.bn1(self.conv1(inputs)))
        outputs = self.drop(outputs)
        outputs = self.bn2(self.conv2(outputs))
        outputs = self.se(outputs)
        return F.silu(outputs + self.shortcut(inputs))


class EnhancedResidualSE_CNN(nn.Module):
    """Custom four-stage residual SE network used in the Kaggle experiment."""

    def __init__(self, num_classes=3, dropout_rate=0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage1 = nn.Sequential(
            ResidualSEBlock(64, 64, drop_rate=0.05),
            ResidualSEBlock(64, 64, drop_rate=0.05),
        )
        self.stage2 = nn.Sequential(
            ResidualSEBlock(64, 128, stride=2, drop_rate=0.10),
            ResidualSEBlock(128, 128, drop_rate=0.10),
        )
        self.stage3 = nn.Sequential(
            ResidualSEBlock(128, 256, stride=2, drop_rate=0.15),
            ResidualSEBlock(256, 256, drop_rate=0.15),
        )
        self.stage4 = nn.Sequential(
            ResidualSEBlock(256, 512, stride=2, drop_rate=0.15),
            ResidualSEBlock(512, 512, drop_rate=0.15),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512 * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)
        features = torch.cat([self.avg_pool(outputs), self.max_pool(outputs)], dim=1)
        return self.classifier(features)


def load_timm_model(candidates, num_classes=3, drop_rate=0.3, pretrained=True):
    """Load the first compatible model from a sequence of timm identifiers."""
    import timm

    candidates = tuple(candidates)
    last_error = None
    for model_id in candidates:
        try:
            model = timm.create_model(
                model_id,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
            )
            return model, model_id
        except Exception as error:
            last_error = error

    requested_models = ", ".join(candidates)
    message = f"Could not load any of the timm models: {requested_models}."
    raise RuntimeError(message) from last_error
