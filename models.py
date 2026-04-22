from __future__ import annotations
import torch
import torchvision

def _adapt_first_conv(conv: torch.nn.Conv2d, in_channels: int) -> torch.nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    new_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv

def create_model(name: str, num_classes: int, in_channels: int = 1, pretrained: bool = False) -> torch.nn.Module:
    name = name.lower()
    if name in ("resnet18", "resnet"):
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.conv1 = _adapt_first_conv(model.conv1, in_channels)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    if name in ("vgg11_bn", "vgg"):
        weights = torchvision.models.VGG11_BN_Weights.DEFAULT if pretrained else None
        model = torchvision.models.vgg11_bn(weights=weights)
        first_conv = model.features[0]
        assert isinstance(first_conv, torch.nn.Conv2d)
        model.features[0] = _adapt_first_conv(first_conv, in_channels)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError("Unknown model name. Use resnet18 or vgg11_bn.")
