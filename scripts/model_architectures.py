# model_architectures.py

# PyTorch imports
import torch
from torch import nn
from torch.nn import functional as F

class ConvNeXtBlock(nn.Module):
    """
    An implementation of ConvNeXt block as described in the ConvNeXt paper for CIFAR-100 with adaptations.
    @Article{liu2022convnet,
        author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
        title   = {A ConvNet for the 2020s},
        journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year    = {2022},
    }
    """
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Adapted to using a smaller kernel size than the ConvNeXt paper for CIFAR-100 low dimensionality
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise/1x1 convs
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # Change to (B, H, W, C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # Change back to (B, C, H, W)
        x = input + self.drop_path(x)
        return x

class DropPath(nn.Module):
    """DropPath (Stochastic Depth) regularization as in timm."""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Generate binary mask
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast across batch
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        # Scale output to maintain expected value
        return x.div(keep_prob) * binary_mask
    
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x
    
class ConvNeXtNet(nn.Module):
    """
    A ConvNeXt architecture integrated with Triple SE blocks adapted for CIFAR-100 classification.
    """
    def __init__(self, in_channels=3, num_classes=100, 
                 depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4, 
                 layer_scale_init_value=1e-6, head_init_scale=1, dropout_prob=0.0
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()    # Stem and 3 downsampling layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1), # Adapted kernel size and stride for CIFAR-100
            LayerNorm2d(dims[0], eps=1e-6) # Adapted for CIFAR-100 input size
        )
        self.downsample_layers.append(stem)

        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final norm layer
        self.dropout = nn.Dropout(dropout_prob)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            self.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def trunc_normal_(self, tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        """
        Fills the input tensor with values drawn from a truncated normal distribution.
        Values are drawn from a normal distribution with given mean and std,
        but are clipped to the range [a, b].
        """
        # Get the shape
        size = tensor.shape

        # Sample from a normal distribution
        tmp = torch.empty(size).normal_(mean=mean, std=std)

        # Truncate values outside [a, b]
        tmp = tmp.clamp(min=a, max=b)

        # Copy to tensor
        tensor.data.copy_(tmp)
        return tensor
    
    def forward_features(self, x):
        for downsample_layer, stage in zip(self.downsample_layers, self.stages):
            x = downsample_layer(x)
            x = stage(x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
    
def create_model(num_classes, device):
    """Create and initialize the model"""
    model = ConvNeXtNet(num_classes=num_classes)
    model = model.to(device)
    return model