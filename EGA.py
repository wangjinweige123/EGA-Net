import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.sigmoid(y)

class PixelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.gelu(y)
        y = self.conv2(y)
        return self.sigmoid(y)

class EdgeAttention(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAttention, self).__init__()
        self.gaussian = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        kernel_size = 5
        sigma = 1.0
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
                
        self.gaussian.weight.data = gaussian_kernel
        self.gaussian.weight.requires_grad = False
                
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
                
        sobel_kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
               
        self.sobel_x.weight.data = sobel_kernel_x
        self.sobel_y.weight.data = sobel_kernel_y
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
  
        self.gray_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.gray_conv.weight, 1.0/in_channels)
        self.gray_conv.weight.requires_grad = False
      
        self.depth_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, groups=1)
        self.point_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
     
        gray = self.gray_conv(x)
        
    
        blurred = self.gaussian(gray)
        
  
        grad_x = self.sobel_x(blurred)
        grad_y = self.sobel_y(blurred)
           
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
         
        edges = grad_magnitude
        
        edge = self.depth_conv(edges)
        edge = self.point_conv(edge)
        
        return self.sigmoid(edge)

class EGA(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super(EGA, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.pa = PixelAttention(in_channels, reduction_ratio // 2)
        self.ea = EdgeAttention(in_channels)
        
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.fusion = FeatureFusionModule(in_channels)
        
    def forward(self, x):

        x_norm = self.bn(x)

        ca_weights = self.ca(x_norm)
        pa_weights = self.pa(x_norm)
        ea_weights = self.ea(x_norm)
        

        guided_pa_weights = pa_weights + ea_weights  

        ca_features = x_norm * ca_weights
        guided_pa_features = x_norm * guided_pa_weights

        ca_features = self.alpha * ca_features
        guided_pa_features = self.beta * guided_pa_features

        concatenated = torch.cat([ca_features, guided_pa_features], dim=1)
        fused_features = self.fusion(concatenated)

        output = x + fused_features
        
        return output

class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":

    batch_size = 2
    channels = 64
    height = 256
    width = 256
    x = torch.randn(batch_size, channels, height, width)

    ega = EGA(in_channels=channels)

    output = ega(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")