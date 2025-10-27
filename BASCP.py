import torch
import torch.nn as nn
import torch.nn.functional as F

class StripPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
       
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))
        self.pool_v = nn.AdaptiveAvgPool2d((None, 1))
               
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
              
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        identity = x
               
        h = self.pool_h(x)
        h = self.conv_h(h)

        v = self.pool_v(x)
        v = self.conv_v(v)
               
        h = F.interpolate(h, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        v = F.interpolate(v, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
             
        out = h + v
        out = self.sigmoid(out)
               
        return out * identity

class BASCP(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
     
        self.conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)  # 1x1卷积
        self.conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')  # 5x5卷积
           
        self.branch_conv1x1 = nn.Conv2d(dim, dim, kernel_size=1)
    
        self.dw_conv_rate1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, dilation=1, padding_mode='reflect')
        self.dw_conv_rate2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, groups=dim, dilation=2, padding_mode='reflect')
        self.dw_conv_rate4 = nn.Conv2d(dim, dim, kernel_size=3, padding=4, groups=dim, dilation=4, padding_mode='reflect')
        
        self.strip_pooling = StripPooling(dim)
          
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 5, dim, 1),  
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        identity = x  
             
        x = self.norm1(x)
        x = self.conv1x1(x)
        x = self.conv5x5(x)
        
        branch1 = self.branch_conv1x1(x)
                
        branch2_input = self.dw_conv_rate1(x)
        
        branch2 = branch2_input + branch1  
                
        branch3_input = self.dw_conv_rate2(x)
     
        branch3 = branch3_input + branch1 + branch2_input  
        
      
        branch4_input = self.dw_conv_rate4(x)
      
        branch4 = branch4_input + branch1 + branch2_input + branch3_input
               
        branch5 = self.strip_pooling(x)
                
        concat = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)
               
        out = self.mlp(concat)
               
        return identity + out

if __name__ == '__main__':
   
    model = BASCP(dim=32)
    input = torch.randn(8, 32, 64, 64)

    output = model(input)

    print('input_size:', input.size())
    print('output_size:', output.size())
   
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params / 1e6:.2f}M')