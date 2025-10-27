import torch
import torch.nn as nn
import torch.nn.functional as F
from BASCP import BASCP
from EGA import EGA  


class DropBlock2D(nn.Module):

    def __init__(self, block_size, keep_prob):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def calculate_gamma(self, x):

        return (1 - self.keep_prob) * x.shape[2] * x.shape[3] / (self.block_size ** 2)

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x

        gamma = self.calculate_gamma(x)
        mask = torch.ones_like(x) * gamma
        mask = torch.clamp(mask, 0.0, 1.0)
        mask = torch.bernoulli(mask)
        mask = 1 - mask
        mask = F.max_pool2d(mask, self.kernel_size, self.stride, self.padding)
        mask = 1 - mask

        mask_sum = mask.sum()
        if mask_sum > 0:
            norm_factor = mask_sum / mask.numel()
            norm_factor = torch.max(norm_factor, torch.tensor(1e-6, device=x.device))
            x = x * mask * (1.0 / norm_factor)
        return x


class UNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 start_neurons=64,
                 keep_prob=0.9,
                 block_size=7,
                 use_saspp=True,
                 use_ega=True):  
        super(UNet, self).__init__()
        self.use_saspp = use_saspp
        self.use_ega = use_ega  

        self.conv1_1 = nn.Conv2d(input_channels, start_neurons, kernel_size=3, padding=1)
        self.drop1_1 = DropBlock2D(block_size, keep_prob)
        self.bn1_1 = nn.BatchNorm2d(start_neurons)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(start_neurons, start_neurons, kernel_size=3, padding=1)
        self.drop1_2 = DropBlock2D(block_size, keep_prob)
        self.bn1_2 = nn.BatchNorm2d(start_neurons)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1   = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(start_neurons, start_neurons*2, kernel_size=3, padding=1)
        self.drop2_1 = DropBlock2D(block_size, keep_prob)
        self.bn2_1   = nn.BatchNorm2d(start_neurons*2)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(start_neurons*2, start_neurons*2, kernel_size=3, padding=1)
        self.drop2_2 = DropBlock2D(block_size, keep_prob)
        self.bn2_2   = nn.BatchNorm2d(start_neurons*2)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2   = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(start_neurons*2, start_neurons*4, kernel_size=3, padding=1)
        self.drop3_1 = DropBlock2D(block_size, keep_prob)
        self.bn3_1   = nn.BatchNorm2d(start_neurons*4)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(start_neurons*4, start_neurons*4, kernel_size=3, padding=1)
        self.drop3_2 = DropBlock2D(block_size, keep_prob)
        self.bn3_2   = nn.BatchNorm2d(start_neurons*4)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3   = nn.MaxPool2d(2)

        self.convm_1 = nn.Conv2d(start_neurons*4, start_neurons*8, kernel_size=3, padding=1)
        self.dropm_1 = DropBlock2D(block_size, keep_prob)
        self.bnm_1   = nn.BatchNorm2d(start_neurons*8)
        self.relum_1 = nn.ReLU(inplace=True)

        if self.use_saspp:
            self.saspp = BASCP(start_neurons*8)
        else:
            self.saspp = nn.Identity()
        self.convm_2 = nn.Conv2d(start_neurons*8, start_neurons*8, kernel_size=3, padding=1)
        self.dropm_2 = DropBlock2D(block_size, keep_prob)
        self.bnm_2   = nn.BatchNorm2d(start_neurons*8)
        self.relum_2 = nn.ReLU(inplace=True)

        if self.use_ega:
            self.ega1 = EGA(start_neurons)
            self.ega2 = EGA(start_neurons*2)
            self.ega3 = EGA(start_neurons*4)
        else:
            self.ega1 = nn.Identity()
            self.ega2 = nn.Identity()
            self.ega3 = nn.Identity()

        self.deconv3 = nn.ConvTranspose2d(start_neurons*8, start_neurons*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3_3 = nn.Conv2d(start_neurons*8, start_neurons*4, kernel_size=3, padding=1)
        self.drop3_3 = DropBlock2D(block_size, keep_prob)
        self.bn3_3   = nn.BatchNorm2d(start_neurons*4)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(start_neurons*4, start_neurons*4, kernel_size=3, padding=1)
        self.drop3_4 = DropBlock2D(block_size, keep_prob)
        self.bn3_4   = nn.BatchNorm2d(start_neurons*4)
        self.relu3_4 = nn.ReLU(inplace=True)


        self.deconv2 = nn.ConvTranspose2d(start_neurons*4, start_neurons*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2_3 = nn.Conv2d(start_neurons*4, start_neurons*2, kernel_size=3, padding=1)
        self.drop2_3 = DropBlock2D(block_size, keep_prob)
        self.bn2_3   = nn.BatchNorm2d(start_neurons*2)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.conv2_4 = nn.Conv2d(start_neurons*2, start_neurons*2, kernel_size=3, padding=1)
        self.drop2_4 = DropBlock2D(block_size, keep_prob)
        self.bn2_4   = nn.BatchNorm2d(start_neurons*2)
        self.relu2_4 = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(start_neurons*2, start_neurons, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1_3 = nn.Conv2d(start_neurons*2, start_neurons, kernel_size=3, padding=1)
        self.drop1_3 = DropBlock2D(block_size, keep_prob)
        self.bn1_3   = nn.BatchNorm2d(start_neurons)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv1_4 = nn.Conv2d(start_neurons, start_neurons, kernel_size=3, padding=1)
        self.drop1_4 = DropBlock2D(block_size, keep_prob)
        self.bn1_4   = nn.BatchNorm2d(start_neurons)
        self.relu1_4 = nn.ReLU(inplace=True)

        self.output_conv = nn.Conv2d(start_neurons, 1, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):

        conv1 = self.relu1_1(self.bn1_1(self.drop1_1(self.conv1_1(x))))
        conv1 = self.relu1_2(self.bn1_2(self.drop1_2(self.conv1_2(conv1))))

        conv1_ega = self.ega1(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.relu2_1(self.bn2_1(self.drop2_1(self.conv2_1(pool1))))
        conv2 = self.relu2_2(self.bn2_2(self.drop2_2(self.conv2_2(conv2))))

        conv2_ega = self.ega2(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.relu3_1(self.bn3_1(self.drop3_1(self.conv3_1(pool2))))
        conv3 = self.relu3_2(self.bn3_2(self.drop3_2(self.conv3_2(conv3))))

        conv3_ega = self.ega3(conv3)
        pool3 = self.pool3(conv3)

        convm = self.relum_1(self.bnm_1(self.dropm_1(self.convm_1(pool3))))
        convm = self.saspp(convm)
        convm = self.relum_2(self.bnm_2(self.dropm_2(self.convm_2(convm))))

        deconv3 = self.deconv3(convm)
        uconv3 = torch.cat([deconv3, conv3_ega], dim=1) 
        uconv3 = self.relu3_3(self.bn3_3(self.drop3_3(self.conv3_3(uconv3))))
        uconv3 = self.relu3_4(self.bn3_4(self.drop3_4(self.conv3_4(uconv3))))

        deconv2 = self.deconv2(uconv3)
        uconv2 = torch.cat([deconv2, conv2_ega], dim=1) 
        uconv2 = self.relu2_3(self.bn2_3(self.drop2_3(self.conv2_3(uconv2))))
        uconv2 = self.relu2_4(self.bn2_4(self.drop2_4(self.conv2_4(uconv2))))

        deconv1 = self.deconv1(uconv2)
        uconv1 = torch.cat([deconv1, conv1_ega], dim=1) 
        uconv1 = self.relu1_3(self.bn1_3(self.drop1_3(self.conv1_3(uconv1))))
        uconv1 = self.relu1_4(self.bn1_4(self.drop1_4(self.conv1_4(uconv1))))

        output = self.output_activation(self.output_conv(uconv1))
        return output