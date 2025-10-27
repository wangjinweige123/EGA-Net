import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D

class BatchActivate(nn.Module):

    def __init__(self, num_features):
        super(BatchActivate, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=2e-05, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvolutionBlockDropblock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), 
                 padding='same', activation=True, keep_prob=0.9, block_size=7):
        super(ConvolutionBlockDropblock, self).__init__()
  
        if padding == 'same':
            if isinstance(kernel_size, tuple):
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            else:
                padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.activation = activation
        
        if activation:
            self.batch_activate = BatchActivate(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.dropblock(x)
        if self.activation:
            x = self.batch_activate(x)
        return x


class ResidualDropBlock(nn.Module):

    def __init__(self, in_channels, num_filters=16, batch_activate=False, 
                 keep_prob=0.9, block_size=7):
        super(ResidualDropBlock, self).__init__()
        
        self.batch_activate_input = BatchActivate(in_channels)
        self.conv_block1 = ConvolutionBlockDropblock(
            in_channels, num_filters, (3, 3), keep_prob=keep_prob, block_size=block_size)
        self.conv_block2 = ConvolutionBlockDropblock(
            num_filters, num_filters, (3, 3), activation=False, 
            keep_prob=keep_prob, block_size=block_size)
        
        self.skip_connection = None
        if in_channels != num_filters:
            self.skip_connection = nn.Conv2d(in_channels, num_filters, 
                                             kernel_size=1, padding=0)
        
        self.batch_activate = batch_activate
        self.batch_activate_output = BatchActivate(num_filters) if batch_activate else None
    
    def forward(self, x):
        residual = self.batch_activate_input(x)
        residual = self.conv_block1(residual)
        residual = self.conv_block2(residual)
        
        if self.skip_connection is not None:
            x = self.skip_connection(x)
        
        x = x + residual
        
        if self.batch_activate:
            x = self.batch_activate_output(x)
        
        return x


class RSAB(nn.Module):

    def __init__(self, num_filters, block_size=7, keep_prob=0.9):
        super(RSAB, self).__init__()
        
        self.batch_activate = BatchActivate(num_filters)
        self.conv_block1 = ConvolutionBlockDropblock(
            num_filters, num_filters, (3, 3), keep_prob=keep_prob, block_size=block_size)
        self.conv_block2 = ConvolutionBlockDropblock(
            num_filters, num_filters, (3, 3), activation=False, 
            keep_prob=keep_prob, block_size=block_size)
        
        self.batch_activate_output = BatchActivate(num_filters)
    
    def forward(self, x):
        residual = self.batch_activate(x)
        residual = self.conv_block1(residual)
        residual = self.conv_block2(residual)

        x = x + residual
        x = self.batch_activate_output(x)
        
        return x