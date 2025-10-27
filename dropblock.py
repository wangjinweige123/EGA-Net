import torch
import torch.nn as nn
import torch.nn.functional as F
class DropBlock1D(nn.Module):
    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format="channels_last"):

        super(DropBlock1D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.kernel_size = block_size
        self.stride = 1
        self.padding = block_size // 2

    def _get_gamma(self, feature_dim):

        return ((1.0 - self.keep_prob) / self.block_size) * \
               (feature_dim / (feature_dim - self.block_size + 1.0))

    def _compute_valid_seed_region(self, seq_length, device):

        seq_range = torch.arange(seq_length, dtype=torch.float, device=device)
        half_block_size = self.block_size // 2
        mask_left = seq_range >= half_block_size
        mask_right = seq_range < (seq_length - half_block_size)
        valid_seed_region = (mask_left & mask_right).float()
        
        return valid_seed_region.view(1, seq_length, 1)

    def _compute_drop_mask(self, shape, device):

        seq_length = shape[1]
        gamma = self._get_gamma(seq_length)
        
 
        mask = (torch.rand(shape, device=device) < gamma).float()
        mask = mask * self._compute_valid_seed_region(seq_length, device)

        mask = mask.permute(0, 2, 1)  # [batch, 1, seq_len]
        mask = F.max_pool1d(
            mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        mask = mask.permute(0, 2, 1)  # [batch, seq_len, 1]
        
        return 1.0 - mask

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        

        if self.data_format == 'channels_first':

            x = x.permute(0, 2, 1)
        
        shape = x.shape
        device = x.device

        if self.sync_channels:
            mask = self._compute_drop_mask([shape[0], shape[1], 1], device)
        else:
            mask = self._compute_drop_mask(shape, device)
        

        scale = torch.prod(torch.tensor(shape, dtype=torch.float, device=device)) / (torch.sum(mask) + 1e-8)
        x = x * mask * scale

        if self.data_format == 'channels_first':
            # [batch, seq_len, channels] -> [batch, channels, seq_len]
            x = x.permute(0, 2, 1)
        
        return x


class DropBlock2D(nn.Module):


    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format="channels_last"):

        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = data_format
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)

    def _get_gamma(self, height, width):

        return ((1.0 - self.keep_prob) / (self.block_size ** 2)) * \
               (height * width / ((height - self.block_size + 1.0) * (width - self.block_size + 1.0)))

    def _compute_valid_seed_region(self, height, width, device):

        h_range = torch.arange(height, dtype=torch.float, device=device)
        w_range = torch.arange(width, dtype=torch.float, device=device)

        h_grid, w_grid = torch.meshgrid(h_range, w_range, indexing='ij')
        half_block_size = self.block_size // 2
        mask_top = h_grid >= half_block_size
        mask_left = w_grid >= half_block_size
        mask_bottom = h_grid < (height - half_block_size)
        mask_right = w_grid < (width - half_block_size)
        valid_seed_region = ((mask_top & mask_left) & (mask_bottom & mask_right)).float()
        return valid_seed_region.unsqueeze(0).unsqueeze(-1)

    def _compute_drop_mask(self, shape, device):
        height, width = shape[1], shape[2]
        gamma = self._get_gamma(height, width)
        mask = (torch.rand(shape, device=device) < gamma).float()
        mask = mask * self._compute_valid_seed_region(height, width, device)
        mask = mask.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        mask = F.max_pool2d(
            mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        mask = mask.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        
        return 1.0 - mask

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x

        if self.data_format == 'channels_first':
            x = x.permute(0, 2, 3, 1)
        
        shape = x.shape
        device = x.device
        if self.sync_channels:
            mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1], device)
        else:
            mask = self._compute_drop_mask(shape, device)

        scale = torch.prod(torch.tensor(shape, dtype=torch.float, device=device)) / (torch.sum(mask) + 1e-8)
        x = x * mask * scale
        if self.data_format == 'channels_first':
            x = x.permute(0, 3, 1, 2)
        
        return x