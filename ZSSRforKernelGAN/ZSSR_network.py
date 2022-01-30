import math

from imresize import imresize
from torch import nn
import torch


class ZSSRNetwork(nn.Module):

    def __init__(self, conf):
        super(ZSSRNetwork, self).__init__()
        # Check if cuda is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Network depth and conv2d parameters
        self.conf = conf
        # need to check if can be added self.sf

        # Stacking conv2d + relu blocks
        blocks = []
        for i in range(self.conf.depth - 1):
            kernel_height, kernel_width, in_channels, out_channels = self.conf.filter_shape[i]
            kernel_size = (kernel_height, kernel_width)
            padding = ((kernel_height - 1) // 2, (kernel_width - 1) // 2)
            blocks += [nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
                       nn.ReLU(True)]
        self.blocks_layer = nn.Sequential(*blocks)
        # build final layer
        kernel_height, kernel_width, in_channels, out_channels = self.conf.filter_shape[-1]
        kernel_size = (kernel_height, kernel_width)
        padding = ((kernel_height - 1) // 2, (kernel_width - 1) // 2)
        self.final_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)

    def forward(self, lr_son, sf, hr_father_shape=None):
        # First gate for the lr-son into the network is interpolation to the size of the father
        # Note: we specify both output_size and scale_factor. best explained by example: say father size is 9 and sf=2,
        # small_son size is 4. if we upscale by sf=2 we get wrong size, if we upscale to size 9 we get wrong sf.
        # The current imresize implementation supports specifying both.
        interpolated_lr_son = imresize(lr_son, sf, hr_father_shape, self.conf.upscale_method)
        # Convert numpy to torch
        interpolated_lr_son = torch.tensor(interpolated_lr_son).float().to(self.device)
        # channel to first dim
        interpolated_lr_son = torch.permute(interpolated_lr_son, dims=(2, 0, 1))
        # Add the batch dimension
        interpolated_lr_son = torch.unsqueeze(interpolated_lr_son, dim=0)
        # throws an error: RuntimeError: expected scalar type Double but found Float
        # verify why
        residual = interpolated_lr_son
        features = self.blocks_layer(interpolated_lr_son)
        # Output image (Add last conv layer result to input, residual learning with global skip connection)
        return self.final_layer(features) + (residual * self.conf.learn_residual)


class WeightsInitZSSR:

    def __init__(self, conf):
        self.conf = conf

    def __call__(self, layer):
        """ initialize weights of ZSSR """
        class_name = layer.__class__.__name__
        if class_name.find('Conv') != -1:
            in_channels = layer.in_channels
            kernel_height, kernel_width = layer.kernel_size
            nn.init.normal_(layer.weight, mean=0.0,
                            std = math.sqrt(self.conf.init_variance / (in_channels*kernel_height*kernel_width)))


class WeightedL1Loss(nn.Module):

    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, target, weights):
        return torch.mean(self.l1(pred, target) * weights)
