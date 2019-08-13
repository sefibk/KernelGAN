import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from util import im2tensor, tensor2im, read_image, shave_a2b
from imresize import imresize


class GuidedBackPropogation:
    def __init__(self, net, target_im, conf):
        self.net = net
        self.conf = conf
        target_im = read_image(conf.input_image_path) / 255.
        self.target_t = im2tensor(target_im).detach().cuda()
        self.initial_guess = imresize(target_im, 2)
        print(self.initial_guess.shape)
        self.input_t = im2tensor(self.initial_guess).detach().requires_grad_().cuda()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam([self.input_t], lr=2e-4, betas=(.5, 0.999))
        self.loss = 0
        self.loss_array = []
        self.output_dir = conf.output_dir_path
        self.vis = Visualizer(conf)

    def train_step(self):
        self.optimizer.zero_grad()
        padding = (self.net.forward_shave, self.net.forward_shave, self.net.forward_shave, self.net.forward_shave)
        input_t = F.pad(self.input_t, padding, 'constant', 0)
        output = self.net.forward(input_t)
        self.loss = self.loss_function(output, self.target_t)
        # print(self.loss)
        self.loss_array.append(self.loss.item())
        self.loss.backward()
        self.optimizer.step()

    def optimize_input(self, num_of_steps=100, freq_of_disp=10):
        for step in range(num_of_steps):
            self.train_step()
            if step % freq_of_disp == 0:
                self.vis.disp(step, self)
            if step % (num_of_steps // 4) == 0:
                for param in self.optimizer.param_groups:
                    param['lr'] /= 5

        # pad = self.net.forward_shave
        # input_t = self.input_t[:, :, pad:-pad, pad:-pad]
        result_im = tensor2im(self.input_t)
        plt.imsave(self.output_dir + '/gbp_img.png', result_im)
        plt.imsave(os.path.join(self.output_dir,  '../ZSSR', self.conf.input_image_path.split('/')[-1]), result_im)
        self.vis.done()


class Visualizer:
    def __init__(self, conf):
        # if conf.skip_log:
        #     return
        self.output_dir_path = conf.output_dir_path
        if not os.path.isdir(os.path.join(conf.output_dir_path, 'figures')):
            os.makedirs(os.path.join(conf.output_dir_path, 'figures'))
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.suptitle('Guided B.P.' + conf.name + ' ' + conf.output_dir_path.split('/')[-2])
        gs = gridspec.GridSpec(2, 4)
        self.init_guess = self.fig.add_subplot(gs[0, 0])
        self.curr_input = self.fig.add_subplot(gs[0, 1])
        self.diff_img = self.fig.add_subplot(gs[0, 2])
        self.loss_fig = self.fig.add_subplot(gs[0, 3])
        self.loss_fig.set_xlim(0, 100 + 1)
        self.loss_fig.set_ylim(0, .001)
        self.plot_loss = self.loss_fig.plot([], [], 'b--')
        self.erase_ticks()
        plt.ion()

    def erase_ticks(self):
        list_of_figures = [self.init_guess, self.curr_input, self.diff_img]
        _ = [[x.axes.get_xaxis().set_ticks([]), x.axes.get_yaxis().set_ticks([])] for x in list_of_figures]

    def disp(self, step, gbp=None):
        # if conf.skip_log:
            # return
        init_guess = gbp.initial_guess
        input_im = tensor2im(gbp.input_t)
        self.init_guess.imshow(init_guess)
        self.curr_input.imshow(input_im)
        self.diff_img.imshow(np.abs(input_im - init_guess), cmap='gray')
        self.plot_loss[0].set_data(range(step + 1), gbp.loss_array)
        self.fig.canvas.draw()
        plt.pause(1e-9)
        plt.savefig(self.output_dir_path + '/figures/gbp_fig_%d' % step)

    def done(self):
        plt.close()

