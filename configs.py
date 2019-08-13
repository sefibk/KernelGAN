import argparse
import torch
import os
from util import est_k_from_2_imgs, draw_x
import numpy as np
import scipy.io
import glob
from time import strftime, localtime
from shutil import copy


# noinspection PyPep8
class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # Paths
        self.parser.add_argument('--input_image_path', default=os.path.dirname(__file__) + '/training_data/input.png', help='path to one specific image file')
        self.parser.add_argument('--gt_kernel_path', default=os.path.dirname(__file__) + '/training_data/kernel.mat', help='path to the true kernel')
        self.parser.add_argument('--output_dir_path', default='/home/sefibe/data/kernelGAN_results', help='Save results to data folder - switch to next row if published')
        self.parser.add_argument('--name', default='KGAN', help='Name of the folder for results')
        self.parser.add_argument('--shot', type=int, default=1, help='The number of the GANs try')

        # Sizes
        self.parser.add_argument('--input_crop_size', type=int, default=64, help='Size of input to the Generator (D gets scale_factor * this size)')
        self.parser.add_argument('--ignore_grad_in_crop', action='store_true', help='chooses crops completely randomly rather than according to the gradient map')
        self.parser.add_argument('--scale_factor', type=float, default=0.5, help='the scale factor the net is trying to imitate')
        self.parser.add_argument('--analytic_sf', action='store_true', help='Computes a X2 kernel and calculates X4 kernel analytically')

        # Network architecture
        self.parser.add_argument('--G_chan', type=int, default=64, help='Number of channels in hidden layer in the Generator')
        self.parser.add_argument('--D_chan', type=int, default=64, help='Number of channels in hidden layer in the Discriminator')
        self.parser.add_argument('--G_kernel_size', type=int, default=13, help='The kernel size G is estimating')
        self.parser.add_argument('--init_G_as_delta', action='store_true', help='Determines if G is initialized as a delta kernel')
        self.parser.add_argument('--D_n_layers', type=int, default=7, help='Discriminators depth')
        self.parser.add_argument('--D_kernel_size', type=int, default=7, help='Discriminators convolution kernels size')

        # Iterations
        self.parser.add_argument('--max_iters', type=int, default=3000, help='# of iterations')
        self.parser.add_argument('--G_iters', type=int, default=1, help='# of sub-iterations for the generator')
        self.parser.add_argument('--D_iters', type=int, default=1, help='# of sub-iterations for the discriminator')

        # Loss map variables
        self.parser.add_argument('--use_loss_map', action='store_true', help='If activated, loss is calculated naively (without gradient)')
        self.parser.add_argument('--loss_map_window', type=int, default=5, help='size of window to blur the gradients in the loss map')
        self.parser.add_argument('--loss_map_percent', type=float, default=.97, help='the part of which below it, gradients are ignored')

        # Loss co-efficients
        self.parser.add_argument('--lambda_sum2one', type=float, default=0.5, help=' regularization constraint on the sum of the kernel values the generator is imitating')
        self.parser.add_argument('--lambda_bicubic', type=float, default=5, help='the amount of weight to be put on the distance from bicubic downscaling')
        self.parser.add_argument('--lambda_bicubic_decay_rate', type=float, default=100., help='The rate of which lambda bicubic decays')
        self.parser.add_argument('--lambda_edges', type=float, default=0.5, help='penalty to the weights on the edges')
        self.parser.add_argument('--lambda_centralized', type=float, default=0, help='Initialization of the weight on the centralized loss')
        self.parser.add_argument('--lambda_centralized_end', type=float, default=1, help='The weight on the centralized loss at the end of optimization')
        self.parser.add_argument('--lambda_sparse', type=float, default=0, help='Initialization of the weight on the sparsity loss')
        self.parser.add_argument('--lambda_sparse_end', type=float, default=5, help='The weight on the sparsity loss at the end of optimization')
        self.parser.add_argument('--lambda_negative', type=float, default=0, help='Initialization of the weight on the negative values in the kernel')
        self.parser.add_argument('--lambda_negative_end', type=float, default=0, help='The weight on the negative values in the kernel at the end of optimization')
        self.parser.add_argument('--lambda_update_freq', type=int, default=200, help='the frequency of which the coefficients are updated')
        self.parser.add_argument('--centralized_power', type=int, default=2, help='the power applied to the kernel')
        self.parser.add_argument('--centralized_func', type=str, default='COM', help='The function calculating the amount of centralization')
        self.parser.add_argument('--bic_loss_to_start_change', type=float, default=0.4, help='bicubic loss value to start changing lambda')

        # Optimization hyper-parameters
        self.parser.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')
        self.parser.add_argument('--d_lr', type=float, default=2e-4, help='initial learning rate for discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')
        self.parser.add_argument('--lr_update_freq', type=int, default=750, help='learning rate update frequency')
        self.parser.add_argument('--lr_update_rate', type=float, default=10.0, help='learning rate update rate')

        # GPU
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

        # Monitoring & logging
        self.parser.add_argument('--file_idx', type=int, default=0, help='file_idx to transfer to zssr')
        self.parser.add_argument('--skip_log', action='store_true', help='Determines whether all logs & plots are saves (place flag if runtime is important)')
        self.parser.add_argument('--log_freq', type=int, default=250, help='frequency of logging')
        self.parser.add_argument('--display_freq', type=int, default=250, help='frequency of showing training results on screen - must be a multiplier of log_freg')

        # Kernel post processing
        self.parser.add_argument('--gaussian', type=float, default=0., help='The sigma of the gaussian added to the kernel; Zero will not add the "noise"')
        self.parser.add_argument('--sharpening', type=str, default='power_1', help='determines the method the kernel is sharpened')
        self.parser.add_argument('--n_filtering', type=float, default=-40, help='number of NNZs that are kept in k; Zero will not filter: -1 will filter 90% of the energy of the kernel')
        self.parser.add_argument('--ensemble_k', action='store_true', help='if true,runs kgan 5 time and chooses the median values of the kernel')

        # Test
        self.parser.add_argument('--zssr_freq', type=int, default=3000, help='how often to run a SR algorithm with the estimated kernel')
        self.parser.add_argument('--do_SR', action='store_true', help='when activated - SR is not performed')

        # other
        self.parser.add_argument('--noise_scale', type=float, default=1., help='The scale of noise added to the image (in gray-scale levels)')
        self.parser.add_argument('--real_image', action='store_true', help='real images get different configurations for SR')
        self.parser.add_argument('--debug', action='store_true', help='sets configs to run quick')
        self.parser.add_argument('--dataset', type=str, default='other', help='What dataset is the image from (BSD/DIV2k/REAL/OTHER)')

    def parse(self, args=None):
        """Create the configuration argument"""
        self.conf = self.parser.parse_args(args=args)
        self.fix_png_tif()
        if self.conf.input_image_path is None:
            print('\n' + '-' * 20, '\nTHERE IS NO IMAGE %d\n' % self.conf.file_idx + '-' * 20 + '\n')
            return self.conf
        print(self.conf)
        self.set_gpu_device()
        self.load_gt_kernel()
        self.prepare_result_dir()
        self.set_downscaling_kernel()
        self.conf.G_structure = [7, 5, 3, 1, 1, 1]
        if self.conf.debug:
            self.conf.max_iters = 500
        # self.conf.do_SR = True
        return self.conf

    def load_gt_kernel(self):
        """stores a ground truth kernel. if no .mat file or gt image is given, stores an X"""
        if os.path.isfile(self.conf.gt_kernel_path):         # If GT kernel is given, use it (only a mat array)
            mat_file = scipy.io.loadmat(self.conf.gt_kernel_path)
            self.conf.gt_kernel = mat_file['Kernel']
        else:        # display an X
            self.conf.gt_kernel = draw_x(self.conf.G_kernel_size)

        # Load Tomer's kernel estimation
        self.conf.tomer_kernel = draw_x(self.conf.G_kernel_size)

    def fix_png_tif(self):
        """if a .tif image is given - fixes the format"""
        raw_input_image_path = self.conf.input_image_path.rsplit('.', 1)[0]
        self.conf.input_image_path = None
        for suf in ['.png', '.tif', '.jpg', '.bmp']:
            if os.path.isfile(raw_input_image_path + suf):
                self.conf.input_image_path = raw_input_image_path + suf

    def prepare_result_dir(self):
        """ Give a proper name to the result folder, create it if doesn't exist & store the code in it (if indicated) """
        if self.conf.shot > 1:
            self.conf.name = self.conf.name + 'RUN=%d' % self.conf.shot
        if not os.path.isdir(os.path.join(self.conf.output_dir_path, 'ZSSR')):
            os.makedirs(os.path.join(self.conf.output_dir_path, 'ZSSR'))
        if not os.path.isdir(os.path.join(self.conf.output_dir_path, 'kernels')):
            os.makedirs(os.path.join(self.conf.output_dir_path, 'kernels'))
        self.conf.output_dir_path += '/' + self.conf.name + strftime('_%b_%d_%H_%M_%S', localtime())
        os.makedirs(self.conf.output_dir_path)
        if not self.conf.skip_log:
            [os.makedirs(self.conf.output_dir_path + '/%s' % x) for x in ['figures', 'ZSSR']]

    def set_gpu_device(self):
        """Sets the GPU device if one is given"""
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(self.conf.gpu_id)

    def set_downscaling_kernel(self):      
        """The kernel used for DownScaleLoss"""
        # A pre-prepared 8x8 bicubic kernel for DownScaleLoss
        self.conf.bic_kernel = [[0.0001373291015625,  0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375,  0.0004119873046875,  0.0001373291015625],
                                [0.0004119873046875,  0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125,  0.0012359619140625,  0.0004119873046875],
                                [-.0013275146484375, -0.0039825439453130,  0.0128326416015625,  0.0491180419921875,  0.0491180419921875,  0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                                [-.0050811767578125, -0.0152435302734375,  0.0491180419921875,  0.1880035400390630,  0.1880035400390630,  0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                                [-.0050811767578125, -0.0152435302734375,  0.0491180419921875,  0.1880035400390630,  0.1880035400390630,  0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                                [-.0013275146484380, -0.0039825439453125,  0.0128326416015625,  0.0491180419921875,  0.0491180419921875,  0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                                [0.0004119873046875,  0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125,  0.0012359619140625,  0.0004119873046875],
                                [0.0001373291015625,  0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375,  0.0004119873046875,  0.0001373291015625]]

