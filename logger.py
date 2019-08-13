from matplotlib import gridspec
from util import *
import datetime
import matplotlib.pyplot as plt
import torch
import os
from imresize import imresize
import scipy.io as sio
import time
from scipy.ndimage.measurements import center_of_mass


class Logger:
    def __init__(self, conf):
        self.conf = conf
        self.vis = Visualizer(conf)
        self.ckp = Writer(conf)

        self.G_loss = []
        self.D_loss_real = []
        self.D_loss_fake = []
        self.bic_loss = []
        self.sum2one_loss = []
        self.edges_loss = []
        self.centralized_loss = []
        self.negative_loss = []
        self.sparse_loss = []

        self.start_time = time.time()
        self.log_less = 500

    def log(self, iteration, gan):
        """Stores losses into arrays for logging and displaying """
        # If false - skip this
        if self.conf.skip_log:
            return
        # Store all losses values in arrays
        self.G_loss.append(move2cpu(gan.loss_G))
        self.D_loss_real.append(move2cpu(gan.loss_D_real))
        self.D_loss_fake.append(move2cpu(gan.loss_D_fake))
        self.bic_loss.append(move2cpu(gan.loss_bicubic * gan.lambda_bicubic))
        self.sum2one_loss.append(move2cpu(5 * gan.loss_sum2one * gan.lambda_sum2one))
        self.edges_loss.append(move2cpu(10 * gan.loss_edges * gan.lambda_edges))
        self.centralized_loss.append(move2cpu(gan.loss_centralized * gan.lambda_centralized))
        self.negative_loss.append(move2cpu(gan.loss_negative * gan.lambda_negative))
        self.sparse_loss.append(move2cpu(0.1 * gan.loss_sparse * gan.lambda_sparse))

        if iteration % self.conf.log_freq == 0 or iteration % self.conf.display_freq == 0 or iteration == self.conf.max_iters - 1:
            # Move current G's kernel to cpu
            curr_k = move2cpu(gan.curr_k)
            # Log the losses every log_freq

            if iteration % self.conf.log_freq == 0:
                self.ckp.write_log('Iter {} [{} sec]:\n\tD loss real: {:.3f}, \tD loss fake: {:.3f}, \tG loss: {:.3f}'
                                   '\n\tBicubic: {:.3f}, \tsum2one: {:.3f},\tEdges: {:.3f},\tCentralizes: {:.3f}\t Sparse: {:.3f}'
                                   '\n\tKernel sum: {:.2f}, \tValues: {:.2f} - {:.2f}\tNegative: {:.3f}'
                                   .format(iteration, int(time.time() - self.start_time), self.D_loss_real[-1], self.D_loss_fake[-1], self.G_loss[-1],
                                           self.bic_loss[-1], self.sum2one_loss[-1], self.edges_loss[-1], self.centralized_loss[-1], self.sparse_loss[-1],
                                           np.sum(curr_k), np.min(curr_k), np.max(curr_k), self.negative_loss[-1]))

            # Display figure of plots
            if iteration % self.conf.display_freq == 0 or iteration == self.conf.max_iters - 1:
                self.vis.disp(i=iteration, gan=gan, curr_k=curr_k, d_loss_real=self.D_loss_real, d_loss_fake=self.D_loss_fake, bic_loss=self.bic_loss,
                              sum2one_loss=self.sum2one_loss, edges_loss=self.edges_loss, centralized_loss=self.centralized_loss, sparse_loss=self.sparse_loss, negative_loss=self.negative_loss)
        if iteration > self.log_less:
            self.log_less = 10e6
            self.conf.display_freq *= 2

    def important_log(self, log, category=None):
        if not self.conf.skip_log:
            self.ckp.write_important_log(log, category)

    def save_key_imgs(self):
        # commented out saving the no_bp_images
        """Saves the following: input, ground truth, zssr and no bp using: tomer_k, bic_k, gt_k and presenting the ratio to the 2 first ones"""
        if self.conf.skip_log:
            return
        path = self.conf.output_dir_path
        idx = self.conf.file_idx
        # Save input and gt files to result dir
        plt.imsave(path + '/ZSSR/aa_img_%d_input.png' % idx, read_image(self.conf.input_image_path))

    def save_kernel(self, kernel):
        sharper_k = post_process_k(kernel, method=self.conf.sharpening, n=self.conf.n_filtering, sigma=self.conf.gaussian)
        if self.conf.analytic_sf:
            sio.savemat(os.path.join(self.conf.output_dir_path, 'kgan_k_x2_%d.mat' % self.conf.file_idx), {'Kernel': sharper_k})
            sio.savemat(os.path.join(self.conf.output_dir_path, '../kernels/kernel_%d_x2.mat' % self.conf.file_idx), {'Kernel': sharper_k})
            sharper_k = analytic_kernel(sharper_k)
            sio.savemat(self.conf.output_dir_path + '/kgan_k_x4_%d.mat' % self.conf.file_idx, {'Kernel': sharper_k})
            sio.savemat(os.path.join(self.conf.output_dir_path, '../kernels/kernel_%d_x4.mat' % self.conf.file_idx), {'Kernel': sharper_k})
        else:
            sio.savemat(self.conf.output_dir_path + '/kgan_k_x2_%d.mat' % self.conf.file_idx, {'Kernel': sharper_k})
            sio.savemat(os.path.join(self.conf.output_dir_path, '../kernels/kernel_%d_x2.mat' % self.conf.file_idx), {'Kernel': sharper_k})

    def done(self, curr_k):
        self.save_kernel(move2cpu(curr_k))
        self.save_key_imgs()
        self.ckp.done(runtime=int(time.time() - self.start_time))


class Visualizer:
    def __init__(self, conf):
        if conf.skip_log:
            return
        self.conf = conf
        input_im = read_image(conf.input_image_path)
        # Calculate ranges for visuals
        self.norm_k_rng = [-0.06, 0.1] if not conf.analytic_sf else [-0.006, 0.01]
        self.gt_rng = [np.min(conf.gt_kernel), np.max(conf.gt_kernel)]
        self.tomer_rng = [np.min(conf.tomer_kernel), np.max(conf.tomer_kernel)]
        target_k_size = conf.G_kernel_size if (True or not conf.analytic_sf) else 3 * conf.G_kernel_size - 2
        self.k_size_vis = min(int(1.5 * target_k_size), max(conf.gt_kernel.shape[0], conf.tomer_kernel.shape[0], target_k_size))
        self.gt_k_vis = pad_a2b(conf.gt_kernel, self.k_size_vis) if conf.gt_kernel.shape[0] <= self.k_size_vis else shave_a2b(conf.gt_kernel, self.k_size_vis)
        self.tomer_k_vis = pad_a2b(conf.tomer_kernel, self.k_size_vis) if conf.tomer_kernel.shape[0] <= self.k_size_vis else shave_a2b(conf.tomer_kernel, self.k_size_vis)

        ###########################################
        #  _______    __________    ________      #
        # |       |  |          |  |        |     #
        # |0:2,0:3|  | 0:2, 3:6 |  |0:2, 6:8|     #
        # |_______|  |__________|  |________|     #
        #  _______  ___  ___  ___  ___  ___  ___  #
        # |       ||3,2||3,3||3,4||3,5||3,6||3,7| #
        # |3:5,0:2| ___  ___  ___  ___  ___  ___  #
        # |_______||4,2||4,3||4,4||4,5||4,6||4,7| #
        ###########################################

        # Define figures and titles
        self.fig = plt.figure(figsize=(18, 9))
        self.fig.suptitle(conf.name + ' ' + conf.output_dir_path.split('/')[-2])
        gs = gridspec.GridSpec(5, 8)
        self.input_fig = self.fig.add_subplot(gs[0:2, 0:3])
        self.input_fig.set_title('Input image')
        self.input_fig.set_ylabel('size=%dx%d' % (input_im.shape[0], input_im.shape[1]))
        self.loss_fig = self.fig.add_subplot(gs[0:2, 3:6])
        self.loss_fig.set_title('Gan Losses')
        self.plot_loss = self.loss_fig.plot([], [], 'b--', [], [], 'r--')
        self.loss_fig.legend(('D\'s loss (real)', 'D\'s loss (fake)'))
        self.penal_fig = self.fig.add_subplot(gs[0:2, 6:8])
        self.penal_fig.set_title('Penalties and Priors')
        self.plot_penal = self.penal_fig.plot([], [], 'b--', [], [], 'c--', [], [], 'r--', [], [], 'k--', [], [], 'g--', [], [], 'm--')
        self.penal_fig.legend(('Bicubic', '5*Sum2one', '10*Edges', 'Centralized', '0.1*Sparse', '1*Negative'))

        self.g_input_fig = self.fig.add_subplot(gs[3:5, 0:2])
        self.g_input_fig.set_title('G\'s input')
        self.g_input_fig.set_ylabel('crop size=%d' % conf.input_crop_size)
        self.downscaled_gt_fig = self.fig.add_subplot(gs[3, 3])
        self.downscaled_gt_fig.set_title('Input*K_gt')
        self.g_output_fig = self.fig.add_subplot(gs[3, 2])
        self.g_output_fig.set_title('G\'s Output')
        self.d_input_fig = self.fig.add_subplot(gs[4, 2])
        self.d_input_fig.set_xlabel('D\'s Input')
        self.d_input_fig.set_ylabel('crop size=%d' % int(conf.input_crop_size * conf.scale_factor))
        self.diff_G_gt_fig = self.fig.add_subplot(gs[4, 3])
        self.diff_G_gt_fig.set_xlabel('|G\'s output - Input*K_gt|')
        self.k_gt_norm_fig = self.fig.add_subplot(gs[4, 4])
        self.k_gt_norm_fig.set_xlabel('Sum = %.2f' % np.sum(conf.gt_kernel))
        self.k_gt_norm_fig.set_ylabel('size=%d' % conf.gt_kernel.shape[0])
        self.k_gt_fig = self.fig.add_subplot(gs[3, 4])
        self.k_gt_fig.set_title('Ground Truth')
        self.k_gt_fig.set_ylabel('NNZ = %d\nsize=%d' % (np.count_nonzero(conf.gt_kernel), conf.gt_kernel.shape[0]))
        self.k_gt_fig.set_xlabel('rng = [%.3f,%.3f]' % (self.gt_rng[0], self.gt_rng[1]))
        self.k_tomer_norm_fig = self.fig.add_subplot(gs[4, 5])
        self.k_tomer_norm_fig.set_xlabel('Sum = %.2f\n\nNormalized=[%.3f, %.3f]\nG\'s Kernel size=%d\nG\'s architecture=%s' %
                                         (conf.tomer_kernel.sum(), self.norm_k_rng[0], self.norm_k_rng[1], conf.G_kernel_size, str(conf.G_structure)))
        self.k_tomer_fig = self.fig.add_subplot(gs[3, 5])
        self.k_tomer_fig.set_title('Tomer\'s')
        self.k_tomer_fig.set_ylabel('NNZ = %d\nsize=%d' % (np.count_nonzero(conf.tomer_kernel), conf.tomer_kernel.shape[0]))
        self.k_tomer_fig.set_xlabel('rng = [%.3f,%.3f]' % (self.tomer_rng[0], self.tomer_rng[1]))
        self.k_gen_norm_fig = self.fig.add_subplot(gs[4, 6])
        self.k_gen_fig = self.fig.add_subplot(gs[3, 6])
        self.k_gen_fig.set_title('KGAN')
        self.k_sharper_fig = self.fig.add_subplot(gs[3, 7])
        self.k_sharper_fig.set_title('Sharper_n=%s' % str(conf.n_filtering))
        self.k_sharper_norm_fig = self.fig.add_subplot(gs[4, 7])
        self.k_sharper_norm_fig.set_ylabel('size=%d' % conf.G_kernel_size)
        plt.ion()

        # Constant visuals
        self.input_fig.imshow(input_im)
        self.k_tomer_fig.imshow(self.tomer_k_vis, cmap='gray', vmin=self.tomer_rng[0], vmax=self.tomer_rng[1])
        self.k_tomer_norm_fig.imshow(self.tomer_k_vis, cmap='gray', vmin=self.norm_k_rng[0], vmax=self.norm_k_rng[1])
        self.k_gt_fig.imshow(self.gt_k_vis, cmap='gray', vmin=self.gt_rng[0], vmax=self.gt_rng[1])
        self.k_gt_norm_fig.imshow(self.gt_k_vis, cmap='gray', vmin=self.norm_k_rng[0], vmax=self.norm_k_rng[1])

        # Don't display the x and y ticks for the following list of figures
        list_of_figures = [self.input_fig, self.g_input_fig, self.downscaled_gt_fig, self.g_output_fig, self.diff_G_gt_fig, self.d_input_fig, self.k_gt_fig, self.k_gt_norm_fig,
                           self.k_gen_fig, self.k_sharper_fig, self.k_sharper_norm_fig, self.k_gen_norm_fig, self.k_tomer_fig, self.k_tomer_norm_fig]
        _ = [[x.axes.get_xaxis().set_ticks([]), x.axes.get_yaxis().set_ticks([])] for x in list_of_figures]

    def disp(self, i, gan, curr_k, d_loss_real, d_loss_fake, bic_loss, sum2one_loss, edges_loss, centralized_loss, sparse_loss, negative_loss):
        """Creates a figure for learning analysis """

        # Monitor graphs
        self.plot_loss[0].set_data(range(i + 1), d_loss_real)
        self.plot_loss[1].set_data(range(i + 1), d_loss_fake)

        self.loss_fig.set_xlim(0, i + 1)
        self.loss_fig.set_ylim(0, 1)

        self.plot_penal[0].set_data(range(i + 1), bic_loss)
        self.plot_penal[1].set_data(range(i + 1), sum2one_loss)
        self.plot_penal[2].set_data(range(i + 1), edges_loss)
        self.plot_penal[3].set_data(range(i + 1), centralized_loss)
        self.plot_penal[3].set_data(range(i + 1), centralized_loss)
        self.plot_penal[4].set_data(range(i + 1), sparse_loss)
        self.plot_penal[5].set_data(range(i + 1), negative_loss)
        self.penal_fig.set_xlim(0, i + 1)
        self.penal_fig.set_ylim(0, 0.8)
        self.penal_fig.set_xlabel('Lambdas:  \nbicubic=%.0e, sum2one=%.1f, edges=%.0e\ncentralized=%.1f, sparse=%.1f, negative=%.1f'
                                  % (gan.lambda_bicubic, gan.lambda_sum2one, gan.lambda_edges, gan.lambda_centralized, gan.lambda_sparse, gan.lambda_negative))

        # Create objects for visualization
        sharper_k = post_process_k(curr_k, method=self.conf.sharpening, n=self.conf.n_filtering, sigma=self.conf.gaussian)
        sharper_no_gaus_k = post_process_k(curr_k, method=self.conf.sharpening, n=self.conf.n_filtering, sigma=0)
        if i == self.conf.max_iters - 1 and self.conf.analytic_sf:
            sharper_k = analytic_kernel(sharper_k)
            sharper_no_gaus_k = shave_a2b(analytic_kernel(sharper_no_gaus_k), self.k_size_vis)
        generator_output = tensor2im(gan.G.forward(gan.G_input))
        input_downscaled_k_gt = shave_a2b(np.clip(imresize(im=tensor2im(gan.G_input), scale_factor=0.5, kernel=self.conf.gt_kernel), 0, 255), generator_output)
        # input_downscaled_k_gt = generator_output

        # Monitor kernels and images
        self.k_gen_fig.imshow(pad_a2b(curr_k, self.k_size_vis), cmap='gray', vmin=np.min(curr_k), vmax=np.max(curr_k))
        self.k_gen_fig.set_ylabel('NNZ = %d\nsize = %d' % (np.count_nonzero(curr_k), curr_k.shape[0]))
        self.k_gen_fig.set_xlabel('rng = [%.3f,%.3f]' % (np.min(curr_k), np.max(curr_k)))
        self.k_gen_norm_fig.imshow(pad_a2b(curr_k, self.k_size_vis), cmap='gray', vmin=self.norm_k_rng[0], vmax=self.norm_k_rng[1])
        self.k_gen_norm_fig.set_xlabel('Sum = %.2f' % np.sum(curr_k))
        self.k_gen_norm_fig.set_ylabel('C.O.M = [%.1f,%.1f]' % center_of_mass(curr_k))
        self.k_sharper_fig.imshow(pad_a2b(sharper_k, self.k_size_vis), cmap='gray', vmin=np.min(sharper_k), vmax=np.max(sharper_k))
        self.k_sharper_fig.set_ylabel('NNZ = %d\nsize = %d' % (np.count_nonzero(sharper_no_gaus_k), sharper_k.shape[0]))
        self.k_sharper_fig.set_xlabel('rng = [%.3f,%.3f]' % (np.min(sharper_k), np.max(sharper_k)))
        self.k_sharper_norm_fig.set_ylabel('C.O.M = [%.1f,%.1f]' % center_of_mass(sharper_k))
        self.k_sharper_norm_fig.imshow(pad_a2b(sharper_k, self.k_size_vis), cmap='gray', vmin=self.norm_k_rng[0], vmax=self.norm_k_rng[1])
        self.k_sharper_norm_fig.set_xlabel('Sum = %.2f\nGaussian %.1f' % (sum(sharper_k.flatten()), self.conf.gaussian))
        self.g_input_fig.imshow(tensor2im(gan.G_input), vmin=0, vmax=255)
        self.g_output_fig.imshow(generator_output, vmin=0, vmax=255)
        self.g_output_fig.set_ylabel('size = %d' % generator_output.shape[0])
        self.downscaled_gt_fig.imshow(input_downscaled_k_gt)
        self.d_input_fig.imshow(tensor2im(gan.D_input), vmin=0, vmax=255)
        self.diff_G_gt_fig.imshow(rgb2gray(np.abs(generator_output-input_downscaled_k_gt)), vmin=0, vmax=255, cmap='gray')

        # Display and save figure
        self.fig.canvas.draw()
        plt.pause(1e-9)
        plt.savefig(self.conf.output_dir_path + '/figures/fig_%d' % i)

        # For last iteration - save the final figure
        if i == self.conf.max_iters - 1:
            plt.savefig(self.conf.output_dir_path + '/final_figure')
            # Save also in the experiment folder
            if not os.path.isdir(os.path.join(self.conf.output_dir_path, '../figures')):
                os.makedirs(os.path.join(self.conf.output_dir_path, '../figures'))
            plt.savefig(os.path.join(self.conf.output_dir_path, '../figures/figure_%d.png' % self.conf.file_idx))
            # make_video(path=self.conf.output_dir_path + '/figures', video_name='../../fig_videos/figure_vid_%d.mp4' % self.conf.file_idx)
            plt.close()


class Writer:
    def __init__(self, conf):
        self.conf = conf
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.dir = conf.output_dir_path

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(conf):
                if arg in 'gt_kernel_bic_kernel_tomer_kernel' or 'path' in arg:
                    continue
                f.write('{}: {}\n'.format(arg, getattr(conf, arg)))
            f.write('\n')
        self.write_important_log(log='', category='start')

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def write_important_log(self, log, category=None):
        # For some categories there is a pre-defined format
        if category == 'ZSSR':
            log = 'Performed ZSSR on image %d, at iteration %d made %.2f PSNR without B.P and %.2f PSNR with BP' % (log[0], log[1], log[2], log[3])
        elif 'LR_G' == category or 'LR_D' == category:
            log = '%s\'s learning rate update: %.0e --> %.0e on iteration %d' % (category[-1], log[0] * log[1], log[1], log[2])
        elif category == 'start':
            log = 'STARTED RUN ON %s IN FOLDER %s' % (self.conf.input_image_path.split('/')[-1], self.conf.name)
        elif category == 'end':
            log = 'FINISHED KGAN ON IMAGE %s IN %d SECONDS' % (log[0], log[1])

        dashes = '-' * len(log)

        self.write_log((dashes + '\n%s\n' + dashes) % log)

    def save_zssr_results(self, conf, iteration, no_bp_sr, zssr_sr, no_bp_perf, zssr_perf):
        # todo: commented out saving the no_bp
        # no_bp_suf = '.png' if no_bp_perf == 0 else '=%.2f.png' % no_bp_perf
        zssr_suf = '.png' if zssr_perf == 0 else '=%.2f.png' % zssr_perf
        path = conf.output_dir_path + '/ZSSR/%s_kgan_k_%s%s'
        str_iter = str(iteration) + '_' if not iteration == conf.max_iters else ''
        max_val = 255 if zssr_sr.dtype == 'uint8' else 1.
        # if not conf.real_image:
        #     plt.imsave(path % ('no_bp', str_iter, no_bp_suf), no_bp_sr, vmin=0, vmax=max_val)
        plt.imsave(path % ('zssr', str_iter, zssr_suf), zssr_sr, vmin=0, vmax=max_val)
        self.write_important_log(log=[conf.file_idx, iteration, no_bp_perf, zssr_perf], category='ZSSR')

        # for final iteration, save in outer folder without psnr
        if iteration == conf.max_iters - 1:
            path = os.path.join(conf.output_dir_path, '../ZSSR/img_%d_zssr.png')
            # if not conf.real_image:
            #     plt.imsave(path % conf.file_idx, no_bp_sr, vmin=0, vmax=max_val)

            # TODO: fixes weird image sizes being cropped for some reason, may cause a problem with latex
            # plt.imsave(path % conf.file_idx, zssr_sr, vmin=0, vmax=max_val)
            plt.imsave(path % conf.file_idx, zssr_sr, vmin=0, vmax=max_val, dpi=1)

    def done(self, runtime):
        self.write_important_log(log=[self.conf.input_image_path.split('/')[-1], runtime], category='end')
        self.log_file.close()

