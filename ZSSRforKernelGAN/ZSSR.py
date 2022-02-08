import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
from Utils.zssr_configs import Config
from Utils.zssr_utils import *
from ZSSRforKernelGAN.ZSSR_network import *

class ZSSR:
    # Basic current state variables initialization / declaration
    kernel = None
    learning_rate = None
    hr_father = None
    lr_son = None
    sr = None
    sf = None
    gt_per_sf = None
    final_sr = None
    hr_fathers_sources = []

    # Output variables initialization / declaration
    reconstruct_output = None
    train_output = None
    output_shape = None

    # Counters and logs initialization
    iter = 0
    base_sf = 1.0
    base_ind = 0
    sf_ind = 0
    mse = []
    mse_rec = []
    interp_rec_mse = []
    interp_mse = []
    mse_steps = []
    loss = []
    learning_rate_change_iter_nums = []
    fig = None

    # Network tensors (all tensors end with _t to distinguish)
    learning_rate_t = None
    lr_son_t = None
    hr_father_t = None
    filters_t = None
    layers_t = None
    net_output_t = None
    loss_t = None
    loss_map_t = None
    train_op = None
    init_op = None

    # Parameters related to plotting and graphics
    plots = None
    loss_plot_space = None
    lr_son_image_space = None
    hr_father_image_space = None
    out_image_space = None

    # A map representing the gradient magnitude of the image at every crop
    prob_map = None
    cropped_loss_map = None
    avg_grad = 1
    loss_map = []
    loss_map_sources = []

    # Tensorflow graph default
    sess = None

    def __init__(self, input_img_path, scale_factor=2, kernel=None, is_real_img=False, noise_scale=1., disc_loss=False):
        # define the writer to log info into TensorBoard
        self.writer = SummaryWriter()
        # Save input image path
        self.input_img_path = input_img_path
        # Acquire meta parameters configuration from configuration class as a class variable
        self.conf = Config(scale_factor, is_real_img, noise_scale, disc_loss)
        # Read input image
        # The first hr father source is the input (source goes through augmentation to become a father)
        self.input = read_im(input_img_path)
        self.gt = None

        # backward support, probably should be deprecated later
        self.sf = np.array(self.conf.scale_factor)

        # set output shape
        self.output_shape = np.uint(np.ceil(np.array(self.input.shape[0:2]) * self.sf))

        # set initial learning rate
        self.learning_rate = self.conf.learning_rate

        # Initialize all counters etc
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.learning_rate_change_iter_nums = [0]

        # Shift kernel to avoid misalignment
        self.kernel = None
        self.set_kernel(kernel)

        # Check if cuda is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Build network computational graph
        self.network = ZSSRNetwork(self.conf).to(self.device)

        # Initialize network weights
        self.weights_initiator = WeightsInitZSSR(self.conf)
        self.network.apply(self.weights_initiator)

        # Create a loss map reflecting the weights per pixel of the image
        # loss maps that correspond to the father sources array
        self.loss_map = create_loss_map(im=self.input) if self.conf.grad_based_loss_map else np.ones_like(self.input)

        # define loss function
        self.criterion = WeightedL1Loss()

        # Optimizers
        self.optimizer_Z = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

        # keep track of number of epchos for ZSSR
        self.epoch_num_Z = -1

        # set to true to stop ZSSR training early
        self.stop_early_Z = False


    def set_kernel(self, kernel):
        if kernel is not None:
            self.kernel = kernel_shift(kernel, self.conf.scale_factor)
        else:
            self.kernel = self.conf.downscale_method

    def set_disc_loss(self, D, DiscLoss):
        self.D = D
        self.DiscLoss = DiscLoss

    def run(self):
        # Run gradually on all scale factors (if only one jump then this loop only happens once)
        print('*' * 60 + '\nSTARTED ZSSR on: \"%s\"...' % self.input_img_path)

        # Train the network
        self.train()

        # Use augmented outputs and back projection to enhance result. Also save the result.
        post_processed_output = self.final_test()

        # Return the final post processed output.
        # noinspection PyUnboundLocalVariable
        return post_processed_output

    def forward_pass(self, lr_son, hr_father_shape=None):
        # Run net on the input to get the output super-resolution (almost final result, only post-processing needed)
        output_img = self.network.forward(lr_son, self.sf, hr_father_shape)
        # Reduce batch dim
        output_img = torch.squeeze(output_img)
        # Channels to last dim
        output_img = torch.permute(output_img, dims=(1, 2, 0))
        # Clip output between 0,1
        output_img = torch.clamp(output_img, min=0, max=1)
        # Convert torch to numpy
        output_img = output_img.detach().cpu().numpy()
        return output_img

    def learning_rate_policy(self):
        # fit linear curve and check slope to determine whether to do nothing, reduce learning rate or finish
        if (not (1 + self.epoch_num_Z) % self.conf.learning_rate_policy_check_every
                and self.epoch_num_Z - self.learning_rate_change_iter_nums[-1] > self.conf.min_iters):
            # noinspection PyTupleAssignmentBalance
            [slope, _], [[var, _], _] = np.polyfit(self.mse_steps[-int(self.conf.learning_rate_slope_range /
                                                                       self.conf.run_test_every):],
                                                   self.mse_rec[-int(self.conf.learning_rate_slope_range /
                                                                     self.conf.run_test_every):],
                                                   1, cov=True)

            # We take the standard deviation as a measure
            std = math.sqrt(var)

            # Determine learning rate maintaining or reduction by the ration between slope and noise
            if -self.conf.learning_rate_change_ratio * slope < std:
                self.learning_rate /= 10

                # Keep track of learning rate changes for plotting purposes
                self.learning_rate_change_iter_nums.append(self.epoch_num_Z)
                return True
        return False

    def quick_test(self):
        # There are four evaluations needed to be calculated:

        # 1. True MSE (only if ground-truth was given), note: this error is before post-processing.
        self.sr = self.forward_pass(self.input)
        if self.gt_per_sf is not None:
            mse = np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - self.sr)))
            self.mse = self.mse + [mse]
            self.writer.add_scalar("True MSE of ground-truth/train", mse, self.epoch_num_Z)
        else:
            self.mse = None

        # 2. Reconstruction MSE, run for reconstruction- try to reconstruct the input from a downscaled version of it
        self.reconstruct_output = self.forward_pass(self.father_to_son(self.input), self.input.shape)
        mse_reconstruct = np.mean(np.ndarray.flatten(np.square(self.input - self.reconstruct_output)))
        self.mse_rec.append(mse_reconstruct)
        self.writer.add_scalar("Reconstruction MSE of low-res/train", mse_reconstruct, self.epoch_num_Z)

        # 3. True MSE of simple interpolation for reference (only if ground-truth was given)
        if self.gt_per_sf is not None:
            interp_sr = imresize(self.input, self.sf, self.output_shape, self.conf.upscale_method)
            mse_interpolation = np.mean(np.ndarray.flatten(np.square(self.gt_per_sf - interp_sr)))
            self.interp_mse = self.interp_mse + [mse_interpolation]
            self.writer.add_scalar("Interpolation MSE of ground-truth/train", mse_interpolation, self.epoch_num_Z)
        else:
            self.interp_mse = None

        # 4. Reconstruction MSE of simple interpolation over downscaled input
        interp_rec = imresize(self.father_to_son(self.input), self.sf, self.input.shape[:], self.conf.upscale_method)
        mse_interpolation_reconstruct = np.mean(np.ndarray.flatten(np.square(self.input - interp_rec)))
        self.interp_rec_mse.append(mse_interpolation_reconstruct)
        self.writer.add_scalar("Interpolation reconstruction MSE of low-res/train", mse_interpolation_reconstruct,
                               self.epoch_num_Z)

        # Track the iters in which tests are made for the graphics x axis
        self.mse_steps.append(self.epoch_num_Z)

    def train(self):
        # main training loop
        for self.iter in tqdm.tqdm(range(self.conf.max_iters), ncols=60):
            self.epoch_Z()
            if self.stop_early_Z:
                break

    def father_to_son(self, hr_father):
        # Create son out of the father by downscaling and if indicated adding noise
        lr_son = imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    def final_test(self):
        # Run over 8 augmentations of input - 4 rotations and mirror (geometric self ensemble)
        # The weird range means we only do it once if output_flip is disabled
        # We need to check if scale factor is symmetric to all dimensions, if not we will do 180 jumps rather than 90

        outputs = []
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
            # Rotate 90*k degrees & mirror flip when k>=4
            test_input = np.rot90(self.input, k) if k < 4 else np.fliplr(np.rot90(self.input, k))

            # Apply network on the rotated input
            tmp_output = self.forward_pass(test_input)

            # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)

            # fix SR output with back projection technique for each augmentation
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(tmp_output, self.input, down_kernel=self.kernel,
                                             up_kernel=self.conf.upscale_method, sf=self.sf)

            # save outputs from all augmentations
            outputs.append(tmp_output)

            # Take the median over all 8 outputs
            almost_final_sr = np.median(outputs, 0)

            # Again back projection for the final fused result
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                almost_final_sr = back_projection(almost_final_sr, self.input, down_kernel=self.kernel,
                                                  up_kernel=self.conf.upscale_method, sf=self.sf)

        # Now we can keep the final result (in grayscale case, colors still need to be added, but we don't care
        # because it is done before saving and for every other purpose we use this result)
        # noinspection PyUnboundLocalVariable
        self.final_sr = almost_final_sr

        # Add colors to result image in case net was activated only on grayscale
        return self.final_sr

    def epoch_Z(self, hr_father, cropped_loss_map, kernel=None):
        # Increment epcho number of ZSSR
        self.epoch_num_Z += 1
        # set the kernel of the for father_to_son
        if kernel:
            self.set_kernel(kernel)
        # Get lr-son from hr-father
        self.lr_son = self.father_to_son(hr_father)
        # run network forward and back propagation, one iteration (This is the heart of the training)
        # Zeroize gradients
        self.optimizer_Z.zero_grad()
        # ZSSR forward pass
        pred = self.network.forward(self.lr_son, self.sf)
        # Convert target to torch
        hr_father = torch.tensor(hr_father).float().to(self.device)
        cropped_loss_map = torch.tensor(cropped_loss_map, requires_grad=False).float().to(self.device)
        # Channels to first dim
        hr_father = torch.permute(hr_father, dims=(2, 0, 1))
        cropped_loss_map = torch.permute(cropped_loss_map, dims=(2, 0, 1))
        # Add batch dimension
        hr_father = torch.unsqueeze(hr_father, dim=0)
        cropped_loss_map = torch.unsqueeze(cropped_loss_map, dim=0)
        # loss (Weighted (cropped_loss_map) L1 loss between label and output layer)
        loss_L1 = self.criterion(pred, hr_father, cropped_loss_map)
        self.writer.add_scalar("L1Loss/train", loss_L1, self.epoch_num_Z)
        if self.conf.disc_loss:
            # Pass ZSSR output through Discriminator
            d_pred_fake = self.D.forward(pred)
            # Calculate the disc loss
            loss_Disc = self.DiscLoss(d_last_layer=d_pred_fake,
                                      is_d_input_real=True,
                                      zssr_shape=True)
            self.writer.add_scalar("DiskLoss/train", loss_Disc, self.epoch_num_Z)
        else:
            # Final loss (Weighted (cropped_loss_map) L1 loss between label and output layer)
            loss_Disc = 0
        # Total loss
        loss = loss_L1 + loss_Disc
        # Initiate backprop
        loss.backward()
        self.optimizer_Z.step()

        """
        # Reduce batch dim
        output_img = torch.squeeze(pred)
        # channels to last dim
        output_img = torch.permute(output_img, dims=(1, 2, 0))
        # Clip output between 0,1
        output_img = torch.clamp(output_img, min=0, max=1)
        # Convert torch to numpy
        output_img = output_img.detach().numpy()
        # need to check why this output is needed
        self.train_output = output_img
        """

        # Test network
        if self.conf.run_test and (not self.epoch_num_Z % self.conf.run_test_every):
            self.quick_test()

        # Consider changing learning rate or stop according to iteration number and losses slope
        if self.learning_rate_policy():
            for param_group in self.optimizer_Z.param_groups:
                param_group['lr'] = self.learning_rate

        # stop when minimum learning rate was passed
        if self.learning_rate < self.conf.min_learning_rate:
            self.stop_early_Z = True
