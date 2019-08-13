import torch
from torch.autograd import Variable
import loss
import networks
import torch.nn.functional as F


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences
class KernelGAN:
    # noinspection PyUnresolvedReferencesl
    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf

        # Define the GAN
        self.G = networks.Generator(conf).cuda() if conf.G_structure != 1 else networks.LinearGenerator(conf).cuda()
        self.D = networks.Discriminator(conf).cuda()

        # Calculate D's input & output shape according to the shaving done by the networks
        # self.D_input_shape = round(conf.input_crop_size * conf.scale_factor) - self.G.forward_shave
        self.D_input_shape = self.G.output_size
        self.D_output_shape = self.D_input_shape - self.D.forward_shave

        # Input tensors
        self.G_input = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()
        self.D_input = torch.FloatTensor(1, 3, self.D_input_shape, self.D_input_shape).cuda()

        # Output tensors
        self.D_loss_map = torch.FloatTensor(1, 1, self.D_output_shape, self.D_output_shape).cuda()
        self.G_loss_map = torch.FloatTensor(1, 1, self.D_output_shape, self.D_output_shape).cuda()

        # The kernel G is imitating
        self.curr_k = torch.FloatTensor(conf.G_kernel_size, conf.G_kernel_size).cuda()

        # Losses
        self.GAN_loss_layer = loss.GANLoss(d_last_layer_size=[1, 1, self.D_output_shape, self.D_output_shape]).cuda()
        self.bicubic_loss = loss.DownScaleLoss(kernel=conf.bic_kernel, scale_factor=conf.scale_factor).cuda()                                       # G's downscaling should be close to bicubic
        self.sum2one_loss = loss.SumOfWeightsLoss().cuda()                                                                                          # K should sum to 1
        self.edges_loss = loss.SparseEdgesLoss(k_size=conf.G_kernel_size).cuda()                                                                    # Weights close to the edges should be small
        self.centralized_loss = loss.CentralizedLoss(k_size=conf.G_kernel_size, power=conf.centralized_power, scale_factor=conf.scale_factor,
                                                     func=conf.centralized_func).cuda()                                                             # Kernel's COM should be centered
        self.negative_loss = loss.NegativeValuesLoss().cuda()                                                             # Kernel's COM should be centered
        self.sparse_loss = loss.SparsityLoss().cuda()                                                                                               # Kernel should be sparse
        # self.diff_from_GT_loss = loss.DownScaleLoss(kernel=conf.gt_kernel, scale_factor=conf.scale_factor).cuda()                                 # test loss = "Cheating" assuming GT kernel is known

        # Loss coef's
        self.lambda_sum2one = conf.lambda_sum2one
        self.lambda_bicubic = conf.lambda_bicubic
        self.lambda_edges = conf.lambda_edges
        self.lambda_centralized = conf.lambda_centralized
        self.lambda_negative = conf.lambda_negative
        self.lambda_sparse = conf.lambda_sparse

        # Define kernels
        # ground truth kernel
        self.gt_kernel = Variable(torch.Tensor(conf.gt_kernel).cuda(), requires_grad=False)

        # Define loss function
        self.criterionGAN = self.GAN_loss_layer.forward

        # Initialize networks weights
        if conf.init_G_as_delta:
            self.G.apply(networks.init_G_as_delta)
        else:
            self.G.apply(networks.weights_init_G)
        self.D.apply(networks.weights_init_D)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=conf.d_lr, betas=(conf.beta1, 0.999))

    def calc_curr_k(self):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(self.G.parameters()):
            curr_k = F.conv2d(delta, w, padding=self.conf.G_kernel_size - 1) if ind == 0 else F.conv2d(curr_k, w)
        self.curr_k = curr_k.squeeze().flip([0, 1])

    def train(self, g_input, d_input, g_loss_map=None, d_loss_map=None):
        assert g_loss_map is None
        assert d_loss_map is None

        self.set_input(g_input, d_input, g_loss_map, d_loss_map)
        for _ in range(self.conf.G_iters):
            self.train_g()
        for _ in range(self.conf.D_iters):
            self.train_d()

    def set_input(self, g_input, d_input, g_loss_map=None, d_loss_map=None):
        assert g_loss_map is None
        assert d_loss_map is None

        # self.G_input.copy_(g_input)
        # self.D_input.copy_(d_input)

        self.G_input = g_input.contiguous()
        self.D_input = d_input.contiguous()

        if g_loss_map is not None:
            self.G_loss_map.copy_(g_loss_map)
            self.D_loss_map.copy_(d_loss_map)
        else:
            self.G_loss_map, self.D_loss_map = None, None

    def train_g(self):
        # Zeroize gradients
        self.optimizer_G.zero_grad()

        # Calculate K which is equivalent to G
        self.calc_curr_k()

        # Generator forward pass
        self.G_pred = self.G.forward(self.G_input)

        # Run generator result through discriminator forward pass
        d_pred_fake = self.D.forward(self.G_pred)

        # Calculate generator loss, based on discriminator prediction on generator result
        self.loss_G = self.criterionGAN(d_last_layer=d_pred_fake, is_d_input_real=True, grad_map=self.G_loss_map)

        # Loss on the difference of the image from bicubic downscaling
        self.loss_bicubic = self.bicubic_loss.forward(g_input=self.G_input, g_output=self.G_pred)

        # Kernel constraints
        self.loss_edges = self.edges_loss.forward(kernel=self.curr_k)
        self.loss_sum2one = self.sum2one_loss.forward(kernel=self.curr_k)
        self.loss_centralized = self.centralized_loss.forward(kernel=self.curr_k)
        self.loss_sparse = self.sparse_loss.forward(kernel=self.curr_k)
        self.loss_negative = self.negative_loss.forward(kernel=self.curr_k)

        # Sum all losses
        self.total_loss_G = self.loss_G + self.loss_bicubic * self.lambda_bicubic + self.loss_sum2one * self.lambda_sum2one + self.loss_edges * self.lambda_edges + \
                            self.lambda_centralized * self.loss_centralized + self.lambda_sparse * self.loss_sparse + self.lambda_negative * self.loss_negative

        # Calculate gradients
        self.total_loss_G.backward()

        # Update weights
        self.optimizer_G.step()

    def train_d(self):
        # Zeroize gradients
        self.optimizer_D.zero_grad()

        # Discriminator forward pass over real example
        self.d_pred_real = self.D.forward(self.D_input)

        # Discriminator forward pass over fake example (generated by generator)
        # Note that generator result is detached so that gradients are not propagating back through generator
        g_output = self.G.forward(self.G_input)
        self.d_pred_fake = self.D.forward((g_output + torch.randn_like(g_output) / 255.).detach())

        # Calculate discriminator loss
        self.loss_D_fake = self.criterionGAN(self.d_pred_fake, is_d_input_real=False, grad_map=self.G_loss_map)
        self.loss_D_real = self.criterionGAN(self.d_pred_real, is_d_input_real=True, grad_map=self.D_loss_map)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        # Calculate gradients
        # Note that gradients are not propagating back through generator
        self.loss_D.backward()

        # Update weights
        # Note that only discriminator weights are updated (by definition of the D optimizer)
        self.optimizer_D.step()

