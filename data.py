import numpy as np
import torch
from torch.utils.data import Dataset

from imresize import imresize
from util import read_image, create_gradient_map, im2tensor, create_probability_map, nn_interpolation


class DataGenerator(Dataset):
    """
    The data generator loads an image once, calculates it's gradient map on initialization and then outputs a cropped version
    of that image whenever called.
    """
    def __init__(self, conf, gan):
        # Default shapes
        self.g_input_shape = conf.input_crop_size
        self.d_input_shape = gan.G.output_size  # shape entering D downscaled by G
        self.d_output_shape = self.d_input_shape - gan.D.forward_shave

        # Read input image
        self.input_image = read_image(conf.input_image_path) / 255.
        self.shave_edges(scale_factor=conf.scale_factor, real_image=conf.real_image)

        self.in_rows, self.in_cols = self.input_image.shape[0:2]

        # Create prob map for choosing the crop
        self.crop_indices_for_g, self.crop_indices_for_d = self.make_list_of_crop_indices(iterations=conf.max_iters, pick_using_grads=not conf.ignore_grad_in_crop, conf=conf)
        # self.input_tensor = im2tensor(self.input_image)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Whenever called, an independent random crop of the data and the corresponding loss map (when needed) is done.
        both for the Generator's Discriminator's inputs.
        """
        g_in = self.random_crop(for_g=True, idx=idx)
        d_in = self.random_crop(for_g=False, idx=idx)

        return g_in, d_in

    def random_crop(self, for_g, idx):
        """Take a random crop of the image and a corresponding of the loss map if provided.
        In case the input is for D - add noise"""
        size = self.g_input_shape if for_g else self.d_input_shape
        top, left = self.get_top_left(size, for_g, idx)
        crop_im = self.input_image[top:top + size, left:left + size, :]
        if not for_g:       # Add noise to the image for d
            crop_im += np.random.randn(*crop_im.shape) / 255.0
            # crop_im = self.input_tensor[:, :, top:top + size, left:left + size]
            # if not for_g:  # Add noise to the image for d
            #     crop_im += torch.randn_like(crop_im) / 255.0
            # return crop_im  # , map2tensor(shave_a2b(lm[top:top + size, left:left + size], self.d_output_shape))
        return im2tensor(crop_im)  # , map2tensor(shave_a2b(lm[top:top + size, left:left + size], self.d_output_shape))

    def make_list_of_crop_indices(self, iterations, pick_using_grads, conf):
        prob_map_big, prob_map_sml = self.create_prob_maps(loss_map_window=conf.loss_map_window, loss_map_percent=conf.loss_map_percent, scale_factor=conf.scale_factor)
        if pick_using_grads:
            crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations, p=prob_map_sml)
            crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations, p=prob_map_big)
        else:
            crop_indices_for_g = np.random.choice(a=len(prob_map_sml), size=iterations)
            crop_indices_for_d = np.random.choice(a=len(prob_map_big), size=iterations)
        return crop_indices_for_g, crop_indices_for_d

    def create_prob_maps(self, loss_map_window, loss_map_percent, scale_factor):
        # Create loss maps for input image and downscaled one
        loss_map_big = create_gradient_map(self.input_image, window=loss_map_window, percent=loss_map_percent)
        loss_map_sml = create_gradient_map(imresize(im=self.input_image, scale_factor=scale_factor, kernel='cubic'), window=loss_map_window, percent=loss_map_percent)
        # Create corresponding probability maps
        prob_map_big, _ = create_probability_map(loss_map_big, self.d_input_shape)
        prob_map_sml, _ = create_probability_map(nn_interpolation(loss_map_sml, int(1/scale_factor)), self.g_input_shape)
        return prob_map_big, prob_map_sml

    def shave_edges(self, scale_factor, real_image):
        if not real_image:
            # Crop 10 pixels to avoid edge effects in artificial examples
            self.input_image = self.input_image[10:-10, 10:-10, :]
        inverse_sf = int(1/scale_factor)
        self.input_image = self.input_image[:-(self.input_image.shape[0] % inverse_sf), :, :] if self.input_image.shape[0] % inverse_sf > 0 else self.input_image
        self.input_image = self.input_image[:, :-(self.input_image.shape[1] % inverse_sf), :] if self.input_image.shape[1] % inverse_sf > 0 else self.input_image

    def get_top_left(self, size, for_g, idx):
        """Picks the top left corner of the crop according to the flag of pick using gradients"""
        center = self.crop_indices_for_g[idx] if for_g else self.crop_indices_for_d[idx]
        row, col = int(center / self.in_cols), center % self.in_cols
        top, left = min(max(0, row - size // 2), self.in_rows - size), min(max(0, col - size // 2), self.in_cols - size)
        return top - top % 2, left - left % 2   # Choose even indices (to avoid misalignment with the loss map for_g)

