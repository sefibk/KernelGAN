import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from imresize import imresize
from ZSSR4KGAN.zssr_utils import kernel_shift
from util import read_image, back_projection


def main():
    #           'tomer' or 'gt' or 'bic             #
    indices = range(1, 31)
    for nf in [0]:
        print('Noise Factor = %.2f' % nf)
        kernel_type = 'gt'
        print('K: %s' % kernel_type)
        base_dir = 'training_data/DIV2K'
        # input_folder = base_dir + '/Real_image_lr' if kernel_type != 'bic' else base_dir + '/lr_x2_bicubic'
        input_folder = base_dir
        input_filename = os.path.join(input_folder, 'lr_x2', 'img_%d.png')
        output_folder = os.path.join(base_dir, 'Naive_BP_w_%s_k_%.2f_U_noise/' % (kernel_type, nf))
        if not os.path.exists(os.path.dirname(output_folder)):
            os.makedirs(os.path.dirname(output_folder))
        # kernel_filename = os.path.join(base_dir, '%s_k_x2' % kernel_type, 'kernel_%d.mat') if kernel_type == 'gt' else os.path.join(base_dir, '%s_k_x2' % kernel_type, 'kernel_%d_x2.mat')
        kernel_filename = os.path.join(base_dir, '%s_k_x2_%.2f_U_noise' % (kernel_type, nf), 'kernel_%d.mat')
        sf = 2
        bicubic_k = np.array([
            [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
            [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
            [-0.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
            [-0.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
            [-0.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
            [-0.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
            [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
            [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]])
        bicubic_k = kernel_shift(bicubic_k, [sf, sf])
        print('Kernel Filename = %s' % kernel_filename % 1)
        for idx in indices:
            kernel = kernel_shift(loadmat(kernel_filename % idx).get("Kernel", loadmat(kernel_filename % idx).get("ker")), [sf, sf]) if kernel_type != 'bic' else bicubic_k
            for suffix in ['.png', '.jpg', '.tif']:
                if not os.path.isfile(input_filename % idx):
                    input_filename = input_filename.split('.')[0] + suffix
            input_img = read_image(input_filename % idx) / 255.
            print('Image: %s' % (input_filename % idx))
            tmp_sr = imresize(input_img, scale_factor=sf)
            for iteration in range(1, 9):
                tmp_sr = back_projection(y_sr=tmp_sr, y_lr=input_img, down_kernel=kernel, up_kernel='cubic', sf=sf)
            plt.imsave(output_folder + 'img_%d.png' % idx, tmp_sr)


if __name__ == '__main__':
    main()
