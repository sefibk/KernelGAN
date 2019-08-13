import os
import numpy as np
from scipy.io import loadmat
from ZSSR4KGAN.ZSSR import ZSSR
import matplotlib.pyplot as plt

"""Performs ZSSR over all low-res images downscaled with the random c kernel
in addition - saves .npy files with PSNR"""

def main():
    #################################################
    #      choose the kernel you want to use        #
    #           'tomer' or 'gt' or 'bic             #
    kernel_types = ['gt']
    back_projection_zssr_bool = [False]
    lr_folder = 'lr_x4'
    indices = range(1, 101)
    #################################################
    for kernel_type in kernel_types:
        base_dir = 'training_data/DIV2K'
        kernel_name = 'gt_k_x4/intermediate/kernel_%d.mat' if kernel_type == 'gt' else 'kgan_k_x4/kernel_%d_x2.mat'    # This is '2' since we are gradually growing
        kernel_filename = os.path.join(base_dir, kernel_name)
        if kernel_type == 'bic':
            bicubic_k = np.array([
                                 [0.0001373291015625,   0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375,  0.0004119873046875,  0.0001373291015625],
                                 [0.0004119873046875,   0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125,  0.0012359619140625,  0.0004119873046875],
                                 [-0.0013275146484375, -0.0039825439453130,  0.0128326416015625,  0.0491180419921875,  0.0491180419921875,  0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                                 [-0.0050811767578125, -0.0152435302734375,  0.0491180419921875,  0.1880035400390630,  0.1880035400390630,  0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                                 [-0.0050811767578125, -0.0152435302734375,  0.0491180419921875,  0.1880035400390630,  0.1880035400390630,  0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                                 [-0.0013275146484380, -0.0039825439453125,  0.0128326416015625,  0.0491180419921875,  0.0491180419921875,  0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                                 [0.0004119873046875,   0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125,  0.0012359619140625,  0.0004119873046875],
                                 [0.0001373291015625,   0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375,  0.0004119873046875,  0.0001373291015625]])
        print('\n USING %s KERNEL\n' % kernel_type)
        print('\n Image folder: %s' % lr_folder)
        print('\n Kernel: %s' % (kernel_name % 1))
        input_filename = os.path.join(base_dir, lr_folder, 'img_%d.png')
        output_path = os.path.join(base_dir, 'zssr_x2+x2_w_%s_k' % kernel_type)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        for idx in indices:
            k = [loadmat(kernel_filename % idx).get("Kernel", loadmat(kernel_filename % idx).get("ker"))]
            _, zssr_step1 = ZSSR(input_img_path=input_filename % idx,
                                 output_path=output_path, scale_factor=2, kernels=k).run()
            plt.imsave(output_path + '/img_%d_step_1.png' % idx, zssr_step1)
            _, zssr_step2 = ZSSR(input_img_path=output_path + '/img_%d_step_1.png' % idx,
                                 output_path=output_path, scale_factor=2, kernels=k).run()
            plt.imsave(output_path + '/img_%d_step_2.png' % idx, zssr_step2)
        print('FINISHED, SR images are in %s' % output_path)


if __name__ == '__main__':
    main()

