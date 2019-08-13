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
    kernel_types = ['bic']
    back_projection_zssr_bool = [False]
    scale = 4
    lr_folder = 'lr_x%d_bicubic' % scale
    indices = range(1, 25)
    #################################################
    for kernel_type in kernel_types:
        for back_projection_zssr in back_projection_zssr_bool:
            base_dir = 'training_data/DIV2K'
            kernel_name = 'kernel_%d.mat' if kernel_type == 'gt' else 'kernel_%d_x%d.mat'
            kernel_filename = os.path.join(base_dir, '%s_k_x%d_bicubic' % (kernel_type, scale), kernel_name)
            # kernel_filename = os.path.join(base_dir, '%s_k_x2_%.2f_U_noise' % (kernel_type, nf), 'kernel_%d.mat') if kernel_type == 'gt' else os.path.join(base_dir, '%s_k_x2' % kernel_type, 'kernel_%d_x2.mat')
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
            # print('kernel filename: %s' % kernel_filename % 1)
            input_filename = os.path.join(base_dir, lr_folder, 'img_%d.png')
            output_path = os.path.join(base_dir, 'zssr_bicubic_w_%s_k_x%d' % (kernel_type, scale))
            # output_path = os.path.join(base_dir, 'zssr_bicubic_w_%s_k_x%d_gradual' % (kernel_type, scale))
            # output_path = os.path.join(base_dir, 'BP_zssr_w_%s_k_%.2f_U_noise' % (kernel_type, nf)) if back_projection_zssr else os.path.join(base_dir, 'classic_zssr_w_%s_k_%.2f_U_noise' % (kernel_type, nf))
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))

            for idx in indices:
                print('ZSSR on image %d\n%s\n with kernel:' % (idx, input_filename % idx))
                # k = [loadmat(kernel_filename % idx).get("Kernel", loadmat(kernel_filename % idx).get("ker"))] if kernel_type != 'bic' else None
                if scale == 4 and kernel_type == 'kgan':
                    k = [loadmat(kernel_filename % (idx, 2)).get("Kernel", loadmat(kernel_filename % (idx, 2)).get("ker")),
                         loadmat(kernel_filename % (idx, 4)).get("Kernel", loadmat(kernel_filename % (idx, 4)).get("ker"))]
                    print('\t', kernel_filename % (idx, 4))
                elif scale == 2 and kernel_type == 'kgan':
                    k = [loadmat(kernel_filename % (idx, 2)).get("Kernel", loadmat(kernel_filename % (idx, 2)).get("ker"))]
                    print('\t', kernel_filename % (idx, 2))
                # Kernel is bicubic
                else:
                    k = None
                    print('\tNone')
                scale = scale if scale == 2 else [[2, 2], [4, 4]]
                # [[1.5, 1.5], [2.0, 2.0], [2.5, 2.5], [3.0, 3.0], [3.5, 3.5], [4, 4]]
                _, zssr_sr = ZSSR(input_img_path=input_filename % idx,
                                  output_path=output_path,
                                  scale_factor=scale, kernels=k,
                                  back_project_input=back_projection_zssr).run()
                plt.imsave(output_path + '/img_%d.png' % idx, zssr_sr)
                # plt.imsave(output_path + '/img_%d_no_bp.png' % idx, no_bp_sr)
            print('FINISHED, SR images are in %s' % output_path)


if __name__ == '__main__':
    main()

