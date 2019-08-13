from imresize import imresize
from util import read_image
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


def add_multiplicative_noise(k, noise_factor):
    # noise = np.random.randn(*k.shape)
    noise = np.random.uniform(low=-1, high=1, size=k.shape)
    noisy_k = k + noise_factor * np.multiply(noise, k)
    return noisy_k / noisy_k.sum()


gt_k_path = os.path.join(os.path.abspath(''), 'training_data', 'DIV2K', 'gt_k_x2', 'kernel_%d.mat')
for nf in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
    gt_output_path = os.path.join(os.path.abspath(''), 'training_data', 'DIV2K', 'gt_k_x2_%.2f_U_noise' % nf)
    if not os.path.isdir(gt_output_path):
        os.makedirs(gt_output_path)
    for idx in range(1, 101):
        print(idx)
        k = loadmat(gt_k_path % idx)['Kernel']
        noisy = add_multiplicative_noise(k, nf)
        sio.savemat(gt_output_path + '/kernel_%d' % idx, {'Kernel': noisy})
        if idx % 20 == 0:
            plt.subplot(1, 2, 1)
            plt.imshow(k, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(noisy, cmap='gray')
            plt.pause(0.5)

    # print(np.mean(np.abs(imresize(hr, scale_factor=.25, kernel=kernel_X4) - lr)))
    # plt.imshow(calculate_k_X4(kernel))
    # plt.pause(1)
    # plt.imsave(im_out_path % idx, lr)

# Kernels that are known to be good:
# [3, 5, 7, 8, 9, 10, 12, 15, 16, 21, 23, 25, 26, 28, 33, 36, 42, 46, 47, 48, 52, 54, 57, 59, 63, 65, 75, 78, 80, 86, 90, 93, 94, 96, 98]
