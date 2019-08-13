import os
import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2
import smtplib, ssl
from imresize import imresize
import scipy.io as sio
from ZSSR4KGAN.ZSSR import ZSSR
import matplotlib.pyplot as plt


def tensor2im(im_t, imtype=np.uint8):
    """Copy's the tensor to the cpu & converts it to imtype in range [0,255]"""
    im_np = np.clip(np.round((np.transpose(move2cpu(im_t).squeeze(0), (1, 2, 0)) + 1) / 2.0 * 255.0), 0, 255)
    return im_np.astype(imtype)


def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def im2tensor(im_np):
    """Copy's an image to the gpu & converts it to float in range [-1,1]"""
    # Converts [0,255] --> [0,1]
    im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    # Converts [0,1] --> [-1,1]
    return torch.FloatTensor(np.transpose(im_np, (2, 0, 1)) * 2.0 - 1.0).unsqueeze(0).cuda()


def map2tensor(gray_map):
    """Moves gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


def resize_tensor_w_kernel(im_t, k, sf=None):
    """Convolves a tensor with a given bicubic kernel according to scale factor"""
    # Expand dimensions to fit convolution: [out_channels, in_channels, k_height, k_width]
    k = k.expand(im_t.shape[1], im_t.shape[1], k.shape[0], k.shape[1])
    # calculate padding
    padding = (k.shape[-1] - 1) // 2
    return F.conv2d(im_t, k, stride=round(1/sf), padding=padding)


def save_tensor(im_t, path):
    """Saves a tensor as image"""
    image_pil = Image.fromarray(tensor2im(im_t), 'RGB')
    image_pil.save(path)


def clean_name(s):
    """Fixes the image name to a cleaner one for the result folder"""
    return s.split('/')[-1].split('.')[0].replace('ZSSR', '').replace('X4', '')


def read_image(path, mode='RGB'):
    """Loads an image"""
    im = Image.open(path)
    if mode is not None:
        im = im.convert(mode)
    im = np.array(im, dtype=np.uint8)
    # im = np.array(im, dtype=np.float32) / 255.0
    return im


def rgb2gray(im):
    """Take a numpy image (gray/color) & change to gray scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def swap_axis(im):
    """Swap axis of a tensor to have a 3 channel image as a 3 batch size image. To fit the generator's input & undo for it's output"""
    return im.transpose(0, 1) if type(im) == torch.Tensor else np.moveaxis(im, 0, 1)


def shave_a2b(a, b):
    """Given a big image or tensor 'a', shave it symmetrically into b's shape"""
    # if dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = (type(a) == torch.Tensor)
    if type(b) == int:
        b = [b, b]
    else:
        b = [b.shape[2], b.shape[3]] if (type(b) == torch.Tensor) else [b.shape[0], b.shape[1]]
    if b[0] > a.shape[0] and b[1] < a.shape[1]:
        return a
    # Calculate how much to crop from each side
    shave_r = a.shape[2] - b[0] if is_tensor else a.shape[0] - b[0]
    shave_c = a.shape[3] - b[1] if is_tensor else a.shape[1] - b[1]
    if len(a.shape) == 2:
        return a[shave_r//2:-shave_r//2, shave_c//2:-shave_c//2]
    return a[:, :, shave_r//2:-shave_r//2, shave_c//2:-shave_c//2] if is_tensor else a[shave_r//2:-shave_r//2, shave_c//2:-shave_c//2, :]


def pad_a2b(a, b):
    """Given a small image or tensor 'a', pad it symmetrically with the minimum value into b's shape"""
    # if dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = (type(a) == torch.Tensor)
    # If b is just the wanted size, create a zero tensor/im of that size
    if type(b) == int:
        b = torch.zeros((1, 1, b, b)) if is_tensor else np.zeros((b, b))
    if b.shape < a.shape:
        return a
    # Calculate how much to pad on each side and if to pad a-symmetrically if one is odd and the other is zero
    pad_r, extra_r = (b.shape[-2] - a.shape[-2]) // 2, (b.shape[-2] - a.shape[-2]) % 2
    pad_c, extra_c = (b.shape[-1] - a.shape[-1]) // 2, (b.shape[-1] - a.shape[-1]) % 2
    # Create the ans tensor or numpy and pad with the minimal value
    # ans = torch.zeros_like(b) + a.min() if is_tensor else np.zeros_like(b) + a.min()
    ans = torch.zeros_like(b) if is_tensor else np.zeros_like(b)
    last_col, last_row = ans.shape[-1], ans.shape[-2]
    if is_tensor:
        ans[:, :, pad_r:last_row-pad_r-extra_r, pad_c:last_col-pad_c-extra_c] = a
    else:
        ans[pad_r:last_row-pad_r-extra_r, pad_c:last_col-pad_c-extra_c] = a
    return ans


def est_k_from_2_imgs(big_im, sml_im, k_size=7, sf=2):
    """ Calculate a size x size kernel by solving least squares of relations between big and small image
    starting from the top left of the image after ignoring padding: A * kernel[:] = b """
    # Load and convert gray-scale
    if type(big_im) is str:
        big_im, sml_im = read_image(big_im), read_image(sml_im)
    big_im, sml_im = rgb2gray(big_im), rgb2gray(sml_im)
    # Calculate the size of edges affected by padding
    edge = (k_size // 2) // sf + (k_size // 2) % sf
    # Init. A s.t. every row is a block convolved with the kernel, and b as the result
    rows, cols = sml_im.shape[0], sml_im.shape[1]
    mat_a, vec_b = np.zeros(shape=((rows - 2 * edge) * (cols - 2 * edge), k_size ** 2)), np.zeros(shape=((rows - 2 * edge) * (cols - 2 * edge), 1))
    for row in range(edge + 1, rows - edge):    # Fill in A and b
        for col in range(edge + 1, cols - edge):
            vec_b[(row-edge) * (col-edge) + col-edge] = sml_im[row, col]
            top, left = sf * row - k_size // 2, sf * col - k_size//2
            mat_a[(row-edge) * (col-edge) + col - edge, :] = np.reshape(big_im[top:top + k_size, left:left + k_size], newshape=(1, -1))

    if np.isfinite(np.linalg.cond(mat_a)):  # If A is invertible - solve the linear equation
        return np.reshape(np.squeeze(np.linalg.lstsq(mat_a, vec_b, rcond=None))[0], newshape=(k_size, k_size))
    else:   # Otherwise return an X shape
        return draw_x(k_size)


def draw_x(size):
    return np.clip(np.identity(size) + np.flip(np.identity(size), axis=1), a_min=0, a_max=1) / (size * 2)


def create_gradient_map(im, window, percent):
    """Given an image, creates a magnitude gradient map, convolves with a rect and clips the values
    to create a weighted loss map emphasizing edges in the image"""
    # Calculate gradient map
    gx, gy = np.gradient(rgb2gray(im))
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    gx, gy = np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / prob_map.flatten().shape[0]
    return prob_vec, None


def gradient_magnitude(im):
    """Gradient magnitude calculation"""
    gx, gy = np.gradient(rgb2gray(im))
    return np.sqrt(gx ** 2 + gy ** 2)


def pad_edges(im, edge):
    """pads an image with zeros"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """zeroize the lower 'percent' in the image and equalize the rest"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min


def post_process_k(k, method, n, sigma):
    """Post process a kernel with different methods, Default settings do nothing """
    sharp_k = sharpen_k(k, method)
    filt_k = filter_n(sharp_k, n)
    return add_gaussian(filt_k, sigma)


def sharpen_k(k, method):
    """Given a kernel, sharpens it according to wanted method"""
    if method is None:
        return k
    elif method == 'softmax':
        return softmax(k)
    elif 'power' in method:
        power = choose_power_to_raise(k) if 'dynamic' in method else int(method[-1])
        return raise_by_p(k, power)


def raise_by_p(z, p):
    """Raises the array to the power of p and normalizes"""
    is_tensor = (type(z) == torch.Tensor)
    if p % 2 == 0:  # z^p-1*|z| / sum(z^p-1*|z|)
        return z ** (p-1) * torch.abs(z) / torch.sum(z ** (p-1) * torch.abs(z)) if is_tensor else z ** (p-1) * np.abs(z) / np.sum(z ** (p-1) * np.abs(z))
    else:   # z^p / sum(z^p)
        return z ** p / torch.sum(z ** p) if is_tensor else z ** p / np.sum(z ** p)


def choose_power_to_raise(k):
    """Chooses the power to raise k that maximizes similarity to a Gaussian kernel"""
    # Create a Gaussian kernel
    target_k = create_gaussian(size=k.shape[0], sigma1=2, is_tensor=type(k) == torch.Tensor)
    # Raise to the powers 1-5 and choose the one minimizing the MSE
    min_mse = 1e6
    for p in range(1, 5):
        if ((raise_by_p(k, p) - target_k) ** 2).mean() < min_mse:
            power = p
    return power


def softmax(z, beta=10.):  # exp^z / sum(exp^z)
    """"Performs softmax on the absolute values of the array and normalizes"""
    is_tensor = (type(z) == torch.Tensor)
    return torch.exp(beta * z) / torch.sum(torch.exp(beta * z)) if is_tensor else np.exp(beta * z) / np.sum(np.exp(beta * z))


def filter_n(k, n):
    """Given a kernel, zeroizes n minimal values and normalizes"""
    if n == 0 or abs(n) >= len(k.flatten()):
        return k
    is_tensor = (type(k) == torch.Tensor)
    if 0 < n <= 1:    # filter values less than n * max
        k_n_min = n * max(k.flatten())
    elif n == -1:
        k_n_min = filter_energy(k, is_tensor)
    else:   # choose n largest
        k_sorted = torch.sort(k.flatten())[0] if is_tensor else np.sort(k.flatten())
        k_n_min = 0.75 * k_sorted[int(n) - 1] if n < 0 else k_sorted[-int(n) - 1]
    filtered_k = torch.clamp(k - k_n_min, min=0, max=100) if is_tensor else np.clip(k - k_n_min, a_min=0, a_max=100)
    return filtered_k / filtered_k.sum()


def filter_energy(k, is_tensor):
    """Filter values that reach to 90% of the kernels energy/support"""
    positive_k = torch.clamp(k, min=0, max=float(k.max())) if is_tensor else np.clip(k, a_min=0, a_max=k.max())
    target_energy = 0.9 * (positive_k * positive_k).sum()
    k_sorted = torch.sort(positive_k.flatten())[0] if is_tensor else np.sort(positive_k.flatten())
    curr_energy = 0
    idx = len(k_sorted) - 1
    while curr_energy < target_energy and idx > 0:
        curr_energy += k_sorted[idx] ** 2
        idx -= 1
    return k_sorted[idx]


def add_gaussian(k, sigma):
    """Given a kernel, adds a gaussian and normalizes"""
    if sigma == 0:
        return k
    is_tensor = (type(k) == torch.Tensor)
    k = k + .5 * create_gaussian(size=k.shape[0], sigma1=sigma, is_tensor=is_tensor)
    return k / k.sum()


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def compute_psnr(imsr, imgt, scale_factor=.5):
    """Compute PSNR according to the Korean Matlab code"""
    # In case a path is given and no GT exists - returns 0
    if type(imgt) == str:
        if not os.path.isfile(imgt):
            return 0, 0
        else:
            imgt = read_image(imgt)
    imsr = (imsr * 255.).astype(np.int) if (imsr.dtype == np.float32 and imsr.max() < 2) else imsr.astype(np.int)
    imgt = (imgt * 255.).astype(np.int) if (imgt.dtype == np.float32 and imgt.max() < 2) else imgt.astype(np.int)
    # Retrieve only luminance channel, crop edges and convert to float 64
    sf = int(1/scale_factor)
    imsr, imgt = np.float64(rgb2ycbcr(imsr))[sf:-sf, sf:-sf, 0], np.float64(rgb2ycbcr(imgt))[sf:-sf, sf:-sf, 0]
    # Compute PSNR
    imdff = (imsr - imgt).flatten()
    rmse = np.sqrt(np.mean(imdff ** 2))
    return rmse, 100 if rmse == 0 else 20 * np.log10(255 / rmse)


def rgb2ycbcr(im_rgb):
    """Taken from stack overflow and compared to Matlab's"""
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0  # to [16/255, 240/255]
    return im_ycbcr


def NNinterpulation(im, sf):
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST), dtype=im.dtype)


def AnalyticKernel(k):
    # OLD CORRECT IN-EFFICIENT CODE
    k_size = k.shape[0]
    big_k = np.zeros((3*k_size - 2, 3*k_size - 2))
    for r in range(k_size):
        for c in range(k_size):
            big_k[2*r:2*r + k_size, 2*c:2*c + k_size] += k[r, c] * k
    # return big_k / big_k.sum()
    # cropped_big_k = big_k[k_size-2:-k_size+2, k_size-2:-k_size+2]
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    return cropped_big_k / cropped_big_k.sum()


def send_email(s='', m=''):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "kernelgan@gmail.com"  # Enter your address
    receiver_email = "sefibk@gmail.com"   # Enter receiver address
    password = 'kernelgan123'

    message = """Subject: %s


    \n
    %s\n
    This message was sent from Python.""" % (s, m)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    print('EMAIL WAS SENT')


def back_project_image(lr, sf=2, down_kernel='cubic', up_kernel='cubic', bp_iters=8):
    """Runs 'bp_iters' iteration of back projection SR technique"""
    tmp_sr = imresize(lr, sf, kernel=up_kernel)
    for _ in range(bp_iters):
        tmp_sr = back_projection(y_sr=tmp_sr, y_lr=lr, down_kernel=down_kernel, up_kernel=up_kernel, sf=sf)
    return tmp_sr


def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None):
    """Projects the error between the downscaled SR image and the LR image"""
    y_sr += imresize(y_lr - imresize(y_sr,
                                     scale_factor=1.0 / sf,
                                     output_shape=y_lr.shape,
                                     kernel=down_kernel),
                     scale_factor=sf,
                     output_shape=y_sr.shape,
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)


def save_final_kernel(k, conf):
    """saves the final kernel and the analytic kernel to the results folder"""
    k_2 = post_process_k(k, method=conf.sharpening, n=conf.n_filtering, sigma=conf.gaussian)
    sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x2.mat' % conf.input_image_path.split('/')[-1]), {'Kernel': k_2})
    if conf.analytic_sf:
        k_4 = AnalyticKernel(k_2)
        sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x4.mat' % conf.input_image_path.split('/')[-1]), {'Kernel': k_4})


def do_SR(k, conf):
    """Performs ZSSR with estimated kernel for wanted scale factor"""
    print(conf.do_SR)
    if conf.do_SR:
        k_2 = post_process_k(k, method=conf.sharpening, n=conf.n_filtering, sigma=conf.gaussian)
        sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x2.mat' % clean_name(conf.input_image_path.split('/')[-1])), {'Kernel': k_2})
        if conf.analytic_sf:
            sio.savemat(os.path.join(conf.output_dir_path, '%s_kernel_x4.mat' % clean_name(conf.input_image_path.split('/')[-1])), {'Kernel': AnalyticKernel(k_2)})
            sr, _ = ZSSR(conf.input_image_path, output_path=conf.output_dir_path, scale_factor=[[2, 2], [4, 4]], kernels=[k_2, AnalyticKernel(k_2)]).run()
        else:
            sr, _ = ZSSR(conf.input_image_path, output_path=conf.output_dir_path, scale_factor=2, kernels=[k_2]).run()
        max_val = 255 if sr.dtype == 'uint8' else 1.
        plt.imsave(os.path.join(conf.output_dir_path, 'ZSSR_%s' % clean_name(conf.input_image_path.split('/')[-1])), sr, vmin=0, vmax=max_val)
