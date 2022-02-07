import os
import tqdm
import time
import matplotlib.pyplot as plt
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from util import analytic_kernel
from ZSSRforKernelGAN.ZSSR_data_handling import ZSSRDataset


def train(conf):
    """Performs ZSSR with estimated kernel for wanted scale factor"""
    # train KerGAN
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    print('KernelGAN estimation complete!')
    gan.save_kernel()

    # train ZSSR
    start_time = time.time()
    print('~' * 30 + '\nRunning ZSSR X%d ' % (
        4 if gan.conf.X4 else 2) + f"with{'' if gan.conf.use_kernel else 'out'} kernel and with{'' if gan.conf.DL else 'out'} discriminator loss...")
    # check which kernel to use
    if gan.conf.use_kernel:
        # get kernel from KerGAN
        final_kernel = gan.get_kernel()
        if gan.conf.X4:
            gan.ZSSR.set_kernel(analytic_kernel(final_kernel))
        else:
            gan.ZSSR.set_kernel(final_kernel)
    # Set gan loss
    gan.ZSSR.set_disc_loss(gan.D, gan.criterionGAN)
    # set ZSSR dataset
    data_Z = ZSSRDataset(conf.input_image_path, gan.ZSSR.conf)
    # start training loop
    for _ in tqdm.tqdm(range(gan.ZSSR.conf.max_iters), ncols=60):
        crop = data_Z[0]
        gan.ZSSR.epoch_Z(crop['HR'], crop['LM'])
        if gan.ZSSR.stop_early_Z:
            break
        if conf.type == 'serial':
            gan.epoch_D_for_ZSSR(crop['HR'])
    sr = gan.ZSSR.final_test()
    max_val = 255 if sr.dtype == 'uint8' else 1.
    # save output image
    plt.imsave(os.path.join(gan.conf.output_dir_path, 'ZSSR_%s.png' % gan.conf.img_name), sr, vmin=0, vmax=max_val,
               dpi=1)
    runtime = int(time.time() - start_time)
    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)
    print('FINISHED RUN (see --%s-- folder)\n' % gan.conf.output_dir_path + '*' * 60 + '\n\n')
    return os.path.join(gan.conf.output_dir_path, 'ZSSR_%s.png' % gan.conf.img_name)

def train_zssr_only(conf, kernel):
    gan = KernelGAN(conf)
    start_time = time.time()
    print('~' * 30 + '\nRunning ZSSR X%d ' % (
        4 if conf.X4 else 2) + f"with{'' if conf.use_kernel else 'out'} kernel and with{'' if conf.DL else 'out'} discriminator loss...")
    # check which kernel to use
    if conf.use_kernel:
        if conf.X4:
            gan.ZSSR.set_kernel(analytic_kernel(kernel))
        else:
            gan.ZSSR.set_kernel(kernel)
    # Set gan loss
    gan.ZSSR.set_disc_loss(gan.D, gan.criterionGAN)
    # set ZSSR dataset
    data_Z = ZSSRDataset(conf.input_image_path, gan.ZSSR.conf)
    # start training loop
    for _ in tqdm.tqdm(range(gan.ZSSR.conf.max_iters), ncols=60):
        crop = data_Z[0]
        gan.ZSSR.epoch_Z(crop['HR'], crop['LM'])
        if gan.ZSSR.stop_early_Z:
            break
        if conf.type == 'serial':
            gan.epoch_D_for_ZSSR(crop['HR'])
    sr = gan.ZSSR.final_test()
    max_val = 255 if sr.dtype == 'uint8' else 1.
    # save output image
    plt.imsave(os.path.join(gan.conf.output_dir_path, 'ZSSR_%s.png' % gan.conf.img_name), sr, vmin=0, vmax=max_val,
               dpi=1)
    runtime = int(time.time() - start_time)
    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)
    print('FINISHED RUN (see --%s-- folder)\n' % gan.conf.output_dir_path + '*' * 60 + '\n\n')
    return os.path.join(gan.conf.output_dir_path, 'ZSSR_%s.png' % gan.conf.img_name)