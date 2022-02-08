import os
import tqdm
import time
import matplotlib.pyplot as plt
from Utils import configs
from ZSSRGAN import ZSSRGAN
from Utils.learner import Learner
from Utils.util import analytic_kernel
from Utils.ZSSR_data_handling import ZSSRDataset
from Utils.data import DataGenerator


def train():
    conf = configs.conf
    """Performs ZSSR with estimated kernel for wanted scale factor"""
    # train KerGAN
    gan = ZSSRGAN()
    # Set gan loss
    gan.ZSSR.set_disc_loss(gan.D, gan.criterionGAN)
    # set ZSSR dataset
    data_Z = ZSSRDataset(conf.input_image_path, gan.ZSSR.conf)
    learner = Learner()
    data = DataGenerator(gan)
    start_time = time.time()

    conf.training(conf, gan, learner, data, data_Z)

    sr = gan.ZSSR.final_test()
    max_val = 255 if sr.dtype == 'uint8' else 1.
    # save output image
    plt.imsave(os.path.join(conf.output_dir_path, 'ZSSR_%s.png' % conf.img_name), sr, vmin=0, vmax=max_val,
               dpi=1)
    runtime = int(time.time() - start_time)
    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)
    print('FINISHED RUN (see --%s-- folder)\n' % conf.output_dir_path + '*' * 60 + '\n\n')
    return os.path.join(conf.output_dir_path, 'ZSSR_%s.png' % conf.img_name)

def train_fixed(conf, gan, learner, data, data_Z):
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    print('KernelGAN estimation complete!')
    gan.save_kernel()
    print('~' * 30 + '\nRunning ZSSR X%d ' % (
        4 if conf.X4 else 2) + f"with{'' if conf.use_kernel else 'out'} kernel and with{'' if conf.DL else 'out'} discriminator loss...")

    set_kernel(conf, gan)
    # start training loop
    for _ in tqdm.tqdm(range(gan.ZSSR.conf.max_iters), ncols=60):
        crop = data_Z[0]
        gan.ZSSR.epoch_Z(crop['HR'], crop['LM'])
        if gan.ZSSR.stop_early_Z:
            break


def train_serial(conf, gan, learner, data, data_Z):
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    print('KernelGAN estimation complete!')
    gan.save_kernel()
    print('~' * 30 + '\nRunning ZSSR X%d ' % (
        4 if conf.X4 else 2) + f"with{'' if conf.use_kernel else 'out'} kernel and with{'' if conf.DL else 'out'} discriminator loss...")

    set_kernel(conf, gan)
    # start training loop
    for _ in tqdm.tqdm(range(gan.ZSSR.conf.max_iters), ncols=60):
        crop = data_Z[0]
        gan.ZSSR.epoch_Z(crop['HR'], crop['LM'])
        if gan.ZSSR.stop_early_Z:
            break
        gan.epoch_D_only_ZSSR(crop['HR'])

def train_semie2e(conf, gan, learner, data, data_Z):
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.set_input(g_in, d_in)
        gan.epoch_G()

        set_kernel(conf, gan)
        crop = data_Z[0]
        gan.ZSSR.epoch_Z(crop['HR'], crop['LM'])
        if gan.ZSSR.stop_early_Z:
            break

        loss_d = (gan.calc_loss_d() + gan.calc_loss_d(crop['HR']))/2
        loss_d.backward()
        gan.optimizer_D.step()

        learner.update(iteration, gan)
    print('KernelGAN estimation complete!')
    gan.save_kernel()

def train_e2e(conf, gan, learner, data, data_Z):
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.set_input(g_in, d_in)
        gan.epoch_G()

        set_kernel(conf, gan)
        crop = data_Z[0]
        gan.ZSSR.epoch_Z(crop['HR'], crop['LM'])
        if gan.ZSSR.stop_early_Z:
            break

        loss_d = gan.calc_loss_d(crop['HR'])
        loss_d.backward()
        gan.optimizer_D.step()

        learner.update(iteration, gan)
    print('KernelGAN estimation complete!')
    gan.save_kernel()

def train_zssr_only(kernel):
    conf = configs.conf
    gan = ZSSRGAN()
    start_time = time.time()
    print('~' * 30 + '\nRunning ZSSR X%d ' % (
        4 if conf.X4 else 2) + f"with{'' if conf.use_kernel else 'out'} kernel and with{'' if conf.DL else 'out'} discriminator loss...")

    set_kernel(conf, gan, kernel=kernel)
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
    plt.imsave(os.path.join(conf.output_dir_path, 'ZSSR_%s.png' % conf.img_name), sr, vmin=0, vmax=max_val,
               dpi=1)
    runtime = int(time.time() - start_time)
    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)
    print('FINISHED RUN (see --%s-- folder)\n' % conf.output_dir_path + '*' * 60 + '\n\n')
    return os.path.join(conf.output_dir_path, 'ZSSR_%s.png' % conf.img_name)

def set_kernel(conf, gan, kernel=None):
    if conf.use_kernel:
        # get kernel from KerGAN
        kernel = kernel or gan.get_kernel()
        if conf.X4:
            gan.ZSSR.set_kernel(analytic_kernel(kernel))
        else:
            gan.ZSSR.set_kernel(kernel)