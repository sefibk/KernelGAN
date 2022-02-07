import os
import tqdm
import time
from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
import warnings
import matplotlib.pyplot as plt
from util import analytic_kernel
from ZSSRforKernelGAN.ZSSR_data_handling import ZSSRDataset
warnings.filterwarnings("ignore")


def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data_K = DataGenerator(conf, gan)
    for epoch in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data_K.__getitem__(epoch)
        gan.train(g_in, d_in)
        learner.update(epoch, gan)
    gan.save_kernel()


    # train ZSSR as GAN
    start_time = time.time()
    print('*' * 60 + '\nSTARTED ZSSR on: \"%s\"...' % conf.input_image_path)
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
    for epoch in tqdm.tqdm(range(gan.ZSSR.conf.max_iters), ncols=60):
        crop = data_Z[0]
        gan.ZSSR.epoch_Z(crop['HR'], crop['LM'])
        if gan.ZSSR.stop_early_Z:
            break
        gan.epoch_D_for_ZSSR(crop['HR'])
    sr = gan.ZSSR.final_test()
    max_val = 255 if sr.dtype == 'uint8' else 1.
    # save output image
    plt.imsave(os.path.join(gan.conf.output_dir_path, 'ZSSR_%s.png' % gan.conf.img_name), sr, vmin=0, vmax=max_val,
               dpi=1)
    runtime = int(time.time() - start_time)
    print('Completed! runtime=%d:%d\n' % (runtime // 60, runtime % 60) + '~' * 30)
    print('FINISHED RUN (see --%s-- folder)\n' % gan.conf.output_dir_path + '*' * 60 + '\n\n')

def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--DL', action='store_true', help='When activated - ZSSR will use an additional discriminator loss.')
    prog.add_argument('--UK', action='store_true', help='When activated - ZSSR will use the kernel of Kergan.')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        conf = Config().parse(create_params(filename, args))
        train(conf)
    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale)]
    if args.X4:
        params.append('--X4')
    if args.DL:
        params.append('--DL')
    if args.UK:
        params.append('--use_kernel')
    if args.real:
        params.append('--real_image')
    return params


if __name__ == '__main__':
    main()
