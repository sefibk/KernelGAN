import os
from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from logger import Logger
from util import move2cpu, post_process_k, compute_psnr, AnalyticKernel, do_SR, save_final_kernel
import time


def train(conf):
    gan = KernelGAN(conf)
    data = DataGenerator(conf, gan)
    logger = Logger(conf)
    learner = Learner(conf)
    start_time = time.time()
    for iteration, [g_in, d_in] in enumerate(data):
        if iteration == conf.max_iters:
            break
        gan.train(g_in, d_in)
        logger.log(iteration, gan)
        learner.update(iteration, gan, logger)
    print('FINISHED TRAINING, RUNTIME = %d' % int(time.time() - start_time))
    save_final_kernel(move2cpu(gan.curr_k), conf)
    do_SR(move2cpu(gan.curr_k), conf)


def main():
    """The main function - performs kernel estimation for all images in the 'test_images' folder.
    Real images (as opposed to synthetically created) should contain 'real' in the filename
    ZSSR (SRX2) is performed if the filename contains 'ZSSR'
    For SR scale factor 4, filename should contain 'X4' """
    input_folder = 'test_images'
    for filename in os.listdir(os.path.abspath(input_folder)):
        flags = ['--analytic_sf'] if 'X4' in input_folder else []
        flags = flags + ['--do_SR'] if 'ZSSR' in filename else flags
        flags = flags + ['--real_image'] if 'real' in filename else flags
        conf = Config().parse(map(str, ['--input_image_path', os.path.join(input_folder, filename),
                                        '--output_dir_path', os.path.join('Results', filename)] + flags))
        print('\n\nRunning KernelGAN on image \"%s\"' % filename)
        train(conf)
    print('\n\nFINISHED RUN (kernels are in the results folder)')


if __name__ == '__main__':
    main()
