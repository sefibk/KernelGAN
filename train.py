import os
from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from logger import Logger
from util import move2cpu, do_SR, save_final_kernel


def train(conf):
    gan = KernelGAN(conf)
    data = DataGenerator(conf, gan)
    logger = Logger(conf)
    learner = Learner(conf)
    for iteration, [g_in, d_in] in enumerate(data):
        if iteration == conf.max_iters:
            break
        gan.train(g_in, d_in)
        logger.log(iteration, gan)
        learner.update(iteration, gan, logger)
    save_final_kernel(move2cpu(gan.curr_k), conf)
    do_SR(move2cpu(gan.curr_k), conf)


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    input_folder = 'test_images'
    for filename in os.listdir(os.path.abspath(input_folder)):
        flags = get_flags(filename)
        conf = Config().parse(map(str, ['--input_image_path', os.path.join(input_folder, filename),
                                        '--output_dir_path', os.path.join('Results', filename)] + flags))
        print('\n\nRunning KernelGAN on image \"%s\"' % filename)
        train(conf)
    print('\n\nFINISHED RUN (kernels are in the --Results-- folder)')


def get_flags(filename):
    """According to the input file_name - determines the scale factor, whether SR is done and configuration for real images"""
    flags = ['--analytic_sf'] if 'X4' in filename else []   # Estimates the X4 kernel
    flags = flags + ['--do_SR'] if 'ZSSR' in filename else flags    # Performs ZSSR
    flags = flags + ['--real_image'] if 'real' in filename else flags   # Configuration is for real world images (relevant only for ZSSR)
    return flags


if __name__ == '__main__':
    main()
