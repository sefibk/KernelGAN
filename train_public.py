import os

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from logger import Logger
from util import save_final_kernel, do_SR


def train(conf):
    gan = KernelGAN(conf)
    data = DataGenerator(conf, gan)
    learner = Learner(conf)
    logger = Logger(conf)

    for iteration, [g_in, d_in] in enumerate(data):
        gan.train(g_in, d_in, iteration)
        logger.log(iteration, gan)
        learner.update(iteration, gan, logger)

    save_final_kernel(gan.curr_k, conf)
    do_SR(gan.curr_k, conf)


def main():
    """The main function - performs kernel estimation for all images in the 'test_images' folder.
    Real images (as opposed to synthetically created) should contain 'real' in the filename
    ZSSR (SRX2) is performed if the filename contains 'ZSSR'
    For SR scale factor 4, filename should contain 'X4' """
    input_folder = 'test_images_X4'

    for filename in os.listdir(os.path.abspath(input_folder)):
        additional_args = [] if 'ZSSR' in filename else ['--skip_zssr']
        additional_args += ['--analytic_sf'] if 'X4' in filename else []
        conf = Config().parse(map(str, ['--input_image_path', os.path.join(input_folder, filename),
                                        '--output_dir_path', 'Results'] + additional_args))
        print('\n\nRunning KernelGAN on image \"%s\"' % conf.input_image_path)
        train(conf)
    print('\n\nFINISHED RUN (kernels are in the results folder)')


if __name__ == '__main__':
    main()
