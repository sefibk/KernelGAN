import os
from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from util import move2cpu, run_zssr, save_final_kernel


def train(conf):
    gan = KernelGAN(conf)
    data = DataGenerator(conf, gan)
    learner = Learner(conf)
    print('\nRunning KernelGAN on image \"%s\"' % conf.input_image_path)
    for iteration, [g_in, d_in] in enumerate(data):
        if iteration == conf.max_iters:
            break
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    save_final_kernel(move2cpu(gan.curr_k), conf)
    run_zssr(move2cpu(gan.curr_k), conf)
    print('\nFINISHED RUN (see --Results-- folder)')


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    input_folder = 'test_images'
    for filename in os.listdir(os.path.abspath(input_folder)):
        conf = Config().parse(map(str, ['--input_image_path', os.path.join(input_folder, filename)] + get_flags(filename)))
        train(conf)


def get_flags(filename):
    """According to the input file_name - determines the SF, whether ZSSR is done and real images configuration"""
    flags = ['--X4'] if 'X4' in filename else []   # Estimates the X4 kernel
    flags = flags + ['--do_ZSSR'] if 'ZSSR' in filename else flags    # Performs ZSSR
    flags = flags + ['--real_image'] if 'real' in filename else flags   # Configuration is for real world images
    return flags


if __name__ == '__main__':
    main()
