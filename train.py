import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner


def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters)):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
    gan.finish()


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
