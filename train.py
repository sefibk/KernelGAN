import os
from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
from logger import Logger
from ZSSR4KGAN.ZSSR import ZSSR
from util import move2cpu, post_process_k, compute_psnr, AnalyticKernel, do_SR, save_final_kernel
import time
from Guided_BP import GuidedBackPropogation


def train(conf, loop_logger=None):
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
    GuidedBackPropogation(gan.G, data.input_image, conf).optimize_input()
    # do_SR(move2cpu(gan.curr_k), conf)


# def test(iteration, conf, kernel, logger, loop_logger):
#     if conf.skip_zssr or (iteration < conf.max_iters - 1 and (iteration == 0 or iteration % conf.zssr_freq != 0)):
#         return
#     kernel = move2cpu(post_process_k(kernel, conf.sharpening, conf.n_filtering, conf.gaussian))
#     sf = int(1/conf.scale_factor)
#     sf = [[sf, sf], [sf ** 2, sf ** 2]] if conf.analytic_sf else sf
#     kernels = [kernel, AnalyticKernel(kernel)] if conf.analytic_sf else [kernel]
#     print('RUNNING ZSSR ON %s' % conf.input_image_path)
#     no_bp_sr, zssr_sr = ZSSR(input_img_path=conf.input_image_path, output_path=conf.output_dir_path,
#                              scale_factor=sf, kernels=kernels,
#                              is_real_img=conf.real_image, noise_scale=conf.noise_scale).run()
#
#     no_bp_mse, no_bp_psnr = compute_psnr(no_bp_sr, conf.gt_image_path, conf.scale_factor)
#     zssr_mse, zssr_psnr = compute_psnr(zssr_sr, conf.gt_image_path, conf.scale_factor)
#     logger.ckp.save_zssr_results(conf, iteration, no_bp_sr, zssr_sr, no_bp_psnr, zssr_psnr)
#
#     # Store performance in a log file for the whole loop
#     if loop_logger is not None:
#         loop_logger.write_results(conf.file_idx, no_bp_psnr, zssr_psnr, no_bp_mse, zssr_mse)


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
