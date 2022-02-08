import os
import warnings
import argparse
from Utils import configs
from train import  train
warnings.filterwarnings("ignore")

def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--DL', action='store_true', help='When activated - ZSSR will use an additional discriminator loss.')
    prog.add_argument('--UK', action='store_true', help='When activated - ZSSR will use the kernel of Kergan.')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    prog.add_argument('--type', type=str, default="fixed", help='Type of training process')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        configs.parse(create_params(filename, args))
        train()
    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale),
              '--type', args.type.upper()]
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
