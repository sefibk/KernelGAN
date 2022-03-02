import argparse
import torch
from train import *
from Utils.training_types import TrainingTypes

parser = argparse.ArgumentParser()
conf = None

training_dict = {
    TrainingTypes.FIXED: train_fixed,
    TrainingTypes.SERIAL: train_serial,
    TrainingTypes.SEMIE2E: train_semie2e,
    TrainingTypes.E2E: train_e2e
}

# Paths
parser.add_argument('--img_name', default='image1', help='image name for saving purposes')
parser.add_argument('--input_image_path', default=os.path.dirname(__file__) + '/training_data/input.png', help='path to one specific image file')
parser.add_argument('--output_dir_path', default=os.path.dirname(__file__) + '/results', help='results path')

# Sizes
parser.add_argument('--input_crop_size', type=int, default=64, help='Generators crop size')
parser.add_argument('--scale_factor', type=float, default=0.5, help='The downscaling scale factor')
parser.add_argument('--X4', action='store_true', help='The wanted SR scale factor')

# Network architecture
parser.add_argument('--G_chan', type=int, default=64, help='# of channels in hidden layer in the G')
parser.add_argument('--D_chan', type=int, default=64, help='# of channels in hidden layer in the D')
parser.add_argument('--G_kernel_size', type=int, default=13, help='The kernel size G is estimating')
parser.add_argument('--D_n_layers', type=int, default=7, help='Discriminators depth')
parser.add_argument('--D_kernel_size', type=int, default=7, help='Discriminators convolution kernels size')
parser.add_argument('--type', type=str, default="fixed", help='Type of training process')

# Iterations
parser.add_argument('--max_iters', type=int, default=3000, help='# of iterations')

# Optimization hyper-parameters
parser.add_argument('--g_lr', type=float, default=2e-4, help='initial learning rate for generator')
parser.add_argument('--d_lr', type=float, default=2e-4, help='initial learning rate for discriminator')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam momentum')

# GPU
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id number')

# Kernel post processing
parser.add_argument('--n_filtering', type=float, default=40, help='Filtering small values of the kernel')

# ZSSR configuration
parser.add_argument('--do_ZSSR', action='store_true', help='when activated - ZSSR is not performed')
parser.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
parser.add_argument('--real_image', action='store_true', help='ZSSRs configuration is for real images')
parser.add_argument('--DL', action='store_true', help='When activated - ZSSR will use an additional discriminator loss.')
parser.add_argument('--use_kernel', action='store_true', help='When activated - ZSSR will use the kernel of Kergan.')
parser.add_argument('--disc_loss_ratio', type=float, default=0.5, help='set the ratio between disc loss and L1 loss in the ZSSR total loss')

def parse(args=None):
    """Parse the configuration"""
    global conf
    conf = parser.parse_args(args=args)
    set_gpu_device()
    set_training_method()
    clean_file_name()
    set_output_directory()
    conf.G_structure = [7, 5, 3, 1, 1, 1]
    print("Scale Factor: %s \tReal Image: %s \t Training type: %s" % (('X4' if conf.X4 else 'X2'), str(conf.real_image), TrainingTypes[conf.type]))

def clean_file_name():
    """Retrieves the clean image file_name for saving purposes"""
    conf.img_name = conf.input_image_path.split(os.sep)[-1].replace('ZSSR', '') \
        .replace('real', '').replace('__', '').split('_.')[0].split('.')[0]

def set_training_method():
    conf.training = training_dict[TrainingTypes[conf.type]]


def set_gpu_device():
    """Sets the GPU device if one is given"""
    if torch.cuda.is_available():
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.gpu_id)
            torch.cuda.set_device(0)
        else:
            torch.cuda.set_device(conf.gpu_id)

def set_output_directory():
    """Define the output directory name and create the folder"""
    conf.output_dir_path = os.path.join(conf.output_dir_path, conf.img_name)
    count = 1
    suffix = ""
    # In case the folder exists - stack 'l's to the folder name
    while os.path.isdir(conf.output_dir_path + suffix):
        suffix = f"({count})"
        count += 1
    conf.output_dir_path += suffix
    os.makedirs(conf.output_dir_path)
