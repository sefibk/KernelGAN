import scipy.io as sio
from configs import Config
from train import create_params
from argparse import Namespace
from util import run_zssr
# Parse the command line arguments
args = Namespace(SR=True, X4=False, input_dir='test_images', noise_scale=1.0, output_dir='results', real=False)
filename = 'im_1.png'
conf = Config().parse(create_params(filename, args))
mat_contents = sio.loadmat('C:\\Users\\shaul\\Data_KerGAN\\im_1_kernel_x2.mat')
final_kernel = mat_contents['Kernel']
run_zssr(final_kernel, conf)
