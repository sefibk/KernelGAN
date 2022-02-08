from torch.utils.data import Dataset
from Utils.zssr_utils import *


class ZSSRDataset(Dataset):

    def __init__(self, image_path, conf):
        super().__init__()
        self.image_path = image_path
        self.conf = conf
        self.im = read_im(image_path)
        self.loss_map = create_loss_map(im=self.im) if self.conf.grad_based_loss_map else np.ones_like(self.im)

    def __len__(self):
        """
        Returns:
          the length of the dataset.
          Our dataset contains a single pair so the length is 1.
        """
        return 1

    def __getitem__(self, idx):
        """
        Args:
          idx (int) - Index of element to fetch. In our case only 1.
        Returns:
          sample (dict) - a dictionary containing two elements:
          Under the key 'HR' a numpy.ndarray representing the high resolution augmented crop.
          Has shape `(height, width, num_channels)`.
          Under the key 'LM' a numpy.ndarray representing loss map (weights) of 'HR' .
          Has shape `(height, width, num_channels)`.
        """

        if idx > 0:
            raise IndexError

        # Use augmentation from original input image to create current father.
        # If other scale factors were applied before, their result is also used (hr_fathers_in)
        # crop_center = choose_center_of_crop(self.prob_map) if self.conf.choose_varying_crop else None
        crop_center = None
        hr_father, cropped_loss_map = \
            random_augment(ims=[self.im],
                           base_scales=[1.0] + [self.conf.scale_factor],
                           leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                           no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                           min_scale=self.conf.augment_min_scale,
                           max_scale=([1.0] + [self.conf.scale_factor])[0],
                           allow_rotation=self.conf.augment_allow_rotation,
                           scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                           shear_sigma=self.conf.augment_shear_sigma,
                           crop_size=self.conf.crop_size,
                           allow_scale_in_no_interp=self.conf.allow_scale_in_no_interp,
                           crop_center=crop_center,
                           loss_map_sources=[self.loss_map])

        return {'HR': hr_father, 'LM': cropped_loss_map}
