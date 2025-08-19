import numpy as np
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose


def transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))

    return Compose(transform_list)


class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask



class CenterShift(object):
    """
    Randomly shift the center coordinate of the image.
    This effectively moves the "center" of the image to make spatial variation
    """
    
    def __init__(self, max_shift_fraction=0.2):
        """
        Maximum shift as fraction of image size (0.2 = 20% of image size)
        """
        self.max_shift_fraction = max_shift_fraction
    
    def __call__(self, sample):
        image, mask = sample
        
        # Get image dimensions (H, W, C format expected)
        height, width = image.shape[:2]
        
        # Calculate maximum shift in pixels
        max_shift_x = int(width * self.max_shift_fraction)
        max_shift_y = int(height * self.max_shift_fraction)
        
        # Random shift amounts
        shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
        shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)
        
        # Shift by rolling arrays (pixels shifted off right edge appear on left, bottom -> top etc)
        shifted_image = np.roll(image, shift_x, axis=1)  # Roll along width
        shifted_image = np.roll(shifted_image, shift_y, axis=0)  # Roll along height
        
        shifted_mask = np.roll(mask, shift_x, axis=1)
        shifted_mask = np.roll(shifted_mask, shift_y, axis=0)
        
        return shifted_image, shifted_mask

