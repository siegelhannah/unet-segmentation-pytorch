from io import BytesIO

import scipy.misc
# import tensorflow as tf

from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.utils import make_grid




class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Logs a scalar value."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        """
        Logs a single image.
        
        Parameters:
          tag (str): Name for the image.
          image (numpy array or torch.Tensor): Must be either HxW (grayscale) or HxWxC.
          step (int): Training step.
        """
        # Convert to numpy array if needed
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        
        # If image is grayscale without channel dimension, add one.
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        # If image has channels in last dimension, convert to CHW format.
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        # Convert to float if needed.
        image = image.astype(np.float32)

        self.writer.add_image(tag, image, step)
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        """
        Logs a list of images, arranging them in a grid.
        
        Parameters:
          tag (str): Name for the image grid.
          images (list): List of images as numpy arrays or torch.Tensors.
          step (int): Training step.
        """
        image_tensors = []
        for img in images:
            # Convert tensor to numpy if needed.
            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()
            if img.ndim == 2:  # grayscale without channel dim
                img = np.expand_dims(img, axis=-1)
            if img.ndim == 3:
                img = np.transpose(img, (2, 0, 1))
            # Convert numpy to torch tensor.
            img_tensor = torch.from_numpy(img.astype(np.float32))
            image_tensors.append(img_tensor)
        
        # Create a grid of images. Adjust nrow if needed.
        grid = make_grid(image_tensors, nrow=len(image_tensors))
        self.writer.add_image(tag, grid, step)
        self.writer.flush()














# class Logger(object):

#     def __init__(self, log_dir):
#         self.writer = SummaryWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()

#     def image_summary(self, tag, image, step):
#         s = BytesIO()
#         scipy.misc.toimage(image).save(s, format="png")

#         # Create an Image object
#         img_sum = tf.Summary.Image(
#             encoded_image_string=s.getvalue(),
#             height=image.shape[0],
#             width=image.shape[1],
#         )

#         # Create and write Summary
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()

#     def image_list_summary(self, tag, images, step):
#         if len(images) == 0:
#             return
#         img_summaries = []
#         for i, img in enumerate(images):
#             s = BytesIO()
#             scipy.misc.toimage(img).save(s, format="png")

#             # Create an Image object
#             img_sum = tf.Summary.Image(
#                 encoded_image_string=s.getvalue(),
#                 height=img.shape[0],
#                 width=img.shape[1],
#             )

#             # Create a Summary value
#             img_summaries.append(
#                 tf.Summary.Value(tag="{}/{}".format(tag, i), image=img_sum)
#             )

#         # Create and write Summary
#         summary = tf.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)
#         self.writer.flush()
