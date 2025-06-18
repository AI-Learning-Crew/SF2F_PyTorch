# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)
        # self.writer = tf.compat.v1.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, images, step, tag="vis"):
        """Log a list of images."""

        img_summaries = []
        if isinstance(images, list):
            for i, img in enumerate(images):
                _tag = '%s/%d' % (tag, i)
                # Create a Summary value
                img_summaries.append(self.add_image(img, _tag))
        elif isinstance(images, dict):
            for key, img in images.items():
                _tag = '%s/%s' % (tag, key)
                # Create a Summary value
                img_summaries.append(self.add_image(img, _tag))
        else:
            raise ValueError("Either a list or a dict is supported.")

        # Create and write Summary
        summary = tf.compat.v1.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def add_image(self, img, tag):
        """Convert numpy array to PIL Image and save as PNG for TensorBoard."""
        # Write the image to a BytesIO stream
        s = BytesIO()
        
        # Convert numpy array to PIL Image and handle different data types
        if img.dtype != np.uint8:
            # Ensure the image is in the range [0, 255] and convert to uint8
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Handle different image formats
        if len(img.shape) == 2:
            # Grayscale (2D) images
            pil_img = Image.fromarray(img, mode='L')
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # RGB images
            pil_img = Image.fromarray(img, mode='RGB')
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA images
            pil_img = Image.fromarray(img, mode='RGBA')
        else:
            # Fallback: try to create image without specifying mode
            pil_img = Image.fromarray(img)
            
        pil_img.save(s, format="PNG")

        # Create an Image object for TensorBoard
        img_sum = tf.compat.v1.Summary.Image(
            encoded_image_string=s.getvalue(),
            height=img.shape[0],
            width=img.shape[1]
        )

        return tf.compat.v1.Summary.Value(tag=tag, image=img_sum)
