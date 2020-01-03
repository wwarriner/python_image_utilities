import unittest

import numpy as np

from image_preprocess import *
from image_util import *


class Test(unittest.TestCase):
    def setUp(self):
        self.side_len = 400
        self.base_shape = (self.side_len, self.side_len)
        self.test_image_path = PurePath("test") / "test.jpg"

    def reduce_contrast(self, image):
        factor = 3.0
        minimum = 50
        return (np.round(image / factor) + 50).astype(np.uint8)

    def generate_image(self):
        image = generate_noise(self.base_shape)
        return self.reduce_contrast(image)

    def read_image(self):
        image = cv2.imread(str(self.test_image_path))
        image = rgb2gray(image)
        return self.reduce_contrast(image)

    def run_fn(self, image, fn, *args, **kwargs):
        out = fn(image, *args, **kwargs)
        vis = np.concatenate((image, out), axis=0)
        tag = "test: {}".format(fn.__name__)
        visualize(vis, tag)
        cv2.moveWindow(tag, 100, 100)
        cv2.waitKey(1200)
        cv2.destroyWindow(tag)

    def standardize(self, image):
        standardized = standardize(image)
        return self.rescale(image)

    def rescale(self, image):
        return rescale(image, out_range=(0, 255)).astype(np.uint8)

    def test_apply_clahe(self):
        self.run_fn(self.read_image(), apply_clahe)
        self.run_fn(self.generate_image(), apply_clahe)

    def test_standardize(self):
        self.run_fn(self.read_image(), self.standardize)
        self.run_fn(self.generate_image(), self.standardize)
        # TODO add structured assertions here

    def test_rescale(self):
        self.run_fn(self.read_image(), self.rescale)
        self.run_fn(self.generate_image(), self.rescale)
        # TODO add structured assertions here

    def test_adjust_gamma(self):
        self.run_fn(self.read_image(), adjust_gamma, 2.0)
        self.run_fn(self.generate_image(), adjust_gamma, 2.0)
        # TODO add structured assertions here

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
