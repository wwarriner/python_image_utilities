import unittest
from pathlib import PurePath, Path
from math import ceil

import numpy as np

from image_util import *


class Test(unittest.TestCase):
    def setUp(self):
        self.side_len = np.iinfo(np.uint8).max
        self.base_shape = (self.side_len, self.side_len)

        self.rgb = np.moveaxis(np.indices(self.base_shape), 0, -1).astype(np.uint8)
        self.rgb = np.concatenate(
            (self.rgb, np.zeros(self.base_shape + (1,)).astype(np.uint8)), axis=2
        )
        self.rgb_shape = self.rgb.shape

        self.fov_radius_ratio = 0.45
        self.fov_offset = (-35, 35)
        self.fov_radius = floor(self.side_len * self.fov_radius_ratio)

        self.mask = generate_circular_fov_mask(
            self.base_shape, self.fov_radius, self.fov_offset
        )
        self.mask_shape = self.mask.shape

        self.wait_time = 500

        # ! set to be relatively prime to side_len
        # ! different to check correct reshaping
        self.patch_shape = (12, 13)

        self.test_image_path = PurePath("test") / "test.jpg"
        self.tulips_image_path = PurePath("test") / "tulips.png"

    def tearDown(self):
        pass

    def read_test_image(self):
        return load(str(self.test_image_path))

    def read_tulips_image(self):
        return load(str(self.tulips_image_path))

    def show(self, image, tag):
        show(image, tag)
        cv2.moveWindow(tag, 100, 100)
        cv2.waitKey(self.wait_time)
        cv2.destroyWindow(tag)

    def reduce_contrast(self, image):
        factor = 3.0
        minimum = 50
        return (np.round(image / factor) + 50).astype(np.uint8)

    def generate_image(self):
        image = generate_noise(self.base_shape)
        return self.reduce_contrast(image)

    def read_gray_image(self):
        image = cv2.imread(str(self.test_image_path))
        image = rgb2gray(image)
        return self.reduce_contrast(image)

    def run_fn(self, image, fn, *args, **kwargs):
        out = fn(image, *args, **kwargs)
        vis = np.concatenate((image, out), axis=0)
        tag = "test: {}".format(fn.__name__)
        self.show(vis, tag)

    def standardize(self, image):
        standardized = standardize(image)
        return self.rescale(standardized)

    def rescale(self, image):
        return rescale(image, out_range=(0, 255)).astype(np.uint8)

    def test_adjust_gamma(self):
        self.run_fn(self.read_gray_image(), adjust_gamma, 2.0)
        self.run_fn(self.generate_image(), adjust_gamma, 2.0)
        # TODO add structured assertions here

    def test_apply_clahe(self):
        self.run_fn(self.read_gray_image(), clahe)
        self.run_fn(self.generate_image(), clahe)
        # TODO add structured assertions here

    def test_montage(self):
        patches, _, _ = patchify(self.rgb, self.patch_shape)
        count = patches.shape[0]
        montage_len = floor(count ** 0.5)
        montage_shape = (montage_len, montage_len)
        # sequential order (baseline)
        m = montage(patches, montage_shape)
        self.show(m, "test: sequential")
        # random order
        m = montage(patches, montage_shape, mode="random")
        self.show(m, "test: random")
        # non-zero start
        start = 5 * count // 13
        m = montage(patches, montage_shape, mode="random", start=start)
        self.show(m, "test: start={}".format(start))
        # with repeats
        m = montage(patches, montage_shape, mode="random", repeat=True, start=start)
        self.show(m, "test: with repeats")
        # auto shape
        m = montage(patches, mode="random", repeat=True, start=start)
        self.show(m, "test: with auto-shape")
        # defined aspect ratio
        m = montage(patches, 2.0, mode="random", repeat=True, start=start)
        self.show(m, "test: with auto-shape")
        # defined aspect ratio
        m = montage(patches, 2.0, mode="random", start=start)
        self.show(m, "test: with auto-shape")

    def test_overlay(self):
        image = self.read_test_image()
        noise = generate_noise(image.shape)[..., np.newaxis]
        color = [0.5, 1.0, 0.2]
        self.show(overlay(image, noise, color, alpha=0.2, beta=0.8), "test: overlay")

    def test_patchify(self):
        counts = np.array(
            [ceil(x / y) for x, y in zip(self.rgb.shape, self.patch_shape)]
        )
        count = counts.prod()
        reqd_padding = counts * np.array(self.patch_shape) - np.array(
            self.rgb.shape[:-1]
        )
        patches, patch_count, padding = patchify(self.rgb, self.patch_shape)

        self.assertEqual(patches.ndim, self.rgb.ndim + 1)
        self.assertEqual(patches.shape[0], count)
        self.assertEqual(patches.shape[1:3], self.patch_shape)
        self.assertEqual(patches.shape[3], self.rgb.shape[2])

        self.assertEqual(len(patch_count), 2)
        self.assertTrue((patch_count == counts.ravel()).all())

        self.assertEqual(len(padding), 2)
        self.assertTrue((padding == reqd_padding.ravel()).all())

    def test_rescale(self):
        self.run_fn(self.read_gray_image(), self.rescale)
        self.run_fn(self.generate_image(), self.rescale)
        # TODO add structured assertions here

    def test_save_load(self):
        try:
            path = PurePath("image_util_test_output.png")
            save(str(path), self.rgb.astype(np.uint8))
            self.show(load(str(path)), "test: save/load")
            cv2.waitKey(self.wait_time)
        finally:
            if Path(path).is_file():
                Path(path).unlink()

    def test_show(self):
        self.show(self.rgb.astype(np.uint8), "test: visualize_rgb (blue and green?)")
        self.show(self.rgb[..., 0], "test: visualize_gray (is gradient?)")
        self.show(
            (self.mask * 255).astype(np.uint8), "test: visualize_gray (is circle?)"
        )
        self.show(rgb2gray(self.read_tulips_image()), "test: visualize_gray (is gray?)")
        self.show(self.read_tulips_image(), "test: visualize_color (is color?)")

    def test_stack(self):
        n = 3
        s = stack(n * (self.rgb,))
        self.assertEqual(s.shape[0], n)
        self.assertEqual(s.shape[1:], self.rgb.shape[0:])
        self.assertIsInstance(s, np.ndarray)

    def test_standardize(self):
        self.run_fn(self.read_gray_image(), self.standardize)
        self.run_fn(self.generate_image(), self.standardize)
        # TODO add structured assertions here

    def test_unpatchify(self):
        input_images = np.stack((self.rgb, self.rgb))
        patches, patch_count, padding = patchify(input_images, self.patch_shape)
        images = unpatchify(patches, patch_count, padding)
        self.assertEqual(images.ndim, self.rgb.ndim + 1)
        self.assertEqual(images.shape, input_images.shape)
        self.assertTrue((input_images == images).all())

    # TODO test_load_folder
    # TODO test_save_images
    # TODO test_mask_images
    # TODO test_get_center
    # TODO test_generate_circular_fov_mask


if __name__ == "__main__":
    unittest.main()
