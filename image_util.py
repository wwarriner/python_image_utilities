from math import floor, ceil, log10
from pathlib import PurePath
from itertools import cycle, islice
from random import shuffle

import numpy as np
import cv2
from PIL import Image

import file_utils

# TODO add pre/postcondition checks


def deinterleave(c):
    a = c[0::2]
    b = c[1::2]
    return a, b


def generate_circular_fov_mask(shape, fov_radius, offset=(0, 0)):
    """Generates a circular field of view mask, with the interior of the circle
    included. The circle is assumed to be centered on the image shape."""

    center = get_center(shape)
    X, Y = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])
    R2 = (X - offset[0]) ** 2 + (Y - offset[1]) ** 2
    fov2 = fov_radius ** 2
    mask = R2 <= fov2
    return mask[..., np.newaxis]


def get_center(shape):
    """Returns the center point of an image shape, rounded down."""

    return [floor(x / 2) for x in shape]


def interleave(a, b):
    c = np.empty((a.size + b.size), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def load(path):
    if PurePath(path).suffix in (".gif"):
        image = Image.open(path)
        image = np.asarray(image)
    else:
        image = cv2.imread(path)
    return image


def load_folder(folder):
    image_files = file_utils.get_contents(folder)
    return [load(str(image_file)) for image_file in image_files]


def mask_images(images, masks):
    masked = images.copy()
    threshold = (masks.max() - masks.min()) / 2.0
    masked[masks <= threshold] = 0
    return masked


def montage(images, shape, mode="sequential", start=0):
    indices = list(range(images.shape[0]))
    if mode == "random":
        shuffle(indices)
    elif mode == "sequential":
        pass
    else:
        assert False
    iterator = cycle(indices)
    stop = np.array(shape).prod() + start
    iterator = islice(iterator, start, stop)

    montage = images[list(iterator)]
    montage = montage.reshape((*shape, *images.shape[1:]))

    a, b = deinterleave(list(range(0, montage.ndim - 1)))
    a, b = list(a), list(b)
    dim_order = (*a, *b, montage.ndim - 1)
    montage = montage.transpose(dim_order)

    image_shape = np.array(shape) * np.array(images.shape[1:-1])
    image_shape = np.append(image_shape, images.shape[-1])
    return montage.reshape(image_shape)


def patchify(images, patch_shape, *args, **kwargs):
    """images has NHWC
    patch_shape has HW"""
    assert images.ndim in (3, 4)
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    out_padding = patch_shape - np.remainder(images.shape[1:-1], patch_shape)
    padding = np.append(out_padding, 0)
    padding = np.insert(padding, 0, 0)
    padding = list(zip((0, 0, 0, 0), padding))
    padded = np.pad(images, padding, *args, **kwargs)

    patch_shape = np.array(patch_shape)
    patch_counts = np.array([x // y for x, y in zip(padded.shape[1:-1], patch_shape)])
    patches_shape = interleave(patch_counts, patch_shape)
    patches_shape = np.append(patches_shape, images.shape[-1])
    patches_shape = np.insert(patches_shape, 0, -1)
    patches = padded.reshape(patches_shape)

    dim_order = deinterleave(range(1, patches.ndim - 1))
    dim_order = np.append(dim_order, patches.ndim - 1)
    dim_order = np.insert(dim_order, 0, 0)
    patches = patches.transpose(dim_order)

    stacked_shape = (-1, *patch_shape, images.shape[-1])
    patches = patches.reshape(stacked_shape)

    return patches, patch_counts, out_padding


def save(path, image):
    cv2.imwrite(path, image)


def save_images(path, name, images):
    """name includes extension, i.e. 'out.png'. Note that files are saved with
    an appropriately formatted number, i.e. if name is 'out.png', 50 images will
    be saved as 'out_01.png' through 'out_50.png'. One image will be saved as
    'out.png'. images must be NHWC format, or a single HWC image."""

    assert images.ndim in (3, 4)
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    ext = PurePath(name).suffix
    base = PurePath(name).stem
    count = images.shape[0]
    if count <= 1:
        digits = 0
    else:
        digits = ceil(log10(count - 1)) - 1

    form = "{base:s}_{number:0{digits}d}{ext:s}"
    if count == 1:
        save(str(PurePath(path) / name), images[0])
    else:
        for i, image in enumerate(images):
            full_name = form.format(base=base, number=i, digits=digits, ext=ext)
            image_path = PurePath(path) / full_name
            save(str(image_path), image)


def stack(images):
    if type(images) is np.ndarray:
        images = (images,)
    images = [image[..., np.newaxis] if image.ndim == 2 else image for image in images]
    return np.stack(images)


def unpatchify(patches, patch_counts, padding):
    """patches has NHWC
    patch_counts has HW"""
    chunk_len = np.array(patch_counts).prod()
    base_shape = np.array(patch_counts) * np.array(patches.shape[1:-1])
    image_shape = np.append(base_shape, patches.shape[-1])
    image_count = patches.shape[0] // chunk_len
    chunk_shape = (*patch_counts, *patches.shape[1:])
    images = []
    for i in range(image_count):
        chunk = patches[i * chunk_len : (i + 1) * chunk_len]
        chunk = np.reshape(chunk, chunk_shape)
        chunk = np.transpose(chunk, (0, 2, 1, 3, 4))
        images.append(np.reshape(chunk, image_shape))
    images = np.stack(images)
    space_shape = base_shape - padding
    slices = [slice(0, x) for x in space_shape]
    slices.append(slice(None))
    slices.insert(0, slice(None))
    images = images[tuple(slices)]
    return images


def visualize(image, tag="UNLABELED_WINDOW", is_opencv=True):
    assert image.ndim in (2, 3)
    if image.ndim == 3:
        assert image.shape[-1] in (1, 3)

    cv2.namedWindow(tag, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(tag, image.shape[0:2][::-1])
    if is_opencv and image.ndim == 3 and image.shape[-1] == 3:
        cv2.imshow(tag, np.flip(image, axis=2))
    else:
        cv2.imshow(tag, image)
    cv2.waitKey(1)
