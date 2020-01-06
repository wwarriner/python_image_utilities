from itertools import chain, cycle, islice
from math import ceil, floor, isinf, isnan, log10
from pathlib import PurePath
from random import shuffle

import cv2
import noise
import numpy as np
from PIL import Image

import file_utils


def adjust_gamma(image, gamma=1.0):
    """Adjusts image gamma. Works on both float and uint8 as expected. Float
    images must be in the range (0.0, 1.0)."""
    igamma = 1.0 / gamma
    if np.issubdtype(image.dtype, np.floating):
        out = image ** igamma
    elif image.dtype == np.uint8:
        lut = np.array([((i / 255.0) ** igamma) * 255.0 for i in range(0, 256)])
        out = cv2.LUT(image, lut.astype(np.uint8))
    else:
        assert False
    return out


def clahe(image):
    """Applies CLAHE equalization to input image."""
    is_float = np.issubdtype(image.dtype, np.floating)
    if is_float:
        image = float_to_uint8(image)

    is_rgb = image.shape[-1] == 3
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image[..., 0] = clahe.apply(image[..., 0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    else:
        image = clahe.apply(image)

    if is_float:
        image = uint8_to_float(image)
    return image


def float_to_uint8(float_image, float_range=(0.0, 1.0), clip=False):
    uint8_image = rescale(float_image, out_range=(0.0, 255.0), in_range=float_range)
    return uint8_image.astype(np.uint8)


def generate_circular_fov_mask(shape, fov_radius, offset=(0, 0)):
    """Generates a circular field of view mask, with the interior of the circle
    included. The circle is assumed to be centered on the image shape."""

    center = get_center(shape)
    X, Y = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])
    R2 = (X - offset[0]) ** 2 + (Y - offset[1]) ** 2
    fov2 = fov_radius ** 2
    mask = R2 <= fov2
    return mask[..., np.newaxis]


def generate_noise(shape, offsets=None):
    if offsets is None:
        scale = 0.1 * np.array(shape).max()
        offsets = np.random.uniform(-1000 * scale, 1000 * scale, 2)

    octaves = 1
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    X = X + offsets[0]
    Y = Y + offsets[1]
    noise_maker = np.vectorize(
        lambda x, y: noise.pnoise2(x / scale, y / scale, octaves=octaves)
    )
    n = noise_maker(X, Y)
    n = ((n - n.min()) / (n.max() - n.min())) * 255
    return n.astype(np.uint8)


def get_center(shape):
    """Returns the center point of an image shape, rounded down."""
    return [floor(x / 2) for x in shape]


def gray2rgb(gray_image):
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)


def lab2rgb(lab_image):
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)


def load(path):
    """Returns image in RGB or grayscale format."""
    if PurePath(path).suffix.casefold() in (".tif", ".tiff"):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = Image.open(path)
        image = np.array(image)
    return image


def load_images(folder, ext):
    image_files = file_utils.get_contents(folder, ext)
    return [load(str(image_file)) for image_file in image_files]


def mask_images(images, masks):
    masked = images.copy()
    threshold = (masks.max() - masks.min()) / 2.0
    masked[masks <= threshold] = 0
    return masked


def montage(images, shape=None, mode="sequential", repeat=False, start=0):
    image_count = images.shape[0] - start
    if shape is None:
        shape = _optimize_shape(image_count)
    elif isinstance(shape, (int, float)):
        shape = _optimize_shape(image_count, width_height_aspect_ratio=shape)

    indices = list(range(images.shape[0]))

    if mode == "random":
        shuffle(images)
    elif mode == "sequential":
        pass
    else:
        assert False

    if repeat:
        iterator = cycle(indices)
    else:
        iterator = chain(indices, cycle([float("inf")]))

    stop = np.array(shape).prod() + start
    iterator = islice(iterator, start, stop)

    montage = np.stack([_get_image_or_blank(images, i) for i in iterator])
    montage = montage.reshape((*shape, *images.shape[1:]))

    a, b = _deinterleave(list(range(0, montage.ndim - 1)))
    a, b = list(a), list(b)
    dim_order = (*a, *b, montage.ndim - 1)
    montage = montage.transpose(dim_order)

    image_shape = np.array(shape) * np.array(images.shape[1:-1])
    image_shape = np.append(image_shape, images.shape[-1])
    return montage.reshape(image_shape)


# TODO fix issue with different channel counts, i.e. gray to rgb for both inputs
def overlay(background, foreground, color, alpha=0.1, beta=1.0, gamma=0.0, clip=False):
    """Applies a color to grayscale foreground and blends with background.
    Background may be RGB or grayscale. Foreground and background may be float
    or uint8. Output is RGB with the same dtype as background."""
    assert np.issubdtype(background.dtype, np.floating) or background.dtype == np.uint8
    assert background.shape[-1] in (1, 3)
    assert np.issubdtype(foreground.dtype, np.floating) or foreground.dtype == np.uint8
    assert foreground.shape[-1] == 1
    assert len(color) == 3

    if background.dtype == np.uint8:
        background = uint8_to_float(background)
    if background.shape[-1] == 1:
        background = gray2rgb(background)

    if foreground.dtype == np.uint8:
        foreground = uint8_to_float(foreground)
    if foreground.ndim == background.ndim:
        foreground = np.squeeze(foreground, axis=-1)

    foreground = np.stack([c * foreground for c in color], axis=-1)
    out = cv2.addWeighted(foreground, alpha, background, beta, gamma)
    if background.dtype == np.uint8:
        out = float_to_uint8(out, clip=clip)
    return out


def patchify(images, patch_shape, *args, **kwargs):
    """images has NHWC
    patch_shape has HW"""
    assert images.ndim in (3, 4)
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    out_padding = patch_shape - np.remainder(images.shape[1:-1], patch_shape)
    padding = np.append(out_padding, 0)
    padding = np.insert(padding, 0, 0)
    pre_pad = np.zeros((images.ndim), dtype=np.uint32)
    padding = list(zip(pre_pad, padding))
    padded = np.pad(images, padding, *args, **kwargs)

    patch_shape = np.array(patch_shape)
    patch_counts = np.array([x // y for x, y in zip(padded.shape[1:-1], patch_shape)])
    patches_shape = _interleave(patch_counts, patch_shape)
    patches_shape = np.append(patches_shape, images.shape[-1])
    patches_shape = np.insert(patches_shape, 0, -1)
    patches = padded.reshape(patches_shape)

    dim_order = _deinterleave(range(1, patches.ndim - 1))
    dim_order = np.append(dim_order, patches.ndim - 1)
    dim_order = np.insert(dim_order, 0, 0)
    patches = patches.transpose(dim_order)

    stacked_shape = (-1, *patch_shape, images.shape[-1])
    patches = patches.reshape(stacked_shape)

    return patches, patch_counts, out_padding


def rescale(
    image, out_range=(0.0, 1.0), in_range=(float("-inf"), float("+inf")), clip=False
):
    """Rescales image from in_range to out_range while retaining input dtype."""
    lo = in_range[0]
    if lo == float("-inf") or isnan(lo):
        lo = np.min(image)

    hi = in_range[1]
    if hi == float("+inf") or isnan(hi):
        hi = np.max(image)

    assert not isinf(lo) and not isnan(lo)
    assert not isinf(hi) and not isnan(hi)
    assert lo < hi

    med = (image - lo) / (hi - lo)
    out = med * (out_range[1] - out_range[0]) + out_range[0]
    if np.issubdtype(image.dtype, np.integer):
        out = np.round(out)
    if clip:
        out = np.clip(out, out_range[0], out_range[1])
    return out.astype(image.dtype)


def rgb2gray(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def rgb2lab(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_LAB2RGB)


def save(path, image):
    im = Image.fromarray(image)
    im.save(path)


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


def standardize(images):
    """Standardizes an (N+1)-D block of N-D images by the usual method, i.e.
    (x-u)/s.

    This function is intended to be used just before application of a machine
    learning model, or for training such a model. There is no guarantee the
    output will be in the usual (0.0, 1.0) range."""
    s = np.std(images)
    u = np.mean(images)
    standardized = (images - u) / s
    return standardized


def uint8_to_float(uint8_image, float_range=(0.0, 1.0), clip=False):
    return rescale(
        uint8_image.astype(np.float), in_range=(0.0, 255.0), out_range=float_range
    )


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


def show(image, tag="UNLABELED_WINDOW"):
    assert image.ndim in (2, 3)
    if image.ndim == 3:
        assert image.shape[-1] in (1, 3)

    cv2.namedWindow(tag, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(tag, image.shape[0:2][::-1])
    if image.shape[-1] == 3:
        image = image[..., ::-1]
    cv2.imshow(tag, image)
    cv2.waitKey(1)


def _deinterleave(c):
    a = c[0::2]
    b = c[1::2]
    return a, b


def _get_image_or_blank(images, index):
    try:
        return images[index]
    except Exception as e:
        return np.zeros(images.shape[1:], dtype=images.dtype)


def _interleave(a, b):
    c = np.empty((a.size + b.size), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def _optimize_shape(count, width_height_aspect_ratio=1.0):
    N = count
    W = np.arange(1, N).astype(np.uint32)
    H = np.ceil(N / W).astype(np.uint32)
    closest = np.argmin(np.abs((W / H) - width_height_aspect_ratio))
    return H[closest], W[closest]
