import functools
import inspect
from itertools import chain, cycle, islice
from math import floor, isinf, isnan
from pathlib import Path, PurePath
from random import shuffle
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image
import cv2
import noise
import numpy as np
import scipy.stats
import skimage
import tifffile.tifffile as tf

from .inc.file_utils.file_utils import get_contents

# TODO
# 3) automate test-cases (no visuals, just check against values from private
#    data folder, use small, highly-compressed files)
# 5) add more test-cases to get better coverage


PathLike = Union[str, Path, PurePath]
Number = Union[int, float]


def _copy(fn) -> Callable:
    @functools.wraps(fn)
    def wrapper(image: np.ndarray, *args, **kwargs) -> np.ndarray:
        kwargs = _update_with_defaults(fn, kwargs)
        print(kwargs)
        out = image.copy()
        out = fn(out, *args, **kwargs)
        return out

    return wrapper


def _as_dtype(dtype) -> Callable:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(image, *args, **kwargs):
            kwargs = _update_with_defaults(fn, kwargs)
            assert "use_signed_negative" in kwargs
            use_signed_negative = kwargs["use_signed_negative"]
            old_dtype = image.dtype
            out = to_dtype(image, dtype=dtype, negative_in=use_signed_negative)
            out = fn(out, *args, **kwargs)
            print(type(out))
            out = to_dtype(out, dtype=old_dtype, negative_out=use_signed_negative)
            assert out.dtype == old_dtype
            return out

        return wrapper

    return decorator


@_as_dtype(np.float64)
def to_colorspace(
    image: np.ndarray, fromspace: str, tospace: str, use_signed_negative: bool = False
) -> np.ndarray:
    out = skimage.color.convert_colorspace(
        image, fromspace=fromspace.upper(), tospace=tospace.upper()
    )
    out = _add_channel_dim(out)
    return out


@_as_dtype(np.float64)
@_copy
def to_gray(image: np.ndarray, use_signed_negative: bool = False):
    assert _is_image(image)
    assert _is_color(image)

    out = skimage.color.rgb2gray(image)
    out = _add_channel_dim(out)

    assert _is_image(out)
    assert _is_gray(out)

    return out


def _as_colorspace(space: str, fromspace: str = "rgb") -> Callable:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(image, *args, **kwargs):
            kwargs.update(_get_default_args(fn))
            assert "use_signed_negative" in kwargs
            use_signed_negative = kwargs["use_signed_negative"]
            assert _is_color(image)
            image = to_colorspace(
                image,
                fromspace=fromspace,
                tospace=space,
                use_signed_negative=use_signed_negative,
            )
            image = fn(image, *args, **kwargs)
            image = to_colorspace(
                image,
                fromspace=space,
                tospace=fromspace,
                use_signed_negative=use_signed_negative,
            )
            assert _is_color(image)
            return image

        return wrapper

    return decorator


@_as_dtype(np.float64)
@_copy
def adjust_gamma(
    image: np.ndarray, gamma: float = 1.0, use_signed_negative: bool = False
) -> np.ndarray:
    """
    Adjusts image gamma. Channels are adjusted uniformly.

    Inputs:
    1) image - HWC input image to adjust.
    2) gamma - Float denoting gamma value of output relative to input gamma of
        1.0
    3) use_signed_negative - Consider negative values in signed input types.

    Output:
    1) Grayscale or RGB ND input image with adjusted gamma.
    """
    assert _is_image(image)
    assert _is_gray(image) or _is_color(image)

    image = image ** (1.0 / gamma)

    assert _is_image(image)
    assert _is_gray(image) or _is_color(image)

    return image


@_copy
def clahe(
    image,
    clip_limit: float = 0.01,
    tile_size: Tuple[int, int] = (8, 8),
    use_signed_negative: bool = False,
):
    """
    Applies CLAHE equalization to input image. Works on color images by
    converting to Lab space, performing clahe on the L channel, and converting
    back to RGB. Note that the underlying image type used in the clahe process
    is uint16 which may create a narrowing conversion for some input dtypes.
    Returns the same dtype as input.

    Inputs:
    1) image - HWC input image to adjust.
    2) tile_size - 2-tuple integer size of tiles to perform clahe on.
    3) use_signed_negative - Consider negative values in signed input types.

    Output:
    1) HWC image of same dtype as input image.
    """
    assert _is_image(image)
    if _is_color(image):
        out = _clahe_rgb(
            image,
            clip_limit=clip_limit,
            tile_size=tile_size,
            use_signed_negative=use_signed_negative,
        )
    else:
        out = _clahe_channel(
            image,
            clip_limit=clip_limit,
            tile_size=tile_size,
            use_signed_negative=use_signed_negative,
        )
    assert _is_image(image)
    assert out.shape == image.shape

    return out


@_copy
def consensus(
    stack: np.ndarray, threshold: Optional[Union[str, float, int]] = None
) -> np.ndarray:
    """
    Builds a consensus label image from a stack of label images. If input is
    NHW1 output is HW1, with the same dtype as input. Input channels must be
    one.

    Inputs:
    1) stack - NHW1 stack of images with dtype bool or integer.
    2) threshold - String, float ratio or integer value
        1) stack has integer dtype
            1) "majority" - (integer dtype required, default) Finds the mode for each pixel.
                Computationally intensive. It is recommended to only use dtype np.uint8 as input.
        2) stack has bool dtype
            1) float - (bool dtype required) Ratio of image stack N in range
               [0.0, 1.0]. Finds pixels where more than the fraction supplied are
               True.
            2) integer - (bool dtype required) Absolute count of images in stack
               which are True is more than the supplied value.

    Output:
    1) HW1 image with same dtype as input
    """
    print(threshold)
    print(type(threshold))
    assert _is_stack(stack)
    assert _is_gray(stack)

    dtype = stack.dtype
    out = stack.copy()
    assert np.issubdtype(dtype, np.integer) or dtype == np.bool
    if threshold is None:
        if np.issubdtype(dtype, np.integer):
            threshold = "majority"
        elif dtype == np.bool:
            threshold = 0.5
        else:
            assert False
    assert isinstance(threshold, (str, float, int))
    if isinstance(threshold, float):
        assert 0.0 <= threshold <= 1.0
        threshold = round(threshold * stack.shape[0])
    if isinstance(threshold, str):
        assert threshold.casefold() == "majority"
        assert np.issubdtype(dtype, np.integer)
    elif isinstance(threshold, int):
        assert 0 <= threshold
        assert dtype == np.bool
    else:
        assert False

    if threshold == "majority":
        out = scipy.stats.mode(out, axis=0, nan_policy="omit")[0]
        out = out[0, ...]
    else:
        out = out.sum(axis=0) > threshold
        out = out.astype(dtype)

    assert out.dtype == dtype
    assert _is_image(out)
    assert _is_gray(out)

    return out


def generate_circular_fov_mask(shape, fov_radius, offset=(0, 0)):
    """
    Generates a circular field of view mask, with the interior of the circle
    included. The circle is assumed to be centered on the image shape.
    """
    center = get_center(shape)
    X, Y = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])
    R2 = (X - offset[0]) ** 2 + (Y - offset[1]) ** 2
    fov2 = fov_radius ** 2
    mask = R2 <= fov2
    return mask[..., np.newaxis]


def generate_noise(
    shape: Sequence[int],
    offsets: Sequence[Union[float, int]] = None,
    octaves: int = 1,
    dtype=np.uint8,
):
    """
    Generates a grayscale image of Perlin noise. Useful for testing.

    Inputs:
    1) shape - A sequence of two positive integers representing the desired output image size.
    2) offsets - (default np.random.uniform) A sequence of floats of the same length as shape, allowing choice/random Perlin noise.
    3) octaves - (default 1) A positive int denoting complexity of Perlin noise.
    4) dtype - Output dtype

    Oututs:
    1) HW1 image of type dtype
    """
    assert len(shape) == 2
    for x in shape:
        assert isinstance(x, int)
        assert 0 < x

    assert isinstance(octaves, int)
    assert 0 < octaves

    scale = 0.1 * np.array(shape, dtype=np.float64).max()
    if offsets is None:
        offsets = np.random.uniform(-1000 * scale, 1000 * scale, 2, dtype=np.float64)

    assert len(offsets) == len(shape)
    for x in offsets:
        assert isinstance(x, (float, int))
        if isinstance(x, int):
            x = float(x)
        assert 0.0 < x

    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    X = X + offsets[0]
    Y = Y + offsets[1]
    noise_maker = np.vectorize(
        lambda x, y: noise.pnoise2(x / scale, y / scale, octaves=octaves)
    )
    n = noise_maker(X, Y)
    return to_dtype(n, dtype=dtype)


def get_center(shape):
    """
    Returns the coordinates of the center point of an image in pixels, rounded
    down.
    """
    return [floor(x / 2) for x in shape]


def load(path: PathLike, force_rgb=False):
    """
    Loads an image from the supplied path in grayscale or RGB depending on the
    source. If force_rgb is True, the image will be returned with 3 channels. If
    the image has only one channel, then all channels will be identical.
    """
    if PurePath(path).suffix.casefold() in (".tif", ".tiff"):
        image = tf.imread(path)
    else:
        image = Image.open(str(path))
        image = np.array(image)

    image = _add_channel_dim(image)

    # convert redundant rgb to grayscale
    if _is_color(image):
        g_redundant = (image[..., 0] == image[..., 1]).all()
        b_redundant = (image[..., 0] == image[..., 2]).all()
        if (g_redundant and b_redundant) and not force_rgb:
            image = image[..., 0]
            image = image[..., np.newaxis]

    if _is_gray(image) and force_rgb:
        image = np.repeat(image, 3, axis=2)

    assert image.ndim == 3
    assert _is_gray(image) or _is_color(image)
    return image


def load_images(
    folder, force_rgb=False, ext=None
) -> Tuple[List[np.array], List[PurePath]]:
    """
    Loads a folder of images. If an extension is supplied, only images with that
    extension will be loaded. Also returns the filenames of every loaded image.
    """
    image_files = get_contents(folder, ext)
    images = []
    names = []
    for image_file in image_files:
        try:
            image = load(str(image_file), force_rgb=force_rgb)
        except Exception as e:
            continue
        images.append(image)
        names.append(image_file)

    return images, names


def mask_images(images, masks):
    """
    Masks out pixels in an image stack based on the masks. There must be either
    one mask, or the same number of images and masks.
    """
    if masks.shape[0] == 1:
        masks = np.repeat(masks, images.shape[0], axis=0)
    assert masks.shape[0] == images.shape[0]

    masked = images.copy()
    threshold = (masks.max() - masks.min()) / 2.0
    masked[masks <= threshold] = 0
    return masked


def montage(
    images,
    shape=None,
    mode="sequential",
    repeat=False,
    start=0,
    maximum_images=36,
    fill_value=0,
):
    """
    Generates a montage image from an image stack.

    shape determines the number of images to tile in each dimension. Must be an
    iterable of length 2 containing positive integers, or a single positive
    float, or None. If a float is provided, that value is assumed to be a
    width-to-height aspect ratio and the shape is computed to best fit that
    aspect ratio. If None is provided, the aspect ratio is set to 1. In both
    latter cases, the montage has tiles at least equal to the number of images
    in the stack if maximum_images is not also set. Default is None.

    mode determines how to sample images from the stack. Must be either
    "sequential" or "random". If "sequential", the images are sampled in the
    order they appear in the stack. If "random", then the stack order is
    shuffled before sampling. Default is "sequential".

    repeat determines whether to sample the stack from the start if more images
    are requested than exist in the stack. If repeat is set to False and more
    images are requested than exist, the remaining tiles are filled with zeros,
    i.e. black. Default is False.

    start determines the starting index for sampling the stack. Default is 0.

    maximum_images determines the limit of images to be sampled from the stack,
    regardless of shape. Default is 36.
    """
    # TODO add stitch border width, color

    assert images.ndim == 4

    image_count = images.shape[0] - start
    if maximum_images is not None:
        image_count = min(maximum_images, image_count)

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

    stop = int(np.array(shape).prod() + start)
    iterator = islice(iterator, start, stop)

    montage = np.stack([_get_image_or_blank(images, i, fill_value) for i in iterator])
    montage = montage.reshape((*shape, *images.shape[1:]))

    a, b = _deinterleave(list(range(0, montage.ndim - 1)))
    a, b = list(a), list(b)
    dim_order = (*a, *b, montage.ndim - 1)
    montage = montage.transpose(dim_order)

    image_shape = np.array(shape) * np.array(images.shape[1:-1])
    image_shape = np.append(image_shape, images.shape[-1])
    return montage.reshape(image_shape)


def overlay(
    background,
    foreground,
    color,
    alpha=0.5,
    beta=0.5,
    gamma=0.0,
    use_signed_negative: bool = False,
):
    """
    Applies a color to the supplied grayscale foreground and then blends it with
    the background using mixing ratio parameters alpha and beta. Background may
    be RGB or grayscale. Foreground and background may both be either float or
    uint*. Output is RGB with the same dtype as background. Gamma is a constant
    ratio parameter to add to all pixels.

    Note that if the sum of alpha, beta and gamma is greater than 1.0, clipping
    can occur.

    Note that underlying computation is performed on int32, which may create a
    narrowing conversion. Returned dtype is same as background.
    """
    assert _is_image(background)
    assert _is_gray(background) or _is_color(background)
    assert _is_image(foreground)
    assert _is_gray(foreground)
    assert len(color) == 3
    for c in color:
        assert 0.0 <= c <= 1.0

    dtype = background.dtype
    background = background.copy()
    background = to_dtype(background, dtype=np.int32, negative_in=use_signed_negative)

    foreground = foreground.copy()
    foreground = to_dtype(foreground, dtype=np.int32, negative_in=use_signed_negative)
    foreground = gray_to_color(foreground, color=color)

    out = cv2.addWeighted(
        src1=foreground, src2=background, alpha=alpha, beta=beta, gamma=gamma
    )
    out = to_dtype(out, dtype=dtype, negative_out=use_signed_negative)

    return out


@_as_dtype(np.float64)
@_copy
def gray_to_color(
    image: np.ndarray, color: Sequence[float], use_signed_negative: bool = False
) -> np.ndarray:
    assert _is_image(image)
    assert _is_gray(image)

    assert len(color) == 3
    for c in color:
        assert 0.0 <= c <= 1.0

    out = image.squeeze()
    out = skimage.color.gray2rgb(out)
    out = out * np.array(color)

    assert _is_image(out)
    assert _is_color(out)

    return out


def patchify(image_stack, patch_shape, offset=(0, 0), *args, **kwargs):
    """
    Transforms an image stack into a new stack made of tiled patches from images
    of the original stack. The size of the patches is determined by patch_shape.
    If patch_shape does not evenly divide the image shape, the excess is padded
    with zeros, i.e. black.

    If there are N images of size X by Y, and patches of size M by N are
    requested, then the resulting stack will have N * ceil(X/M) * ceil(Y/N)
    images. The first ceil(X/M) * ceil(Y/N) patches all come from the first
    image, sampled along X first, then Y.

    Returns a tuple consisting of an image stack of patches, the number of
    patches in each dimension, and the padding used. The latter two values are
    used for unpatching.

    Inputs:

    image_stack: Stack of images of shape NHW or NHWC. Assumed to have 2D
    images.

    patch_shape: Spatial shape of patches. All channel information is retained.
    If patch shape is size M by N, then the resulting stack of patches will have
    N * ceil(X/M) * ceil(Y/N) images. Every ceil(X/M) * ceil(Y/N) patches in the
    stack belong to a single image. Patches are sampled along X first, then Y.
    If M divides X and N divides Y, then a non-zero offset will change the
    number of patches.

    offset: Offset is the spatial location of the lower-right corner of the
    top-left patch relative to the image origin. Because the patch boundaries
    are periodic, any value greater than patch_shape in any dimension is reduced
    module patch_shape. The value of any pixels outside the image are assigned
    according to arguments for np.pad().
    """
    if image_stack.ndim == 3:
        image_stack = image_stack[np.newaxis, ...]
    assert _is_stack(image_stack)

    assert len(patch_shape) == 2
    assert len(offset) == 2

    offset = [o % p for o, p in zip(offset, patch_shape)]

    # determine pre padding
    pre_padding = [((p - o) % p) for o, p in zip(offset, patch_shape)]
    pre_padding = np.append(pre_padding, 0)
    pre_padding = np.insert(pre_padding, 0, 0)
    # compute post padding from whatever is left
    pre_image_shape = [
        s + pre for s, pre in zip(image_stack.shape[1:-1], pre_padding[1:-1])
    ]
    post_padding = patch_shape - np.remainder(pre_image_shape, patch_shape)
    post_padding = np.append(post_padding, 0)
    post_padding = np.insert(post_padding, 0, 0)
    padding = list(zip(pre_padding, post_padding))
    padded = np.pad(image_stack, padding, *args, **kwargs)
    out_padding = padding[1:-1]

    patch_shape = np.array(patch_shape)
    patch_counts = np.array([x // y for x, y in zip(padded.shape[1:-1], patch_shape)])
    patches_shape = _interleave(patch_counts, patch_shape)
    patches_shape = np.append(patches_shape, image_stack.shape[-1])
    patches_shape = np.insert(patches_shape, 0, -1)
    patches = padded.reshape(patches_shape)

    dim_order = _deinterleave(range(1, patches.ndim - 1))
    dim_order = np.append(dim_order, patches.ndim - 1)
    dim_order = np.insert(dim_order, 0, 0)
    patches = patches.transpose(dim_order)

    stacked_shape = (-1, *patch_shape, image_stack.shape[-1])
    patches = patches.reshape(stacked_shape)

    return patches, patch_counts, out_padding


def rescale(
    image: np.ndarray,
    out_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    in_range: Optional[Tuple[Optional[Number], Optional[Number]]] = None,
    clip: bool = False,
    dtype=None,
    negative_allowed_out: bool = False,
) -> np.ndarray:
    """
    Rescales image values from in_range to out_range. Note that in_range and
    out_range need not have any relation to the underlying dtypes or each other.
    This means, for example, that a uint8 image can have values from 0-127
    scaled out to 0-255, washing out values 128-255. An optionally supplied
    dtype may be used to change the dtype. The function uses np.float64 as a
    common language. All scaling is performed linearly.

    If a dtype is supplied, the output image will have that dtype.

    If clip is set to True, the resulting image values are clipped to the
    narrower of out_range or the limits of the output dtype. Naturally, values
    are always clipped to the limits of the output dtype.

    Default behavior is to scale the image values to the full range of the
    output dtype. Default output dtype is the input dtype.

    Inputs:
    1) image - 2D image whose values will be rescaled.
    2) out_range - (optional) Range of values that in_range is mapped to. Default is range
       of output dtype. See below for more information on allowed values.
    3) in_range - (optional) Range of values to map to out_range. Default is range of
       values in image. See below for more information on allowed values.
    4) clip - clips values to narrower of out_range and dtype range if True. Default is False
    5) dtype - (optional) Output will have this dtype. May cause narrowing
       conversion if care is not taken with in_range/out_range relationship.
    6) negative_allowed_out - If True, instead of zero, uses negative minimum in
       default range for signed integer output dtypes. Default is False.

    Allowed values of *_range inputs:
    1) If *_range is None, the default range is used.
    2) If *_range is a tuple with at least one: None; (-/+)inf; or NaN; the
       default range min or max is used for 1st and 2nd tuple position,
       respectively of (-/+).
    3) First tuple value of range (minimum) must be less than or equal to second
       tuple value (maximum).
    4) If minimum and maximum are equal then the image is returned unchanged.
    5) Ranges of dtypes are full width for integral types and [0.0, 1.0] for
       floating types.
    """
    if dtype is None:
        dtype = image.dtype
    assert dtype is not None

    image = image.copy().astype(np.float64)

    # IN RANGE
    if in_range is None:
        in_range = (None, None)

    in_lo = in_range[0]
    if in_lo is None or in_lo == float("-inf") or isnan(in_lo):
        in_lo = np.nanmin(image)
    assert not isinf(in_lo) and not isnan(in_lo) and in_lo is not None

    in_hi = in_range[1]
    if in_hi is None or in_hi == float("+inf") or isnan(in_hi):
        in_hi = np.nanmax(image)
    assert not isinf(in_hi) and not isnan(in_hi) and in_hi is not None

    assert in_lo <= in_hi
    in_range = (in_lo, in_hi)

    # OUT RANGE
    if out_range is None:
        out_range = (None, None)

    out_lo = out_range[0]
    if out_lo is None or out_lo == float("-inf") or isnan(out_lo):
        out_lo = _get_dtype_range(dtype, allow_negative=negative_allowed_out)[0]
    assert not isinf(in_lo) and not isnan(in_lo) and in_lo is not None

    out_hi = out_range[1]
    if out_hi is None or out_hi == float("+inf") or isnan(out_hi):
        out_hi = _get_dtype_range(dtype, allow_negative=negative_allowed_out)[1]
    assert not isinf(out_hi) and not isnan(out_hi) and out_hi is not None

    assert out_lo <= out_hi
    out_range = (out_lo, out_hi)

    if in_range[0] == in_range[1] or out_range[0] == out_range[1]:
        out = image
    else:
        med = (image - in_range[0]) / (in_range[1] - in_range[0])
        out = med * (out_range[1] - out_range[0]) + out_range[0]

    if clip:
        out = np.clip(out, out_range[0], out_range[1])

    return out.astype(dtype)


def resize(image, method="linear", size=None, scale=1.0):
    """
    Resizes image to a specific size or by a scaling factor using supplied method.

    Inputs:
    1) image - 2D image to resize
    2) method - one of:
        1) nearest - resized pixel will be nearest neighbor pixel from original image
        2) linear - (default) bilinear interpolation
        3) cubic - bicubic interpolation
        4) area - resample by pixel area relation, moire free for image shrinking, not suitable for image growing
        5) lanczos4 - 8x8 lanczos interpolation
        6) nearest_exact - bit exact nearest neighbor interpolation, same as PIL, scikit-image, MATLAB
        7) linear_exact - bit exact bilinear interpolation
    """
    METHODS = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
        "linear_exact": cv2.INTER_LINEAR_EXACT,
    }
    assert method in METHODS

    if size is None:
        if isinstance(scale, float):
            scale = (scale, scale)
        assert isinstance(scale, tuple)
        assert len(scale) == 2
        assert isinstance(scale[0], float)
        assert isinstance(scale[1], float)

        out = cv2.resize(image, (0, 0), None, scale[0], scale[1], METHODS[method])
    elif scale is None:
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert isinstance(size[0], int)
        assert isinstance(size[1], int)

        out = cv2.resize(image, size)
    elif scale is None and size is None:
        assert False
    else:
        assert False
    return out


def save(image, path: PathLike, dtype=None):
    """
    Saves an image to disk at the location specified by path.
    """
    image = image.copy()
    if dtype is not None:
        image = to_dtype(image, dtype=dtype)
    if image.shape[-1] == 1:
        image = image.squeeze(axis=-1)
    if PurePath(path).suffix.casefold() in (".tif", ".tiff"):
        image = tf.imwrite(path, data=image)
    else:
        im = Image.fromarray(image)
        im.save(str(path))


def save_images(paths, image_stack):
    """
    Saves an image stack to disk as individual images using save() with index
    appended to the supplied file name, joined by delimiter.

    paths is the file paths for images to be written.

    images is a Numpy array whose shape is of the form (NHWC) where N is the
     number of images, HW are spatial dimensions, and C are the channels. N may
     be any positive number, H and W may be any positive numbers, and C must be
     1 or 3.
    """
    assert _is_stack(image_stack)
    assert len(paths) == len(image_stack)
    for path, image in zip(paths, image_stack):
        save(path, image)


def show(image, tag="UNLABELED_WINDOW"):
    """
    Displays an image in a new window labeled with tag.
    """
    assert image.ndim in (2, 3)
    if image.ndim == 3:
        assert _is_gray(image) or _is_color(image)

    cv2.namedWindow(tag, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(tag, image.shape[0:2][::-1])
    if _is_color(image):
        image = image[..., ::-1]
    cv2.imshow(tag, image)
    cv2.waitKey(1)


def stack(images):
    """
    Converts a single image or an iterable of images to an image stack. An image
    stack is a numpy array whose first dimension is the image index.
    """
    if type(images) is np.ndarray:
        images = (images,)
    images = [image[..., np.newaxis] if image.ndim == 2 else image for image in images]
    return np.stack(images)


def standardize(images):
    """
    Standardizes an (N+1)-D block of N-D images by the usual method, i.e.
    (x-u)/s.

    This function is intended to be used just before application of a machine
    learning model, or for training such a model. There is no guarantee the
    output will be in the usual (0.0, 1.0) range.
    """
    s = np.std(images)
    u = np.mean(images)
    standardized = (images - u) / s
    return standardized


def to_dtype(
    image: np.ndarray, dtype, negative_in: bool = False, negative_out: bool = False
) -> np.ndarray:
    """
    Converts image to desired dtype by rescaling input dtype value range into
    output dtype value range. If negative_* is True, the range [dtype_min,
    dtype_max] is used for signed integer types, otherwise [0, dtype_max] is
    used. Allowed dtypes include any subdtype of np.integer or np.floating.

    Inputs:
    1) image - 2D image to convert to dtype.
    2) dtype - Desired output dtype.
    3) negative_in - (bool) Use full signed integer range for input dtype.
       Default is False.
    4) negative_out - (bool) Use full signed integer range for output dtype.
       Default is False.

    IMPORTANT: For single-channel images, most file formats only support
    unsigned integer types. TIFF allow signed integer types. For multi-channel
    images, most file formats only uint8 images.
    """
    in_range = _get_dtype_range(image.dtype, allow_negative=negative_in)
    out_range = _get_dtype_range(dtype, allow_negative=negative_out)
    return rescale(image, out_range=out_range, in_range=in_range, dtype=dtype)


def unpatchify(patches, patch_counts, padding):
    """
    Inverse of patchify(). Transforms an image stack of patches produced using
    patchify() back into an image stack of the same shape as the original
    images. Requires the patch_count and padding returned by patchify().
    """
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
    padding = list(zip(*padding))
    pre_padding = padding[0]
    post_padding = padding[1]
    space_shape = [
        base - pre - post
        for base, pre, post in zip(base_shape, pre_padding, post_padding)
    ]
    # space_shape = base_shape - padding
    slices = [slice(pre, pre + x) for pre, x in zip(pre_padding, space_shape)]
    slices.append(slice(None))
    slices.insert(0, slice(None))
    images = images[tuple(slices)]
    return images


def _deinterleave(c):
    """
    Separates two interleaved sequences into a tuple of two sequences of the
    same type.
    """
    a = c[0::2]
    b = c[1::2]
    return a, b


def _get_dtype_range(dtype, allow_negative=False):
    if np.issubdtype(dtype, np.integer):
        ii = np.iinfo(dtype)
        if not allow_negative and np.issubdtype(dtype, np.signedinteger):
            value = (0, ii.max)
        else:
            value = (ii.min, ii.max)
    elif np.issubdtype(dtype, np.floating):
        value = (0.0, 1.0)
    elif dtype == np.bool:
        value = (False, True)
    else:
        raise TypeError("supplied dtype is unsupported: {:s}".format(str(dtype)))
    return value


def _get_image_or_blank(images, index, fill_value=0):
    try:
        return images[index]
    except Exception as e:
        return np.zeros(images.shape[1:], dtype=images.dtype) + fill_value


def _interleave(a, b):
    """
    Interleaves two sequences of the same type into a single sequence.
    """
    c = np.empty((a.size + b.size), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c


def _is_color(image: np.ndarray) -> bool:
    if image.ndim == 3:
        is_rgb = image.shape[-1] == 3
    else:
        is_rgb = False
    return is_rgb


def _is_gray(image: np.ndarray) -> bool:
    if image.ndim > 2:
        is_gray = image.shape[-1] == 1
    else:
        is_gray = False
    return is_gray


def _optimize_shape(count, width_height_aspect_ratio=1.0):
    """
    Computes the optimal X by Y shape of count objects given a desired
    width-to-height aspect ratio.
    """
    N = count
    W = np.arange(1, N).astype(np.uint32)
    H = np.ceil(N / W).astype(np.uint32)
    closest = np.argmin(np.abs((W / H) - width_height_aspect_ratio))
    return H[closest], W[closest]


def _add_channel_dim(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., np.newaxis]
    return image


def _is_stack(image):
    return image.ndim == 4


def _is_image(image):
    return image.ndim == 3


@_as_colorspace("hsv")
def _clahe_rgb(image, *args, **kwargs):
    assert _is_color(image)
    out = _add_channel_dim(image[..., -1])
    out = _clahe_channel(out, *args, **kwargs)
    image[..., -1] = out[..., -1]
    assert _is_color(image)

    return image


@_as_dtype(np.uint16)
def _clahe_channel(
    image,
    clip_limit: float,
    tile_size: Tuple[int, int],
    use_signed_negative: bool = False,
):
    assert _is_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    out = clahe.apply(image)
    out = _add_channel_dim(out)
    assert _is_gray(out)

    return out


def _get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def _update_with_defaults(fn: Callable, kwargs: dict) -> dict:
    for k, v in _get_default_args(fn).items():
        if k not in kwargs:
            kwargs[k] = v
    return kwargs
