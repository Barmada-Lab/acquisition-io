import logging
import pathlib as pl
import random
from collections.abc import Generator
from itertools import product

import dask
import dask.array as da
import numpy as np
import tifffile
import xarray as xr
from skimage import transform  # type: ignore

logger = logging.getLogger(__name__)


def read_tiff_delayed(shape: tuple, reshape: bool = True):
    def read(path: pl.Path) -> np.ndarray:
        try:
            logger.debug(f"Reading {path}")
            img = tifffile.imread(path)
            if img.shape != shape and reshape:
                img = transform.resize(
                    img, shape, preserve_range=True, anti_aliasing=True
                )
            elif img.shape != shape and not reshape:
                raise ValueError(
                    f"Image shape {img.shape} does not match expected shape {shape};"
                    "you can pass reshape=True to resize the image to a standard shape."
                )
            return img.astype(np.float32)
        except (ValueError, NameError, FileNotFoundError) as e:
            logger.warning(
                f"Error reading {path}: {e}\nThis field will be filled based on"
                "surrounding fields and timepoints."
            )
            img = np.zeros(shape, dtype=np.float32)
            img[:] = np.nan
            return img

    return dask.delayed(read)


def read_tiff_toarray(path: pl.Path, shape: tuple = (2048, 2048)):
    return da.from_delayed(read_tiff_delayed(shape)(path), shape, dtype=np.float32)


def _get_float_color(hexcode: str):
    h = hexcode.lstrip("#")
    rgb = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    max_val = max(rgb)
    rgb_corrected = tuple(x / max_val for x in rgb)
    return rgb_corrected


_float_colors = {
    "DAPI": "#007fff",
    "RFP": "#ffe600",
    "GFP": "#00ff00",
    "Cy5": "#ff0000",
    "white_light": "#ffffff",
}


def get_float_color(channel: str):
    if channel in _float_colors:
        return _get_float_color(_float_colors[channel])
    else:
        raise ValueError(f"Channel {channel} is not known")


def iter_idx_prod[T: xr.DataArray](
    arr: T, subarr_dims=None, shuffle=False
) -> Generator[T, None, None]:
    """
    Iterates over the product of an array's indices. Can be used to iterate over
    all the (coordinate-less) XY(Z) planes in an experiment.
    """
    if subarr_dims is None:
        subarr_dims = []
    indices = [name for name in arr.indexes if name not in subarr_dims]
    idxs = list(product(*[arr.indexes[name] for name in indices]))
    if shuffle:
        random.shuffle(idxs)
    for coords in idxs:
        selector = dict(zip(indices, coords, strict=False))
        yield arr.sel(selector)
