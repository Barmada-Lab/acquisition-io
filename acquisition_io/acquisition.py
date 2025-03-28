import random
from collections.abc import Generator
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import TypeVar

import xarray as xr

T = TypeVar("T", xr.DataArray, xr.Dataset)


class ExperimentType(StrEnum):
    CQ1 = "cq1"
    ND2 = "nd2"
    LUX = "lux"
    LEGACY = "legacy"
    LEGACY_ICC = "legacy-icc"


def load_experiment(
    path: str | Path, experiment_type: ExperimentType, fillna: bool = False
) -> xr.DataArray:
    if isinstance(path, str):
        path = Path(path)

    match experiment_type:
        case ExperimentType.LEGACY:
            from acquisition_io.loaders.legacy_loader import load_legacy

            return load_legacy(path, fillna)
        case ExperimentType.LEGACY_ICC:
            from acquisition_io.loaders.legacy_loader import load_legacy_icc

            return load_legacy_icc(path, fillna)
        case ExperimentType.ND2:
            from acquisition_io.loaders.nd2_loader import load_nd2_collection

            return load_nd2_collection(path)
        case ExperimentType.LUX:
            from acquisition_io.loaders.lux_loader import load_lux

            return load_lux(path, fillna)
        case ExperimentType.CQ1:
            from acquisition_io.loaders.cq1_loader import load_cq1

            return load_cq1(path)


def apply_ufunc_xy(func, arr: xr.DataArray, ufunc_kwargs=None, **kwargs):
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    return xr.apply_ufunc(
        func,
        arr,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        vectorize=True,
        kwargs=ufunc_kwargs,
        **kwargs,
    )


def iter_idx_prod(arr: T, subarr_dims=None, shuffle=False) -> Generator[T, None, None]:
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
