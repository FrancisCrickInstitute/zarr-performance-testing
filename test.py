import dask.array as da
import logging
import numpy as np
import os
import shutil
import zarr

from src.zarr_performance_testing.Timer import Timer
from src.zarr_performance_testing.zarr_testing import create_zarr as create_zarr
from src.zarr_performance_testing.zarr_testing import read_zarr as read_zarr
from src.zarr_performance_testing.ome_zarr_testing import create_zarr as create_zarr_ome
from src.zarr_performance_testing.ome_zarr_testing import read_zarr as read_zarr_ome


def create_data(shape, chunk_size):
    data = da.random.randint(0, 255, size=shape, chunks=chunk_size, dtype=np.uint8)
    return data


def init_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger('ome_zarr').setLevel(logging.WARNING)


def validate_zarr_data(filename, data, chunk_size, shard_size):
    node = zarr.open(filename, mode='r')
    if isinstance(node, zarr.Array):
        new_data = node
    else:
        key = sorted(node)[0]
        new_data = node[key]
    assert new_data.shape == data.shape
    assert new_data.dtype == data.dtype
    assert new_data.chunks == chunk_size
    assert new_data.shards == shard_size
    return data


def test(create_func, read_func, filename, data, axes, chunk_size, shard_size, ome_version='0.5'):
    init_logging()

    if os.path.exists(filename):
        shutil.rmtree(filename, ignore_errors=True)

    with Timer(create_func.__name__ + f' {data.shape} {chunk_size} {shard_size}'):
        create_func(filename=filename, data=data, axes=axes,
                    chunk_size=chunk_size, shard_size=shard_size, ome_version=ome_version)
    validate_zarr_data(filename=filename, data=data,
                       chunk_size=chunk_size, shard_size=shard_size)
    with Timer(read_func.__name__ + f' {data.shape} {chunk_size} {shard_size}'):
        read_func(filename=filename)


def test_data_range(filename):
    axes = 'yx'

    shape_chunks_shards_sets = [
        ((102400, 102400), (1024, 1024), None),
        ((102400, 102400), (10240, 10240), None),
        ((102400, 102400), (1024, 1024), (10240, 10240)),
        ((102400, 102400), (1024, 1024), (10240, 10240)),

        ((1024000, 1024000), (1024, 1024), None),
        ((1024000, 1024000), (10240, 10240), None),
        ((1024000, 1024000), (1024, 1024), (1024, 1024)),
        ((1024000, 1024000), (1024, 1024), (10240, 10240)),
    ]

    for shape, chunk_size, shard_size in shape_chunks_shards_sets:
        data = create_data(shape, chunk_size)
        test(create_zarr, read_zarr, filename, data, axes, chunk_size, shard_size)


def test_packages():
    shape = (1024, 1024)
    axes = 'yx'
    chunk_size = (1024, 1024)
    shard_size = (1024, 1024)

    data = create_data(shape, chunk_size)
    test(create_zarr, read_zarr, 'test.zarr', data, axes, chunk_size, shard_size)

    data = create_data(shape, chunk_size)
    # Sharding not currently supported for ome-zarr through dask array
    test(create_zarr_ome, read_zarr_ome, 'test.ome.zarr', data, axes, chunk_size, shard_size=None)


if __name__ == "__main__":
    filename = '/nemo/project/proj-ccp-vem/test.zarr'
    test_data_range(filename)
