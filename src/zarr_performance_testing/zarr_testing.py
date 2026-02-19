import dask.array as da
import zarr

from src.zarr_performance_testing.util import create_compressor


def create_zarr(filename, data, axes, chunk_size, shard_size, compressor='blosc', ome_version='0.5'):
    zarr_format = 3 if float(ome_version) >= 0.5 else 2
    compressors = []
    if compressor is not None:
        if compressor.lower() == 'blosc':
            compressor = create_compressor('zstd', 3, zarr_format)
        compressors.append(compressor)
    if isinstance(data, da.Array):
        data.to_zarr(filename, mode='w', chunks=chunk_size, shards=shard_size)
    else:
        array = zarr.create_array(filename, shape=data.shape, dtype=data.dtype, chunks=chunk_size, shards=shard_size,
                                  dimension_names=list(axes), compressors=compressors, zarr_format=zarr_format, overwrite=True)
        array[:] = data


def read_zarr(filename):
    data = zarr.open_array(filename, mode='r')
    return data[:]
