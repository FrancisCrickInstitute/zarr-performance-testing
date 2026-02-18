import dask.array as da
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from ome_zarr.writer import write_image

from src.zarr_performance_testing.util import create_compressor


def create_zarr(filename, data, axes, chunk_size, shard_size, compressor='blosc', ome_version='0.5'):
    # sharding from dask array not currently supported through da.to_zarr()
    zarr_format = 3 if float(ome_version) >= 0.5 else 2
    compressors = []
    if compressor is not None:
        if compressor.lower() == 'blosc':
            compressor = create_compressor('zstd', 3, zarr_format)
        compressors.append(compressor)
    storage_options = {
        'chunks': chunk_size,
        'shards': shard_size
    }
    root = zarr.create_group(store=filename, zarr_format=zarr_format, overwrite=True)
    write_image(image=data, group=root, axes=axes, chunks=chunk_size, shards=shard_size, storage_options=storage_options, compressors=compressors)


def read_zarr(filename, level=0):
    reader = Reader(parse_url(filename))
    # nodes may include images, labels etc
    nodes = list(reader())
    # first node will be the image pixel data
    image_node = nodes[0]

    # list of dask arrays at different pyramid levels
    data = image_node.data
    if level >= 0:
        data = data[level]
        if isinstance(data, da.Array):
            data = data.compute()
    else:
        if isinstance(data, (list, tuple)) and isinstance(data[0], da.Array):
            data = [d.compute() for d in data]
    return data
