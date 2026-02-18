import numcodecs
import zarr


def create_compressor(cname, clevel, zarr_format):
    if zarr_format == 2:
        return numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=numcodecs.Blosc.SHUFFLE)
    else:
        return zarr.codecs.BloscCodec(cname=cname, clevel=clevel, shuffle='shuffle')
