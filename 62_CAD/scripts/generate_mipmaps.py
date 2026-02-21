import OpenEXR
import Imath
import numpy as np

def read_exr(path):
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    data = [np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(size[1], size[0]) for c in channels]
    return np.stack(data, axis=-1)  # shape: (H, W, 3)

def write_exr(path, arr):
    H, W, C = arr.shape
    assert C == 3, "Only RGB supported"
    header = OpenEXR.Header(W, H)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = {
        'R': arr[:, :, 0].astype(np.float32).tobytes(),
        'G': arr[:, :, 1].astype(np.float32).tobytes(),
        'B': arr[:, :, 2].astype(np.float32).tobytes()
    }
    exr = OpenEXR.OutputFile(path, header)
    exr.writePixels(channels)

def mipmap_exr():
    img = read_exr("../../media/tiled_grid_mip_0.exr")
    h, w, _ = img.shape
    base_path = "../../media/tiled_grid_mip_"
    tile_size = 128
    mip_level = 1
    tile_length = h // (2 * tile_size)
    
    while tile_length > 0:
        # Reshape and average 2x2 blocks
        reshaped = img.reshape(h//2, 2, w//2, 2, 3)
        mipmap = reshaped.mean(axis=(1, 3))
        write_exr(base_path + str(mip_level) + ".exr", mipmap)
        img = mipmap
        mip_level = mip_level + 1
        tile_length = tile_length // 2
        h = h // 2
        w = w // 2

mipmap_exr()