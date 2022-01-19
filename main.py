from almiky.quantization.scalar import UniformQuantizer
from almiky.embedding.qim import dm
from almiky.utils.scan import maps
from almiky.utils.scan.scan import ScanMapping
from almiky.attacks import noises

import numpy as np


def main():
    q = UniformQuantizer(step=30)
    d = dm.BinaryDither(step=30, d0=-3)
    emb = dm.BinaryDM(q, d)

    block = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29, 30, 31], 
        [32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47],
        [48, 49, 50, 51, 52, 53, 54, 55],
        [56, 57, 58, 59, 60, 61, 62, 63]
    ])

    # Selecting coefficient in 8 index in ZIG ZAG scaning
    scan = ScanMapping(maps.ZIGZAG_8x8)
    scanning = scan(block)
    amplitud = scanning[8]
    assert amplitud == 17

    # Insert a bit (0)
    scanning[8] = emb.embed(amplitud, 0)
    noisy = noises.gaussian_noise(block, percent_noise=5)
    noisy = noises.salt_pepper_noise(noisy, density=0.1)
    scanning = scan(noisy)
    amplitud = scanning[8]

    # Extact and test success
    assert emb.extract(amplitud) == 0
    
    # Insert a bit (1)
    scanning[8] = emb.embed(amplitud, 1)
    noisy = noises.gaussian_noise(block, percent_noise=5)
    noisy = noises.salt_pepper_noise(noisy, density=0.1)
    scanning = scan(noisy)
    amplitud = scanning[8]

    # Extact and test success
    assert emb.extract(amplitud) == 0

    

if __name__ == "__main__":
    main()

