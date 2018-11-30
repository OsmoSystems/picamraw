from enum import Enum


class PiCameraVersion(Enum):
    V1 = 'OV5647'
    V2 = 'IMX219'


class BayerOrder(Enum):
    ''' There are four supported arrangements of the R, G, G, and B pixels:
        RGGB:
              RG
              GB
        GBRG:
              GB
              RG
        BGGR:
              BG
              GR
        GRBG:
              GR
              BG
    '''

    RGGB = 'RGGB'
    GBRG = 'GBRG'
    BGGR = 'BGGR'
    GRBG = 'GRBG'


BAYER_ORDER_TO_RGB_CHANNEL_COORDINATES = {
    # (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
    BayerOrder.RGGB: ((0, 0), (1, 0), (0, 1), (1, 1)),
    BayerOrder.GBRG: ((1, 0), (0, 0), (1, 1), (0, 1)),
    BayerOrder.BGGR: ((1, 1), (0, 1), (1, 0), (0, 0)),
    BayerOrder.GRBG: ((0, 1), (1, 1), (0, 0), (1, 0)),
}
