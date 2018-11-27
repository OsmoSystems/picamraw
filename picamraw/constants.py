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
