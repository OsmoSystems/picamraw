import pkg_resources

import numpy as np
import pytest

from .constants import BayerOrder, PiCameraVersion
from . import main as module

picamv2_jpeg_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2.jpeg')
picamv2_BGGR_bayer_array_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2_BGGR_bayer_array.npy')
picamv2_3d_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2_3d.npy')


# Integration test using a known image
class TestPiRawBayer:
    raw_bayer = module.PiRawBayer(
        filepath=picamv2_jpeg_path,
        camera_version=PiCameraVersion.V2,
        sensor_mode=0
    )

    def test_extracts_bayer_order(self):
        assert self.raw_bayer.bayer_order == BayerOrder.BGGR

    def test_extracts_raw_data(self):
        # Spot-check some known pixels
        assert self.raw_bayer.bayer_array[0][0] == 88
        assert self.raw_bayer.bayer_array[-1][-1] == 65

        # Compare to full np array
        expected = np.load(picamv2_BGGR_bayer_array_path)
        assert np.array_equal(self.raw_bayer.bayer_array, expected)

    def test_to_3d(self):
        expected = np.load(picamv2_3d_path)
        assert np.array_equal(self.raw_bayer.to_3d(), expected)


class TestBayerArrayTo3D:
    bayer_array = np.array([
        [1, 2],
        [3, 4],
    ])

    @pytest.mark.parametrize('bayer_order,expected', [
        (BayerOrder.RGGB, np.array([
            [[1, 0, 0], [0, 2, 0]],
            [[0, 3, 0], [0, 0, 4]],
        ])),
        (BayerOrder.GBRG, np.array([
            [[0, 1, 0], [0, 0, 2]],
            [[3, 0, 0], [0, 4, 0]],
        ])),
        (BayerOrder.BGGR, np.array([
            [[0, 0, 1], [0, 2, 0]],
            [[0, 3, 0], [4, 0, 0]],
        ])),
        (BayerOrder.GRBG, np.array([
            [[0, 1, 0], [2, 0, 0]],
            [[0, 0, 3], [0, 4, 0]],
        ])),
    ])
    def test_splits_to_rgb_using_bayer_order(self, bayer_order, expected):
        actual = module.bayer_array_to_3d(self.bayer_array, bayer_order)

        assert np.array_equal(actual, expected)

    def test_integration(self):
        bayer_array = np.load(picamv2_BGGR_bayer_array_path)
        actual = module.bayer_array_to_3d(bayer_array, BayerOrder.BGGR)

        expected = np.load(picamv2_3d_path)

        assert np.array_equal(actual, expected)
