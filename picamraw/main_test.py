import io
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
        camera_version=PiCameraVersion.V2.value,
        sensor_mode=0
    )

    def test_extracts_bayer_order(self):
        assert self.raw_bayer.bayer_order == BayerOrder.BGGR

    def test_extracts_raw_data(self):
        # Spot-check some known pixels
        assert self.raw_bayer.bayer_array[0][0] == 88
        assert self.raw_bayer.bayer_array[-1][-1] == 65

        # Compare to full np array
        actual = self.raw_bayer.bayer_array
        expected = np.load(picamv2_BGGR_bayer_array_path)

        np.testing.assert_array_equal(actual, expected)

    def test_to_3d(self):
        actual = self.raw_bayer.to_3d()
        expected = np.load(picamv2_3d_path)

        np.testing.assert_array_equal(actual, expected)

    def test_get_raw_bytes(self, mocker):
        mock_stream_with_correct_prefix = io.BytesIO(b'initial-file-contents-BRCM-test-stream')

        mocker.patch.object(self.raw_bayer, '_get_raw_block_size').return_value = 16

        actual = self.raw_bayer._get_raw_bytes(
            byte_stream=mock_stream_with_correct_prefix
        )

        assert actual == b'BRCM-test-stream'

    def test_get_raw_bytes__raises_if_missing_prefix(self, mocker):
        with pytest.raises(ValueError):
            mock_stream_with_missing_prefix = io.BytesIO(b'initial-file-contents-test-stream')

            mocker.patch.object(self.raw_bayer, '_get_raw_block_size').return_value = 16
            self.raw_bayer._get_raw_bytes(
                byte_stream=mock_stream_with_missing_prefix
            )


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

        np.testing.assert_array_equal(actual, expected)

    def test_integration(self):
        bayer_array = np.load(picamv2_BGGR_bayer_array_path)
        actual = module.bayer_array_to_3d(bayer_array, BayerOrder.BGGR)

        expected = np.load(picamv2_3d_path)

        np.testing.assert_array_equal(actual, expected)
