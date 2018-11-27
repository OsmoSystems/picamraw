import pkg_resources
from unittest.mock import sentinel

import numpy as np
import pytest

from .constants import BayerOrder, PiCameraVersion
from . import main as module

picamv2_jpeg_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2.jpeg')
picamv2_BGGR_bayer_array_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2_BGGR_bayer_array.npy')
picamv2_3d_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2_3d.npy')


class TestGetRawBayerBytes:
    def test_get_raw_bayer_bytes(self, mocker):
        mock_stream_with_correct_prefix = b'initial-file-contents-BRCM-test-stream'

        mocker.patch.object(module, '_get_raw_block_size').return_value = 16

        actual = module._get_raw_bayer_bytes(
            jpeg_data_as_bytes=mock_stream_with_correct_prefix,
            camera_version=sentinel.camera_version,
            sensor_mode=sentinel.sensor_mode,
        )

        assert actual == b'BRCM-test-stream'

    def test_get_raw_bayer_bytes__raises_if_missing_prefix(self, mocker):
        with pytest.raises(ValueError):
            mock_stream_with_missing_prefix = b'initial-file-contents-test-stream'

            mocker.patch.object(module, '_get_raw_block_size').return_value = 16
            module._get_raw_bayer_bytes(
                jpeg_data_as_bytes=mock_stream_with_missing_prefix,
                camera_version=sentinel.camera_version,
                sensor_mode=sentinel.sensor_mode,
            )


class TestGetRawBlockSize:
    @pytest.mark.parametrize('camera_version,sensor_mode', [
        (camera_version_enum.value, sensor_mode)
        for camera_version_enum in PiCameraVersion
        for sensor_mode in range(0, 8)
    ])
    def test_get_raw_block_size__has_values_for_all_camera_versions_and_sensor_modes(self, camera_version, sensor_mode):
        actual = module._get_raw_block_size(camera_version, sensor_mode)
        assert actual is not None


class TestExtractRawFromJpeg:
    bayer_array, bayer_order = module.extract_raw_from_jpeg(
        filepath=picamv2_jpeg_path,
        camera_version=PiCameraVersion.V2.value,
        sensor_mode=0
    )

    def test_extracts_bayer_order(self):
        assert self.bayer_order == BayerOrder.BGGR

    def test_extracts_raw_data(self):
        # Spot-check some known pixels
        assert self.bayer_array[0][0] == 88
        assert self.bayer_array[-1][-1] == 65

        # Compare to full np array
        actual = self.bayer_array
        expected = np.load(picamv2_BGGR_bayer_array_path)

        np.testing.assert_array_equal(actual, expected)


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
