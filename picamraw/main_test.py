import pkg_resources
from unittest.mock import sentinel, MagicMock

import numpy as np
import pytest

from .constants import BayerOrder, PiCameraVersion
from . import main as module

picamv2_jpeg_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2.jpeg')
picamv2_BGGR_bayer_array_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2_BGGR_bayer_array.npy')
picamv2_3d_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2_3d.npy')
picamv2_rgb_path = pkg_resources.resource_filename(__name__, 'test_fixtures/picamv2_rgb.npy')


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
        (camera_version_enum, sensor_mode)
        for camera_version_enum in PiCameraVersion
        for sensor_mode in range(0, 8)
    ])
    def test_get_raw_block_size__has_values_for_all_camera_versions_and_sensor_modes(self, camera_version, sensor_mode):
        actual = module._get_raw_block_size(camera_version, sensor_mode)
        assert actual is not None


class TestExtractRawFromJpeg:
    bayer_array, bayer_order = module.extract_raw_from_jpeg(
        filepath=picamv2_jpeg_path,
        camera_version=PiCameraVersion.V2,
        sensor_mode=0
    )

    def test_extracts_bayer_order(self):
        assert self.bayer_order == BayerOrder.BGGR

    def test_extracts_raw_data(self):
        # Spot-check some known pixels
        assert self.bayer_array[0][0] == 91
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


class TestBayerArrayToRGB:
    @pytest.mark.parametrize('bayer_order,expected', [
        (BayerOrder.RGGB, np.array([
            [[1, 2.5, 5]],
        ])),
        (BayerOrder.GBRG, np.array([
            [[3, 3, 2]],
        ])),
        (BayerOrder.BGGR, np.array([
            [[5, 2.5, 1]],
        ])),
        (BayerOrder.GRBG, np.array([
            [[2, 3, 3]],
        ])),
    ])
    def test_splits_to_rgb_using_bayer_order(self, bayer_order, expected):
        bayer_array = np.array([
            [1, 2],
            [3, 5],
        ])
        actual = module.bayer_array_to_rgb(bayer_array, bayer_order)

        np.testing.assert_array_equal(actual, expected)

    def test_uneven_shape_raises(self):
        bayer_array = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])

        # pytest uses regex to match error messages
        expected_error_message = r'Incoming data is the wrong shape: width \(3\) is not a multiple of 2'
        with pytest.raises(ValueError, match=expected_error_message):
            module.bayer_array_to_rgb(bayer_array, BayerOrder.RGGB)

    def test_integration(self):
        bayer_array = np.load(picamv2_BGGR_bayer_array_path)
        actual = module.bayer_array_to_rgb(bayer_array, BayerOrder.BGGR)

        expected = np.load(picamv2_rgb_path)

        np.testing.assert_array_equal(actual, expected)


class TestUnpack10BitValues:
    def test_unpack_10bit_values(self):
        # In the camera's RAW format, every 5 bytes in the source data contains the high 8-bits of
        # 4 values followed by the low 2-bits of 4 values packed into 5th byte.
        # Spec: https://linuxtv.org/downloads/v4l-dvb-apis-new/uapi/v4l/pixfmt-srggb10p.html
        input_five_byte_set = [
            # Note: In python, "0b" appears to the left of any binary value.
            0b11111111,
            0b10010010,
            0b01001001,
            0b00000000,
            # The fifth byte is made up of the 2-bit sets that will be paired with the first 4 bytes
            # In this test those 2-bit sets will be
            # 0b11 for byte 3, 0b10 for byte 2, 0b01 for byte 1, 0b00 for byte 0
            # Mash those together and you get...
            0b11100100
        ]
        # Those 5 bytes are converted to 10-bit values
        # Each 10-bit output value will be made up of 8 bits from one of the first 4 input bytes
        # and 2 bits from the 5th input byte.
        # The spec linked above determines which bits end up where, but let's also lay it out inline:
        expected_ten_bit_outputs = [
            # The rightmost (least significant) 2-bits of the fifth source byte (0b00 in this case)
            # should end up as the low bits of the first input byte
            0b1111111100,
            # The second input byte gets the next 2 least significant bits (0b01) from the fifth byte
            0b1001001001,
            # The third byte gets the next 2 (0b10)
            0b0100100110,
            # And finally the fourth byte gets the leftmost, most-significant 2 bits (0b11)
            0b0000000011,
        ]

        # In the image array, there are multiple sets of these bytes to unpack, in a 2-dimensional array
        # Let's just do a basic job of simulating that by repeating the same values a few times.
        mock_pixel_bytes_2d = np.array(
            [
                input_five_byte_set * 2,
                input_five_byte_set * 2,
            ],
            dtype=np.uint8,
        )

        expected = np.array(
            [
                expected_ten_bit_outputs * 2,
                expected_ten_bit_outputs * 2,
            ],
            dtype=np.uint16,
        )

        actual = module._unpack_10bit_values(mock_pixel_bytes_2d)

        np.testing.assert_array_equal(actual, expected)

    def test_unpack_10bit_values__correct_shape_doesnt_raise(self):
        mock_pixel_bytes_2d = np.zeros((10, 25)).astype(np.uint8)
        module._unpack_10bit_values(mock_pixel_bytes_2d)

    def test_unpack_10bit_values__incorrect_shape_raises(self):
        # pytest uses regex to match error messages
        expected_error_message = r'Incoming data is the wrong shape: width \(26\) is not a multiple of 5'
        with pytest.raises(ValueError, match=expected_error_message):
            mock_pixel_bytes_2d = np.zeros((10, 26))
            module._unpack_10bit_values(mock_pixel_bytes_2d)


class TestPixelBytesToArray:
    def test_pixel_bytes_to_array(self):
        mock_header = MagicMock(
            height=16,
            width=16,
            padding_right=0,
            padding_down=0,
        )
        # Build up an array of length 512 to make it reshapeable into the default minimum 32x16 padded shape

        # This group of five bytes unpacks to [0b1001, 0b1001, 0b1001, 0b1001]:
        # all the "0b10"s get "01" added as low bits, ending as "0b1001"
        five_byte_group = [0b10, 0b10, 0b10, 0b10, 0b01010101]
        expected_output_byte = 0b1001
        # It takes 4 such sets of 5 bytes to unpack to 16 pixels of data
        # The remaining 12 bytes are padding (to get the minimum width of 32) that will be cropped
        mock_32_byte_row = five_byte_group * 4 + [0] * 12
        mock_1D_pixel_array = np.array(mock_32_byte_row * 16, dtype=np.uint8)

        # After unpacking, I expect all values in the array to be "9"
        expected = np.ones((16, 16), dtype=np.uint16) * expected_output_byte

        actual = module._pixel_bytes_to_array(mock_1D_pixel_array, mock_header)

        np.testing.assert_array_equal(actual, expected)


# Integration test using a known image
class TestPiRawBayer:
    def test_extracts_raw_data(self):
        raw_bayer = module.PiRawBayer(
            filepath=picamv2_jpeg_path,
            camera_version=PiCameraVersion.V2,
            sensor_mode=0
        )

        assert raw_bayer.bayer_order == BayerOrder.BGGR

        # Spot-check some known pixels
        assert raw_bayer.bayer_array[0][0] == 91
        assert raw_bayer.bayer_array[-1][-1] == 65

        # Compare to full np array
        actual = raw_bayer.bayer_array
        expected = np.load(picamv2_BGGR_bayer_array_path)

        np.testing.assert_array_equal(actual, expected)

    def test_rgb_array_property(self, mocker):
        mocker.patch.object(module, 'bayer_array_to_rgb').return_value = sentinel.rgb_array

        raw_bayer = module.PiRawBayer(
            filepath=picamv2_jpeg_path,
            camera_version=PiCameraVersion.V2,
            sensor_mode=0
        )

        assert raw_bayer.to_rgb() == sentinel.rgb_array

    def test_array_3d_property(self, mocker):
        mocker.patch.object(module, 'bayer_array_to_3d').return_value = sentinel.array_3d

        raw_bayer = module.PiRawBayer(
            filepath=picamv2_jpeg_path,
            camera_version=PiCameraVersion.V2,
            sensor_mode=0
        )

        assert raw_bayer.to_3d() == sentinel.array_3d
