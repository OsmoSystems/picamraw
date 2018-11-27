import io
import ctypes

import numpy as np

from .constants import PiCameraVersion, BayerOrder, BAYER_ORDER_TO_RGB_CHANNEL_COORDINATES
from .resolution import PiResolution


def bayer_array_to_3d(bayer_array, bayer_order: BayerOrder):
    ''' Convert the 2D `bayer_array` attribute to a 3D RGB array, in which each value in the original 2D array is
    moved to one of the three R,G, or B channels.

    Example:
        bayer_array_to_3d(
            bayer_array=np.array([
                [1, 2],
                [3, 4],
            ]),
            bayer_order=BayerOrder.RGGB
        )
        >>> np.array([
            [[1, 0, 0], [0, 2, 0]],
            [[0, 3, 0], [0, 0, 4]],
        ])

    Args:
        bayer_array: the 2D bayer array to convert
        bayer_order: A `BayerOrder` enum that indicates the bayer pattern used by the `bayer_array`
    '''

    # Prepare an empty 3D array that has the same 2D dimensions as the bayer array
    array_3d = np.zeros(bayer_array.shape + (3,), dtype=bayer_array.dtype)

    (
        (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
    ) = BAYER_ORDER_TO_RGB_CHANNEL_COORDINATES[bayer_order]

    # Keeps pixels in the same 2D location, but separates into RGB channels based on bayer order
    R_CHANNEL_INDEX = 0
    G_CHANNEL_INDEX = 1
    B_CHANNEL_INDEX = 2

    array_3d[ry::2, rx::2, R_CHANNEL_INDEX] = bayer_array[ry::2, rx::2]  # Red
    array_3d[gy::2, gx::2, G_CHANNEL_INDEX] = bayer_array[gy::2, gx::2]  # Green
    array_3d[Gy::2, Gx::2, G_CHANNEL_INDEX] = bayer_array[Gy::2, Gx::2]  # Green
    array_3d[by::2, bx::2, B_CHANNEL_INDEX] = bayer_array[by::2, bx::2]  # Blue

    return array_3d


class BroadcomRawHeader(ctypes.Structure):
    _fields_ = [
        ('name',          ctypes.c_char * 32),
        ('width',         ctypes.c_uint16),
        ('height',        ctypes.c_uint16),
        ('padding_right', ctypes.c_uint16),
        ('padding_down',  ctypes.c_uint16),
        ('dummy',         ctypes.c_uint32 * 6),
        ('transform',     ctypes.c_uint16),
        ('format',        ctypes.c_uint16),
        ('bayer_order',   ctypes.c_uint8),
        ('bayer_format',  ctypes.c_uint8),
    ]


class PiRawBayer():
    '''
    Extracts the raw 10-bit bayer data from a Raspberry Pi camera JPEG+RAW file into a 16-bit numpy array
    '''
    BROADCOM_BAYER_ORDER_TO_ENUM = {
        0: BayerOrder.RGGB,
        1: BayerOrder.GBRG,
        2: BayerOrder.BGGR,
        3: BayerOrder.GRBG,
    }

    # Size of the block (in bytes) of the raw bayer data within the full JPEG+RAW file
    RAW_BLOCK_SIZE_BY_VERSION_AND_MODE = {
        PiCameraVersion.V1.value: {
            0: 6404096,
            1: 2717696,
            2: 6404096,
            3: 6404096,
            4: 1625600,
            5: 1233920,
            6: 445440,
            7: 445440,
        },
        PiCameraVersion.V2.value: {
            0: 10270208,
            1: 2678784,
            2: 10270208,
            3: 10270208,
            4: 2628608,
            5: 1963008,
            6: 1233920,
            7: 445440,
        },
    }

    # Byte offsets for header (metadata) and pixel data within the raw bayer data
    HEADER_BYTE_OFFSET = 176
    PIXEL_BYTE_OFFSET = 32768

    def __init__(self, filepath, camera_version: PiCameraVersion, sensor_mode=0):
        ''' Initializing a PiRawBayer object results in extracting the raw bayer data from the provided JPEG+RAW file

        Args:
            filepath: The full path of the JPEG+RAW image to extract raw data from
            camera_version: A `PiCameraVersion` enum representing the camera hardware version used to capture the image
            sensor_mode: Optional - defaults to 0. An integer representing the `sensor_mode` used to capture the image
        '''
        super(PiRawBayer, self).__init__()

        self._header = None
        self._camera_version = camera_version
        self._sensor_mode = sensor_mode
        self._filepath = filepath

        self._extract(filepath)

    @property
    def bayer_order(self):
        ''' A `BayerOrder` enum representing the arrangement of R,G,G,B pixels in the bayer array '''
        return self.BROADCOM_BAYER_ORDER_TO_ENUM[self._header.bayer_order]

    def _get_raw_block_size(self):
        return self.RAW_BLOCK_SIZE_BY_VERSION_AND_MODE[self._camera_version][self._sensor_mode]

    def _extract(self, filepath):
        with open(filepath, mode='rb') as file:
            jpeg_data = file.read()

        byte_stream = io.BytesIO()
        byte_stream.write(jpeg_data)
        byte_stream.flush()

        raw_bytes = self._get_raw_bytes(byte_stream)

        # Extract header (metadata) and pixel data using known byte offsets
        self._header = BroadcomRawHeader.from_buffer_copy(
            raw_bytes[self.HEADER_BYTE_OFFSET:self.HEADER_BYTE_OFFSET + ctypes.sizeof(BroadcomRawHeader)]
        )

        # Extract the 1D array of 8-bit (1-byte) values that collectively represent the pixel data
        # Note: pixel data is actually 10-bits per pixel, but is packed into 8-bit values
        pixel_bytes = np.frombuffer(
            raw_bytes,
            dtype=np.uint8,
            offset=self.PIXEL_BYTE_OFFSET
        )

        self.bayer_array = self._pixel_bytes_to_array(pixel_bytes)

        byte_stream.close()

    def _get_raw_bytes(self, byte_stream):
        ''' Extract the bytes that represent the raw bayer data from the contents of a JPEG+RAW file '''
        # The raw data is at the end of the file, so extract an appropriately-sized block of data from the end
        raw_block_size = self._get_raw_block_size()
        raw_bytes = byte_stream.getvalue()[-raw_block_size:]

        # Bayer data should start with 'BCRM'
        if raw_bytes[:4] != b'BRCM':
            raise ValueError('Unable to locate Bayer data at end of buffer')

        return raw_bytes

    def _pixel_bytes_to_array(self, pixel_bytes):
        ''' Convert the 1D array of 8-bit values ("packed" 10-bit values) to a 2D array of 10-bit values. Every 5 bytes
            contains the high 8-bits of 4 values followed by the low 2-bits of 4 values packed into the fifth byte
        '''
        # Reshape and crop the data. The crop's width is multiplied by 5/4 to
        # deal with the packed 10-bit format; the shape's width is calculated
        # in a similar fashion but with padding included (which involves
        # several additional padding steps)
        crop = PiResolution(
            self._header.width * 5 // 4,
            self._header.height
        )

        shape = PiResolution(
            (((self._header.width + self._header.padding_right) * 5) + 3) // 4,
            (self._header.height + self._header.padding_down)
        ).pad()

        pixel_bytes_2d = pixel_bytes.reshape((shape.height, shape.width))[:crop.height, :crop.width]

        # Unpack 10-bit values; every 5 bytes contains the high 8-bits of 4 values followed by the low 2-bits of
        # 4 values packed into the fifth byte

        data = pixel_bytes_2d.astype(np.uint16) << 2
        for byte in range(4):
            data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 3)

        array = np.zeros(
            (data.shape[0], data.shape[1] * 4 // 5), dtype=np.uint16)

        for i in range(4):
            array[:, i::4] = data[:, i::5]

        return array

    def to_3d(self):
        return bayer_array_to_3d(self.bayer_array, self.bayer_order)

    def to_rgb(self):
        # TODO: collapse 2x2 into single pixel, averaging green channel
        pass
