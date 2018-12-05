import ctypes

import numpy as np

from .constants import PiCameraVersion, BayerOrder, BAYER_ORDER_TO_RGB_CHANNEL_COORDINATES
from .resolution import PiResolution


class PiRawBayer:
    ''' Extracts the raw 10-bit bayer data from a Raspberry Pi camera JPEG+RAW file into a 16-bit numpy array

    Attrs:
        bayer_data: The raw bayer data as a 16-bit 2D numpy array
        bayer_order: A `BayerOrder` enum that indicates the bayer pattern used by the `bayer_array`
    '''
    def __init__(self, filepath, camera_version: PiCameraVersion, sensor_mode=0):
        ''' Initializing a PiRawBayer object results in extracting the raw bayer data from the provided JPEG+RAW file.

        Args:
            filepath: The full path of the JPEG+RAW image to extract raw data from
            camera_version: A `PiCameraVersion` enum representing the camera hardware version used to capture the image
            sensor_mode: Optional - defaults to 0. An integer representing the `sensor_mode` used to capture the image.
                See https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes for more information
                on sensor_modes.
        '''
        bayer_array, bayer_order = extract_raw_from_jpeg(filepath, camera_version, sensor_mode)
        self.bayer_array = bayer_array
        self.bayer_order = bayer_order

    def to_3d(self):
        '''
        Returns: A 16-bit 3D numpy array. This array has the same 2D dimensions as the input `bayer_array`, but pulls
            each pixel out into either the R, G, or B channel in the 3rd dimension. Thus, each "pixel" in the output
            array will be [R, G, B] where 2 of R, G, and B are 0 and the other contains a value from bayer_array.
            It determines whether a given pixel is R, G, or B using the provided `bayer_order`.
        '''
        return bayer_array_to_3d(self.bayer_array, self.bayer_order)

    def to_rgb(self):
        '''
        Returns: A 16-bit 3D numpy array. Every 2x2 containing R, G1, G2, B in the original array is collapsed into a
            single [R, (G1+G2)/2, B] pixel. Thus, this array is 1/4 the size of the input `bayer_array` - both width and
            height are halved.
        '''
        return bayer_array_to_rgb(self.bayer_array, self.bayer_order)


BROADCOM_BAYER_ORDER_TO_ENUM = {
    0: BayerOrder.RGGB,
    1: BayerOrder.GBRG,
    2: BayerOrder.BGGR,
    3: BayerOrder.GRBG,
}

# Byte offsets for header (metadata) and pixel data within the raw bayer data
HEADER_BYTE_OFFSET = 176
PIXEL_BYTE_OFFSET = 32768


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


def extract_raw_from_jpeg(filepath, camera_version, sensor_mode):
    ''' Extracts the raw 10-bit bayer data from a Raspberry Pi camera JPEG+RAW file into a 16-bit numpy array

    Args:
        filepath: The full path of the JPEG+RAW image to extract raw data from
        camera_version: A `PiCameraVersion` enum representing the camera hardware version used to capture the image
        sensor_mode: Optional - defaults to 0. An integer representing the `sensor_mode` used to capture the image.
            See https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes for more information
            on sensor_modes.

    Returns: (bayer_data, bayer_order)
        bayer_data: The raw bayer data as a 16-bit 2D numpy array
        bayer_order: A `BayerOrder` enum that indicates the bayer pattern used by the `bayer_array`
    '''

    with open(filepath, mode='rb') as file:
        jpeg_data_as_bytes = file.read()

    raw_bytes = _get_raw_bayer_bytes(jpeg_data_as_bytes, camera_version, sensor_mode)

    # Extract header (metadata) and pixel data using known byte offsets
    header = BroadcomRawHeader.from_buffer_copy(raw_bytes, HEADER_BYTE_OFFSET)

    # Extract the 1D array of 8-bit (1-byte) values that collectively represent the pixel data
    # Note: pixel data is actually 10-bits per pixel, but is packed into 8-bit values
    pixel_bytes = np.frombuffer(raw_bytes, dtype=np.uint8, offset=PIXEL_BYTE_OFFSET)

    bayer_array = _pixel_bytes_to_array(pixel_bytes, header)
    bayer_order = BROADCOM_BAYER_ORDER_TO_ENUM[header.bayer_order]

    return bayer_array, bayer_order


def _guard_attribute_is_a_multiple_of(attribute_name, attribute_value, multiple):
    if not attribute_value % multiple == 0:
        raise ValueError(
            'Incoming data is the wrong shape: {attribute_name} ({attribute_value}) is not a multiple of {multiple}'
            .format(**locals())
        )


def bayer_array_to_3d(bayer_array, bayer_order: BayerOrder):
    ''' Convert the 2D `bayer_array` to a 3D RGB array, in which each value in the original 2D array is
        moved to one of the three R,G, or B channels.

    Args:
        bayer_array: the 2D bayer array to convert
        bayer_order: A `BayerOrder` enum that indicates the bayer pattern used by the `bayer_array`

    Returns:
        A 3D numpy array. This array has the same 2D dimensions as the input `bayer_array`, but pulls each pixel
        out into either the R, G, or B channel in the 3rd dimension. Thus, each "pixel" in the output array will be
        [R, G, B] where 2 of R, G, and B are 0 and the other contains a value from bayer_array.
        It determines whether a given pixel is R, G, or B using the provided `bayer_order`.

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
    '''

    # Prepare an empty 3D array that has the same 2D dimensions as the bayer array
    array_3d = np.zeros(bayer_array.shape + (3,), dtype=bayer_array.dtype)

    ((ry, rx), (gy, gx), (Gy, Gx), (by, bx)) = BAYER_ORDER_TO_RGB_CHANNEL_COORDINATES[bayer_order]

    # Keeps pixels in the same 2D location, but separates into RGB channels based on bayer order
    # Increment by 2: a given color will be in every other column in every other row in the bayer array
    # "Seed" this incrementating using the (x,y) coordinates of that color in the first 2x2 corner of the array
    R_CHANNEL_INDEX, G_CHANNEL_INDEX, B_CHANNEL_INDEX = [0, 1, 2]
    array_3d[ry::2, rx::2, R_CHANNEL_INDEX] = bayer_array[ry::2, rx::2]  # Red
    array_3d[gy::2, gx::2, G_CHANNEL_INDEX] = bayer_array[gy::2, gx::2]  # Green
    array_3d[Gy::2, Gx::2, G_CHANNEL_INDEX] = bayer_array[Gy::2, Gx::2]  # Green
    array_3d[by::2, bx::2, B_CHANNEL_INDEX] = bayer_array[by::2, bx::2]  # Blue

    return array_3d


def bayer_array_to_rgb(bayer_array, bayer_order: BayerOrder):
    ''' Convert the 2D `bayer_array` to a 3D RGB array, in which each value in the original 2D array is
        moved to one of the three R,G, or B channels.

    Args:
        bayer_array: the 2D bayer array to convert
        bayer_order: A `BayerOrder` enum that indicates the bayer pattern used by the `bayer_array`

    Returns:
        A 3D numpy array. Every 2x2 containing R, G1, G2, B in the original array is collapsed into a single
        [R, (G1+G2)/2, B] pixel. Thus, this array is 1/4 the size of the input `bayer_array` - both width and height
        are halved.

    Example:
        bayer_array_to_rgb(
            bayer_array=np.array([
                [1, 2],  # R  G1
                [3, 4],  # G2 B
            ]),
            bayer_order=BayerOrder.RGGB
        )
        >>> np.array([
            [[1, 2.5, 4]],  # R, (G1+G2)/2, B
        ])
    '''
    # Initialize a new array that is the expected shape of 1/2 width and height dimensions
    original_height = bayer_array.shape[0]
    original_width = bayer_array.shape[1]

    _guard_attribute_is_a_multiple_of('width', original_width, 2)
    _guard_attribute_is_a_multiple_of('height', original_height, 2)

    rgb_array = np.zeros((int(original_height/2), int(original_width/2), 3))

    ((ry, rx), (gy, gx), (Gy, Gx), (by, bx)) = BAYER_ORDER_TO_RGB_CHANNEL_COORDINATES[bayer_order]

    # Increment by 2: a given color will be in every other column in every other row in the bayer array
    # "Seed" this incrementating using the (x,y) coordinates of that color in the first 2x2 corner of the array
    R_CHANNEL_INDEX, G_CHANNEL_INDEX, B_CHANNEL_INDEX = [0, 1, 2]
    rgb_array[:, :, R_CHANNEL_INDEX] = bayer_array[ry::2, rx::2]
    rgb_array[:, :, G_CHANNEL_INDEX] = (bayer_array[gy::2, gx::2] + bayer_array[Gy::2, Gx::2])/2
    rgb_array[:, :, B_CHANNEL_INDEX] = bayer_array[by::2, bx::2]

    return rgb_array


def _pixel_bytes_to_array(pixel_bytes, header):
    ''' Convert the 1D array of 8-bit values ("packed" 10-bit values) to a 2D array of 10-bit values. Every 5 bytes
        contains the high 8-bits of 4 values followed by the low 2-bits of 4 values packed into the fifth byte
    '''
    # Reshape and crop the data. The crop's width is multiplied by 5/4 to deal with the packed 10-bit format;
    # the shape's width is calculated in a similar fashion but with padding included
    crop = PiResolution(
        header.width * 5 // 4,
        header.height
    )

    # Honestly I don't entirely grok this code - it seems to imply that the raw data comes with extra (random?)
    # bytes that form padding along the right and bottom edges of the image. To get the actual image data, this padding
    # must be accounted for when reshaping the 1D array of bytes into a 2D array, and then discarded by cropping to the
    # actual image resolution.
    shape = PiResolution(
        (((header.width + header.padding_right) * 5) + 3) // 4,
        (header.height + header.padding_down)
    ).pad()

    pixel_bytes_2d = pixel_bytes.reshape((shape.height, shape.width))[:crop.height, :crop.width]

    array = _unpack_10bit_values(pixel_bytes_2d)

    return array


def _unpack_10bit_values(pixel_bytes_2d):
    ''' Unpack 10-bit values; every 5 bytes contains the high 8-bits of 4 values followed by the low 2-bits of
        4 values packed into the fifth byte
    '''
    # This code assumes that bytes in each row come in sets of 5. If the width is not a multiple of 5, it breaks
    width = pixel_bytes_2d.shape[1]
    _guard_attribute_is_a_multiple_of('width', width, 5)

    # Bitshift left by two to make room for the low 2-bits
    data = pixel_bytes_2d.astype(np.uint16) << 2

    # In each row, split up every 5th byte and unpack it into the low 2-bits of the four preceding bytes
    for byte in range(4):
        data[:, byte::5] |= ((data[:, 4::5] >> ((4 - byte) * 2)) & 3)

    # Set up a new array with the correct shape: same height but width reduced by 4/5
    array = np.zeros(
        (data.shape[0], data.shape[1] * 4 // 5), dtype=np.uint16)

    # Copy over the properly-unpacked four out of every five bytes
    for i in range(4):
        array[:, i::4] = data[:, i::5]

    return array


# Size of the block of the raw bayer data (in bytes) within the full JPEG+RAW file
RAW_BLOCK_SIZE_BY_VERSION_AND_MODE = {
    PiCameraVersion.V1: {
        0: 6404096,
        1: 2717696,
        2: 6404096,
        3: 6404096,
        4: 1625600,
        5: 1233920,
        6: 445440,
        7: 445440,
    },
    PiCameraVersion.V2: {
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


def _get_raw_block_size(camera_version, sensor_mode):
    return RAW_BLOCK_SIZE_BY_VERSION_AND_MODE[camera_version][sensor_mode]


def _get_raw_bayer_bytes(jpeg_data_as_bytes, camera_version, sensor_mode):
    ''' Extract the bytes that represent the raw bayer data from the contents of a JPEG+RAW file '''
    # The raw bayer data is at the end of the file, so extract an appropriately-sized block of data from the end
    raw_block_size = _get_raw_block_size(camera_version, sensor_mode)
    raw_bytes = jpeg_data_as_bytes[-raw_block_size:]

    # Bayer data should start with 'BCRM'
    if raw_bytes[:4] != b'BRCM':
        raise ValueError('Unable to locate Bayer data at end of buffer')

    return raw_bytes
