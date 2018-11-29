from collections import namedtuple


class PiResolution(namedtuple('PiResolution', ('width', 'height'))):
    '''
    A :func:`~collections.namedtuple` derivative which represents a resolution
    with a `width` and `height`.

    Attrs:
        width: The width of the resolution in pixels
        height: The height of the resolution in pixels
    '''

    __slots__ = ()  # workaround python issue #24931

    def pad(self, pad_width=32, pad_height=16):
        '''
        Returns the resolution padded up to the nearest multiple of *pad_width*
        and *pad_height* which default to 32 and 16 respectively (the camera's
        native block size for most operations).

        Args:
            pad_width: Optional - defaults to 32. The multiple to pad the width to.
            pad_height: Optional - defaults to 16. The multiple to pad the height to.

        Returns:
            A `PiResolution` object with the padding applied
        '''
        return PiResolution(
            width=((self.width + (pad_width - 1)) // pad_width) * pad_width,
            height=((self.height + (pad_height - 1)) // pad_height) * pad_height,
        )

    def __str__(self):
        return '%dx%d' % (self.width, self.height)
