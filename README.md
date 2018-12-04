# picamraw
Library for extracting raw bayer data from a Raspberry Pi JPEG+RAW file.

Installable from [PyPI](https://pypi.org/project/picamraw/); usable without camera hardware present.


# Usage example
## Extract raw bayer array
```python
from picamraw import PiRawBayer, PiCameraVersion

raw_bayer = PiRawBayer(
    filepath='path/to/image.jpeg',  # A JPEG+RAW file, e.g. an image captured using raspistill with the "--raw" flag
    camera_version=PiCameraVersion.V2,
    sensor_mode=0
)
raw_bayer.bayer_array   # A 16-bit 2D numpy array of the bayer data
raw_bayer.bayer_order   # A `BayerOrder` enum that describes the arrangement of the R,G,G,B pixels in the bayer_array
raw_bayer.to_rgb()      # A 16-bit 3D numpy array of bayer data collapsed into RGB channels (see docstring for details).
raw_bayer.to_3d()       # A 16-bit 3D numpy array of bayer data split into RGB channels (see docstring for details).
```


# Testing

This package is tested using [`tox`](https://tox.readthedocs.io/).
To run tests, simply `pip install tox` and then run `tox`.

Note: this code has only been tested against an image captured with camera version V2 and sensor_mode 0.


# Attribution
This library was forked from the [PiCamera](https://github.com/waveform80/picamera) package and heavily modified.
