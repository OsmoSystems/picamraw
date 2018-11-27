# picamraw
Library for extracting raw bayer data from a Raspberry Pi JPEG+RAW file

# Usage example
## Extract raw bayer array
```
from picamraw.main import PiRawBayer
from picamraw.constants import PiCameraVersion

raw_bayer = PiRawBayer(
    filepath='path/to/image.jpeg',
    camera_version=PiCameraVersion.V2,
    sensor_mode=0
)
raw_bayer.bayer_array   # A 16-bit 2D numpy array of the bayer data
raw_bayer.bayer_order   # A `BayerOrder` enum that describes the arrangement of the R,G,G,B pixels in the bayer_array
```
