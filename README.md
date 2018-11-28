# picamraw
Library for extracting raw bayer data from a Raspberry Pi JPEG+RAW file.

# Usage example
## Extract raw bayer array
```
from picamraw import PiRawBayer, PiCameraVersion

raw_bayer = PiRawBayer(
    filepath='path/to/image.jpeg',
    camera_version=PiCameraVersion.V2,
    sensor_mode=0
)
raw_bayer.bayer_array   # A 16-bit 2D numpy array of the bayer data
raw_bayer.bayer_order   # A `BayerOrder` enum that describes the arrangement of the R,G,G,B pixels in the bayer_array
```

# Testing
Note: this code has only been tested against an image captured with camera version V2 and sensor_mode 0.

# Attribution
This library was forked from the [PiCamera](https://github.com/waveform80/picamera) package and heavily modified using [this gist](https://gist.github.com/rwb27/c2ea3afb204a634f9f21a99bd8dd1541) as inspiration.
