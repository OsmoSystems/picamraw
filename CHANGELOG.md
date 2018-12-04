# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org).


## [1.1.0] - 2018-12-03
### Added
- `PiRawBayer.to_rgb()` method that collapses the extracted raw bayer 2D array to a 3D RGB array, with no demosaicing other than averaging the two green channels.


## [1.0.0] - 2018-11-29
### Added
- `PiRawBayer` class that extracts raw bayer data out of a JPEG+RAW file.
- `PiRawBayer.to_3d()` method that splits the extracted raw bayer 2D array into a 3D RGB array - moving each pixel into either the R, G, or B channel according to the `PiRawBayer.bayer_order` attribute.