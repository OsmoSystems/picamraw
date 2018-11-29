#!/usr/bin/env python
from setuptools import find_packages, setup


with open('README.md', 'r') as fh:
    long_description = fh.read()


setup(
    name='picamraw',
    version='1.0.0',
    author='Osmo Systems',
    author_email='dev@osmobot.com',
    description='Library for extracting raw bayer data from a Raspberry Pi JPEG+RAW file',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://www.github.com/osmosystems/picamraw',
    packages=find_packages(),
    python_requires='~=3.6',
    install_requires=[
        'numpy',
    ],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Multimedia :: Graphics :: Capture :: Digital Camera',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=[
        'Raspberrypi',
        'camera',
        'RAW',
        'bayer'
    ]
)
