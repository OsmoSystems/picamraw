#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='picamraw',
    version='0.0.1',
    author='Osmo Systems',
    author_email='dev@osmobot.com',
    description='Library for extracting raw bayer data from a Raspberry Pi JPEG+RAW file',
    url='https://www.github.com/osmosystems/picamraw.git',
    packages=find_packages(),
    entry_points={},
    install_requires=[
        'numpy',
    ],
    include_package_data=True
)
