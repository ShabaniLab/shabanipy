#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    print('Please install or upgrade setuptools or pip to continue')
    sys.exit(1)

sys.path.insert(0, os.path.abspath('.'))
from shabanipy.version import __version__


def read(filename):
    with open(filename, 'rb') as f:
        return f.read().decode('utf8')


requirements = ['numpy', 'h5py', 'matplotlib', 'lmfit', 'scipy', 'pandas']


setup(name='shabanipy',
      description='Data analysis tools used in Shabani Lab',
      version=__version__,
      maintainer='Matthieu Dartiailh',
      maintainer_email='m.dartiailh@gmail.com',
      url='https://github.com/ShabaniLab/shabanipy',
      license='MIT License',
      python_requires='>=3.9',
      install_requires=requirements,
      packages=find_packages(),
      platforms="Linux, Windows,Mac",
      use_2to3=False,
      zip_safe=False)
