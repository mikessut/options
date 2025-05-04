"""
To build cython:
python setup.py build_ext --inplace

"""

from setuptools import setup
from Cython.Build import cythonize
import numpy


setup(name='Options',
      version='0.1',
      description='Black Scholes Option Pricing',
      packages=['options'],
      install_requires=['arch', 'numpy', 'scipy', 'holidays'],
      ext_modules=cythonize('options/cython_rec.pyx'),
      include_dirs=[numpy.get_include()],
      include_package_data=True,
      package_data={'options': ['override.yml']}
      )