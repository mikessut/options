from setuptools import setup

setup(name='Options',
      version='0.1',
      description='Black Scholes Option Pricing',
      packages=['options'],
      install_requires=['arch', 'numpy', 'scipy', 'holidays']
      )