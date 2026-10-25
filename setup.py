"""CN: 核心发行包不编译二进制后端。EN: The core distribution does not build binary backends."""

from setuptools import setup


setup(ext_modules=[], cmdclass={})
