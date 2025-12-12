from setuptools import find_packages
from setuptools import setup

setup(
    name='lunar_nav',
    version='0.0.0',
    packages=find_packages(
        include=('lunar_nav', 'lunar_nav.*')),
)
