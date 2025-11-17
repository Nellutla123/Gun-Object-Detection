"""Setup script for the Guns Object Detection System package."""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Guns Object Detection",
    version="0.1",
    author="Kabyik",
    packages=find_packages(),
    install_requires = requirements,
)