#!/usr/bin/python3
from setuptools import find_packages, setup

setup(
    name="k_armed_test_bed",
    version="0.0.1",
    description="K Armed Test Bed",
    author="Alejandro Alfonso",
    author_email="alejandroalfonso1994@gmail.com",
    packages=find_packages(exclude=['tests']),
    install_requires=[r.strip() for r in open("requirements.txt").readlines()]
)
