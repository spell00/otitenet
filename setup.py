#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 2021
@author: Simon Pelletier
"""


from setuptools import setup


setup(
    name='otitenet',
    version='0.1',
    packages=[
        "otitenet",
        "otitenet.app",
        "otitenet.data",
        "otitenet.infer",
        "otitenet.logging",
        "otitenet.ml",
        "otitenet.models",
        "otitenet.train",
        "otitenet.utils",
        "RandConv",
        "RandConv.lib",
        "RandConv.lib.networks",
    ],
    url='https://github.com/spell0/otitenet',
    license='MIT',
    author='Simon Pelletier',
    author_email='simonjpelletier@gmail.com',
    description='A description of your project goes here.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
