#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 2021
@author: Simon Pelletier
"""


from setuptools import find_namespace_packages, setup


setup(
    name='otitenet',
    version='0.3.0',
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src", include=["otitenet", "otitenet.*"]),
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
    python_requires='>=3.11',
)
