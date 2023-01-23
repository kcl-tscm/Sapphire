from setuptools import setup, find_packages
import os
import re
import sys

with open('README') as fd:
    long_description = fd.read()
    
setup(
    name='Sapphire',
    version='1.0.0',
    url='https://github.com/kcl-tscm/Sapphire.git',
    description='A pythonic post-processing environment for the analysis on NanoAlloys',
    author='Robert Michael Jones',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author_email='Robert.M.Jones@kcl.ac.uk',
    package_dir={'': 'main'},
    packages=find_packages(where="main"),
    install_requires=[
        'numpy',
        'networkx',
        'numba',
        'networkx',
        'pandas',
        'ase',
        'scipy',
        'sklearn', 
        'ruptures',
        'tensorflow',
        'pygdm2',
        'networkx',
        'scikit-learn'
    ],
    extras_require={'plotting': ['matplotlib', 'jupyter', 'seaborn']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    long_description=long_description
)

