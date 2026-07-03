#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

__author__ = ['Irene Ballantini']
__email__ = ['irene.ballantini@studio.unibo.it']

PACKAGE_NAME = 'nsclc_survival'
PACKAGE_VERSION = '0.0.1'
DESCRIPTION = 'NSCLC Radiomics: Survival Time prediction using CT-extracted features and clinical data.'
AUTHOR = 'Irene Ballantini'
EMAIL = 'irene.ballantini@studio.unibo.it'
REQUIRES_PYTHON = '>=3.8, <3.11'
URL = 'https://github.com/irene-ballantini/nsclc_survival'
DOWNLOAD_URL = URL

setup(
  name=PACKAGE_NAME,
  version=PACKAGE_VERSION,
  description=DESCRIPTION,
  author=AUTHOR,
  author_email=EMAIL,
  python_requires=REQUIRES_PYTHON,
  install_requires=[
      "pydicom",
    "rt-utils",
    "SimpleITK",
    "tcia-utils",
    "numpy",
    "pandas",
    "scikit-learn",
    "lifelines",
    "scikit-survival",
    "torch",
    "pyradiomics",
    "matplotlib",
    "ruamel.yaml",
  ],
  url=URL,
  download_url=DOWNLOAD_URL,
  setup_requires=[],
  packages=[
    PACKAGE_NAME,
  ],
  package_data={
    PACKAGE_NAME: [],
  },
  include_package_data=True,
  platforms='any',
  classifiers=[
    'Natural Language :: English',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: Implementation :: CPython',
    'Programming Language :: Python :: Implementation :: PyPy'
  ],
  entry_points={'console_scripts': [
    'nsclc_survival = nsclc_survival.__main__:main',
    ],
  },
)
