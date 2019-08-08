# Copyright (C) 2019 Krishnaswamy Lab, Yale University

import os
import sys
from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.14.0',
    'scipy>=1.1.0,!=1.3.0',
    'graphtools>=0.1.8.1',
    'pandas<0.24',
    'scprep',
    'pygsp'
]

test_requires = [
    'nose',
    'nose2',
    'coverage',
    'coveralls',
    'scikit-learn',
]

doc_requires = [
    'sphinx<=1.8.5',
    'sphinxcontrib-napoleon',
    'autodocsumm',
]

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >=3.5 required.")

version_py = os.path.join(os.path.dirname(
    __file__), 'meld', 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

readme = open('README.md').read()

setup(name='meld',
      version=version,
      description='MELD',
      author='Daniel Burkhardt, Krishnaswamy Lab, Yale University',
      author_email='daniel.burkhardt@yale.edu',
      packages=find_packages(),
      license='Dual License - See LICENSE file',
      install_requires=install_requires,
      extras_require={'test': test_requires,
                      'doc': doc_requires},
      test_suite='nose2.collector.collector',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/KrishnaswamyLab/MELD',
      download_url="https://github.com/KrishnaswamyLab/MELD/archive/v{}.tar.gz".format(
          version),
      keywords=['big-data',
                'manifold-learning',
                'computational-biology'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ]
      )

# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
