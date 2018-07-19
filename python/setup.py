import os
import sys
from setuptools import setup

install_requires = [
    'numpy>=1.14.0',
    'scipy>=1.1.0',
    'future',
    'graphtools>=0.1.8.1',
]

version_py = os.path.join(os.path.dirname(
    __file__), 'meld-convex', 'version.py')
version = open(version_py).read().strip().split(
    '=')[-1].replace('"', '').strip()

if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version 2.7 or >=3.5 required.")

readme = open('README.md').read()

setup(name='meld-convex',
      version=version,
      description='MELD',
      author='Daniel Burkhardt, Krishnaswamy Lab, Yale University',
      author_email='daniel.burkhardt@yale.edu',
      packages=['meld-convex', ],
      license='GNU General Public License Version 2',
      install_requires=install_requires,
      long_description=readme,
      url='https://github.com/KrishnaswamyLab/MELD',
      download_url="https://github.com/KrishnaswamyLab/MELD/archive/v{}.tar.gz".format(
          version),
      keywords=['big-data',
                'manifold-learning',
                'computational-biology']
      )

# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
