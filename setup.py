from distutils.core import setup
from Cython.Build import cythonize
from setuptools import find_packages
import os


meta = {}
with open(os.path.join('crender', '__version__.py')) as f:
    exec(f.read(), meta)

print('Found packages:', find_packages())

setup(
    ext_modules=cythonize('**/*.pyx'),
    name=meta['__title__'],
    packages=find_packages(),
    version=meta['__title__'],
    description=meta['__description__'],
    long_description='...',
    author=meta['__author__'],
    author_email=meta['__contact__'],
    url=meta['__contact__'],
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[]
)
