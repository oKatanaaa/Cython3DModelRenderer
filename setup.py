import setuptools
import os


meta = {}
with open(os.path.join('crender', '__version__.py')) as f:
    exec(f.read(), meta)


setuptools.setup(
    name=meta['__title__'],
    packages=setuptools.find_packages(),
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
