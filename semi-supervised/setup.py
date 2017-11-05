from setuptools import setup, find_packages
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='semi-supervised',

    author='Jesper Wohlert',

    author_email='jesper@wohlert.nu',

    # useful: python setup.py sdist bdist_wheel upload
    version='0.0.2',

    description='PyTorch package of semi-supervised learning models',

    url='https://github.com/wohlert/semi-supervised-pytorch',

    license='MIT',

    classifiers=[
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5'
    ],

    packages=['semi-supervised'],

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=required,
)