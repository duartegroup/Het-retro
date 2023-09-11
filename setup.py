#!/usr/bin/env python
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rxn-opennmt-py",
    description="Fork of OpenNMT-py for use in RXN projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="1.1.5",
    packages=find_packages(),
    project_urls={
        "Documentation": "http://opennmt.net/OpenNMT-py/",
        "Source": "https://github.com/rxn4chemistry/OpenNMT-py/",
    },
    install_requires=[
        "six",
        "tqdm",
        "torch>=1.2,<1.6",  # versions 1.6 or higher fail with the current fork (i.e. fork from the official repo in March 2020)
        "torchtext==0.4.0",
        "future",
        "configargparse",
    ],
    entry_points={
        "console_scripts": [
            "onmt_server=onmt.bin.server:main",
            "onmt_train=onmt.bin.train:main",
            "onmt_translate=onmt.bin.translate:main",
            "onmt_preprocess=onmt.bin.preprocess:main",
        ],
    },
)
