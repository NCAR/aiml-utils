'''
    Setup file for the necessary packages to build and install holodecml.
    For ease, this can be installed from BitBucket via:
    pip3 install git+https://[USER_NAME]@https://github.com/NCAR/aiml-utils/aiml-utils.git

    It is recommeneded that you install this into a Python or Conda Virtual Environment.
'''

import codecs
import os
import re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


    
with open("README.md") as f:
    long_description = f.read()
    

with open("requirements.txt") as f:
    required_libraries = f.read().splitlines()
    

setup(
    name='aimlutils',
    version=find_version("./", "__version__.py"),
    author='AIML',
    description=('This repository contains various pieces of code that is shared across different AIML projects, as well as notebooks for blogs'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NCAR/aiml-utils',
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="",
    install_requires=required_libraries,
    packages=find_packages(exclude=['aimlutils/tests']),
#    test_suite='tests',
    zip_safe=False,
)
